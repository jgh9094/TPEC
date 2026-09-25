# Base EA framework

This document explains how the abstract base classes cooperate to define one *evolutionary
algorithm (EA)* run. Nothing here is runnable on its own — [base_ea.py](Source/Base/base_ea.py),
[individual.py](Source/Base/individual.py), [model_param_space.py](Source/Base/model_param_space.py),
and [archive.py](Source/Base/archive.py) are all abstract. The concrete `ea.py` files under
[Source/HPO](Source/HPO) and [Source/CASH](Source/CASH) subclass them and fill in the abstract
hooks. Read this file to understand the *contract* every derived EA follows; read the domain
`ea.md` files for what each domain plugs into that contract.

There is no `Source/Base/ea.py` — the base entry point is the class `BaseEA` in
[base_ea.py](Source/Base/base_ea.py). "Execution of ea.py" below means "a derived EA's
`evolve()` loop, running against the base machinery."

---

## The four collaborators

| File | Class | Role |
|------|-------|------|
| [base_ea.py](Source/Base/base_ea.py) | `BaseEA(ABC)` | The orchestrator: owns the data, the RNG, the population, CV folds, and the generational skeleton. |
| [individual.py](Source/Base/individual.py) | `Individual(ABC)` | One candidate solution — a genotype plus its measured performances and provenance tags. |
| [model_param_space.py](Source/Base/model_param_space.py) | `ModelParams(ABC)` + `DataContext` | A single component's hyperparameter search space, plus the sampling/mutation operators over it. |
| [archive.py](Source/Base/archive.py) | `Archive(ABC)` + `ArchiveEntry` | The immutable log of every evaluated individual — the single source of history. |

These four never import a concrete domain. They talk to each other only through the abstract
interface, so the same skeleton drives both single-model HPO and full-pipeline CASH.

---

## `BaseEA` — the orchestrator

`BaseEA.__init__` ([base_ea.py:28](Source/Base/base_ea.py#L28)) stores the run-level knobs
(`seed`, `pop_size`, `cores`, `mut_prob`, `mut_var`, `crossover_prob`, `classification`) and
constructs the one RNG (`self.rng = np.random.default_rng(seed)`) that every stochastic decision
in the run draws from. It sets `self.metric_name` to `"AUC"` for classification or `"R2"` for
regression, and declares — but does not fill — the data attributes (`X_train`, `cv_splits`, …).

### Data loading and the CV structure

`load_data_pd` ([base_ea.py:100](Source/Base/base_ea.py#L100)) is concrete and shared. It:

1. Splits the frame into features `X` and target `y`.
2. Chooses the problem type from `self.classification`: classification derives `self.labels`
   (`np.unique(y)`) and `self.binary_classification`, and stratifies; regression leaves those
   `None` and uses a plain shuffled split.
3. Builds a single train/test split (`train_test_split`, seeded with `self.seed`).
4. Builds the `n_folds`-fold CV split over the *training* set (`StratifiedKFold` for
   classification, `KFold` for regression), stored as `self.cv_splits`.
5. Calls `_prepare_cv_folds`.

`load_data_pd` is guarded to run exactly once (`assert self.n_folds is None`), because the CV
structure it wires up is immutable for the rest of the run.

`_prepare_cv_folds` ([base_ea.py:195](Source/Base/base_ea.py#L195)) preprocesses **each fold
independently**: it calls the abstract `_build_preprocessor()`, fits it on that fold's training
partition only (no leakage), transforms both partitions, and puts the four resulting arrays
(`X_train, y_train, X_val, y_val`) into the Ray object store as separate references. The per-fold
reference tuples are stored in `self.cv_splits_ref` and handed out by `get_cv_splits()`. A derived
EA's `evaluation()` launches one Ray task per `(individual, fold)` pair using these references.

`_build_preprocessor()` is **abstract** ([base_ea.py:234](Source/Base/base_ea.py#L234)) because
the right baseline differs by domain (HPO must fully scale + encode; CASH only one-hot-encodes and
leaves scaling to the evolved pipeline).

`smallest_cv_train_size()` ([base_ea.py:252](Source/Base/base_ea.py#L252)) returns the row count
of the smallest CV training fold. Derived EAs feed this into the `DataContext` so any
training-size-dependent bound is realizable on *every* fold.

### The variation skeleton

`variation_order(offspring_cnt, crossover_prob)` ([base_ea.py:281](Source/Base/base_ea.py#L281))
is the one concrete variation helper. Per offspring it draws `'m'` (mutation, 1 parent) or `'c'`
(crossover, 2 parents) using `crossover_prob`, and returns the operator list plus the total number
of parents to select. A derived `evolve()` calls this *before* parent selection so it knows how
many parents to pick.

Everything else about producing and scoring individuals is abstract and left to the subclass:
`crossover`, `mutate`, `model_test_evaluation`, `evolve`, `initialize_population`, `evaluation`,
`save_results`.

---

## `Individual` — the candidate

`Individual` ([individual.py](Source/Base/individual.py)) is a small data holder. Its base
`__init__` declares:

- `genotype` — the actual solution (set by the subclass; `get_genotype()` returns a deep copy).
- Performance slots: `train_performance`, `val_performance`, `test_performance`, filled by the EA
  through the write-once setters (`set_val_performance` asserts it was unset — a value is never
  silently overwritten).
- `ei` — expected improvement, set only for TPE-constructed offspring.
- Provenance tags the EA sets at creation and the archive consumes: `construction`
  (`"random"`/`"tpe"`), `operation` (`"mutation"`/`"crossover"`/`"crossover_mutation"`),
  `parent_ids`, `archive_id`, and `eval_error`.

The subclass only has to define the genotype representation and `__repr__`. The lifecycle of these
fields is: **EA sets `construction`/`operation`/`parent_ids` at creation → `evaluation()` sets the
performances and `eval_error` → `record_history()` reads them all and writes `archive_id` back.**

---

## `ModelParams` + `DataContext` — the search space

`ModelParams` ([model_param_space.py:52](Source/Base/model_param_space.py#L52)) owns one
component's `param_space`: a `{name: {"type": ..., "bounds": ...}}` dict where each entry is an
`IntParam`, `FloatParam` (optionally `log`-scaled), `CatParam`, or `BoolParam`
([model_param_space.py:10-25](Source/Base/model_param_space.py#L10)). It provides the operators
the EA uses to move through that space:

- `generate_random_parameters(rng)` — sample a whole genotype uniformly (used to seed the initial
  population). Floats respect `log` scaling via `sample_float_parameter`.
- `mutate_parameters_shift(params, var, mut_rate, rng)` — per-gene, with probability `mut_rate`,
  nudge the value by a multiplicative Gaussian (`shift_int_parameter`/`shift_float_parameter`,
  clipped to bounds). This is the **local** move used on the TPE/exploit path.
- `mutate_parameters_random(params, mut_rate, rng)` — per-gene resample uniformly from the full
  range. This is the **global** move used on the explore path.
- `tpe_parameters(params)` — return a TPE-compatible encoding of the genotype (identity deep copy
  by default; a subclass may override, e.g. to expand a structured gene into numeric dimensions).
- `eval_parameters(params, random_state)` — **abstract**; map an evolved genotype to concrete
  scikit-learn kwargs, folding in `random_state` for stochastic estimators so the worker can build
  `Estimator(**eval_parameters(...))` generically.
- `get_model_type()` — **abstract**; the component's string identifier.

`DataContext` ([model_param_space.py:27](Source/Base/model_param_space.py#L27)) is a frozen
dataclass (`n_samples`, `n_features`, `n_classes`) that the EA builds *after* `load_data_pd` and
passes to every `ModelParams.__init__`, so a space can size dataset-dependent bounds (e.g.
`n_quantiles`, Nystroem components, or GradientBoosting's legal `loss` set) at construction.

The `rng` threaded into every method is always the EA's single `self.rng`, which is what makes a
run reproducible from `seed` alone.

---

## `Archive` + `ArchiveEntry` — the history log

`Archive` ([archive.py:118](Source/Base/archive.py#L118)) is the append-only record of everything
evaluated. `ArchiveEntry` ([archive.py:58](Source/Base/archive.py#L58)) is one immutable row:
`id`, `key`, `generation`, `construction`, `operation`, `parent_ids`, `ei`, deep-copied `genome`,
`train_performance`, `val_performance`, and `error`.

Shared, concrete behavior lives in the base class:

- `add(individual, generation, construction, error, ei, operation, parent_ids)`
  ([archive.py:182](Source/Base/archive.py#L182)) — validates provenance (random offspring must
  carry `ei == -inf`; TPE offspring must carry a real `ei`; a mutation's two parent ids must be
  equal), reads the performances off the individual, computes the key, appends the entry, indexes
  it by key, and returns it (with its new `id`).
- `contains` / `entries_for` / `get_by_key` / `get_by_id` — keyed and positional lookup. Duplicate
  genomes are **kept** but share a key, which is how the EA answers "have we evaluated this
  before?" in O(1).
- `best(rng)` ([archive.py:295](Source/Base/archive.py#L295)) — the highest-validation
  non-errored entry, ties broken uniformly with the passed `rng`.
- `to_records()` — every entry as a JSON-safe dict (`_json_safe_float` collapses non-finite
  scores, e.g. the `-inf` regression penalty, to `null`).

Two hooks are **abstract**:

- `compute_key(individual)` — collapse a genome to a canonical string (subclasses serialize their
  own genotype shape via `_canonical_key`, which sorts keys deterministically).
- `build_individual(entry)` — rebuild a fresh, unevaluated `Individual` from a stored genome (used
  when the EA re-selects the best, or rebuilds TPE fitting samples).

---

## How they interact across one `evolve()` run

The derived `evolve()` methods all follow the same shape. Tracing it shows how the four
collaborators hand off to each other (line refs point at HPO's implementation as the canonical
example; CASH mirrors it):

1. **Init.** `initialize_population()` asks `ModelParams.generate_random_parameters(self.rng)` for
   a random genotype per slot and wraps each in an `Individual` tagged `construction = "random"`.

2. **Evaluate.** `evaluation(population)`:
   - `_resolve_duplicates` uses `Archive.compute_key` / `entries_for` to reuse any genome already
     scored and collapse in-batch repeats — only distinct, unseen genomes become Ray tasks.
   - For each pending individual it calls `ModelParams.eval_parameters(genotype, random_state=seed)`
     and launches one Ray task per `(individual, fold)` over `get_cv_splits()`.
   - As folds return, it means the per-fold scores and writes them onto the `Individual` via the
     setters; any failed fold flags `eval_error` and forces the mode's worst penalty.

3. **Record.** `record_history(evaluated, generation)` calls `Archive.add(...)` once per
   individual, passing its `construction`/`operation`/`parent_ids`/`ei`/`eval_error`, and writes
   the returned `entry.id` back to `individual.archive_id` so future offspring can cite it as a
   parent.

4. **Select survivors + track best.** `_drop_errored` removes errored individuals from the
   breeding pool (they stay in the archive). `update_best_seen` advances the scalar `best_perf`.

5. **Breed.** For each generation: `variation_order()` decides the per-offspring operators and
   parent count → `parent_selection()` runs tournaments over validation performance →
   `generate_offspring()` fits TPE from the archive (see the domain `tpe.md`) and calls `mutate` /
   `crossover`, which use `ModelParams.mutate_parameters_shift` (exploit) or
   `mutate_parameters_random` (explore) and tag each child's provenance via `_build_offspring`.
   Then back to step 2 with the offspring.

6. **Finish.** After the last generation, `_select_best()` draws the winner from
   `Archive.best(self.rng)`, rebuilds it with `build_individual`, and `model_test_evaluation`
   fits it on the full training split and scores the held-out test set. `save_results` /
   `save_archive` serialize the result and `Archive.to_records()`.

The through-line: **`BaseEA` owns the loop and the data; `ModelParams` proposes and mutates
genotypes; `Individual` carries a genotype and its results between steps; `Archive` remembers
everything and answers "seen before?" and "best so far?".** The TPE surrogate that ranks
candidates in step 5 is documented in [tpe.md](Source/Base/tpe.md).
