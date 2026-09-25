# HPO EA

This document explains how [ea.py](Source/HPO/ea.py), [individual.py](Source/HPO/individual.py),
[cv_evaluation.py](Source/HPO/cv_evaluation.py), and [archive.py](Source/HPO/archive.py) interact
across one run of the HPO `EA.evolve()`. It covers only HPO. The TPE surrogate is described
separately in [tpe.md](Source/HPO/tpe.md); the abstract base contract these files implement is in
[Source/Base/ea.md](Source/Base/ea.md).

## What HPO optimizes

HPO tunes the hyperparameters of **one fixed model type** (e.g. `RF`, `MLP`, `KSVC`) — the model
is chosen up front and never changes. Every individual in the run carries a
`{parameter_name: value}` genotype for that one estimator. The objective is 5-fold CV performance:
ROC-AUC for classification, R² for regression.

---

## The four collaborators

| File | Class / functions | Role in HPO |
|------|-------------------|-------------|
| [ea.py](Source/HPO/ea.py) | `EA(BaseEA)` | Orchestrates the generational loop for a single model. |
| [individual.py](Source/HPO/individual.py) | `HPOIndividual(Individual)` | Genotype = one estimator's hyperparameter dict. |
| [cv_evaluation.py](Source/HPO/cv_evaluation.py) | `cv_*` / `cv_*_reg` Ray tasks | Fit + score one model on one CV fold, in parallel. |
| [archive.py](Source/HPO/archive.py) | `HPOArchive(Archive)` | Keyed history of every evaluated hyperparameter set. |

---

## `HPOIndividual` — the genotype

[individual.py](Source/HPO/individual.py). A trivial subclass of `Individual`: the genotype is the
hyperparameter mapping and it additionally stores `self.model_type` (the same string for the whole
run, e.g. `"RF"`). `__repr__` prints the model type and params. Everything else (performance
slots, provenance tags) is inherited. Example genotype:

```python
HPOIndividual({'n_estimators': 300, 'max_depth': 12, 'criterion': 'gini',
               'max_features': 0.4, 'max_samples': 0.8, 'class_weight': 'None'}, 'RF')
```

---

## `HPOArchive` — the history log

[archive.py](Source/HPO/archive.py). Implements the two abstract `Archive` hooks for the HPO
genotype:

- `compute_key(individual)` serializes `{"model_type": ..., "params": genotype}` into a canonical
  string. Because the run's model type is fixed, two individuals match iff their hyperparameters
  are identical. This key is what powers duplicate detection (`contains` / `entries_for`).
- `build_individual(entry)` rebuilds a fresh `HPOIndividual` from a stored genome, stamping it with
  the archive's fixed `model_type` (passed once at construction in `load_data_pd`).

The archive is HPO's **single source of history** — it feeds duplicate reuse, TPE fitting, best
selection, and the saved `archive.json`.

---

## `cv_evaluation.py` — parallel per-fold evaluation

[cv_evaluation.py](Source/HPO/cv_evaluation.py) holds two families of `@ray.remote` functions, one
per estimator, with an identical signature so the EA can dispatch any of them uniformly:

- **Classification** (`cv_random_forest`, `cv_extra_trees`, `cv_kernel_svc`, `cv_gradient_boost`,
  `cv_knn`, `cv_mlp`): fit the classifier, score with ROC-AUC (`predict_proba`). Binary uses the
  positive column; multi-class uses one-vs-one over `labels`.
- **Regression** (`*_reg` siblings): fit the regressor, score with R² (`predict`). They accept
  `binary_class`/`labels` only for signature parity and ignore them.

Each returns `(id, train_score, val_score, status)` where `status = 1.0` on success and `-1.0` if
the fit/score raised (scores returned as `0.0/0.0`, but the EA overwrites an errored individual's
performance with the mode's full penalty). The `cv_mlp` / `cv_mlp_reg` tasks recombine the evolved
per-layer genes `layer_1..layer_5` into scikit-learn's `hidden_layer_sizes` tuple. `model_params`
already contains `random_state` when the caller seeded it (see below).

These functions receive already-preprocessed fold arrays — the preprocessing happened once in
`BaseEA._prepare_cv_folds` and lives in the Ray object store.

---

## `EA` — the orchestrator

### Construction and data loading

`EA.__init__` ([ea.py:73](Source/HPO/ea.py#L73)) validates the `model` token against the
task-appropriate list (the kernel-SVM token is mode-specific: `'KSVC'` classification vs `'SVR'`
regression) and stores the TPE knobs (`tpe_prob`, `gamma`, `num_offspring`), tournament size, and
the `error_penalty` (`0.0` for AUC, `-inf` for R²). The `param_space` and archive are **deferred**
because they need the loaded data.

`load_data_pd` ([ea.py:173](Source/HPO/ea.py#L173)) calls the base loader, then:

1. Builds a `DataContext(n_samples=smallest_cv_train_size(), n_features=..., n_classes=...)` —
   `n_samples` is capped at the smallest CV fold so training-size-dependent bounds hold on every
   fold.
2. Looks up `(param_space, ray_cv_func)` for the chosen `model` from a task-specific map
   (`model_configs`), e.g. `'RF' → (RandomForestParams(ctx), cv_random_forest)` for classification.
3. Constructs `self.eval_archive = HPOArchive(param_space.get_model_type())`.

`_build_preprocessor` ([ea.py:972](Source/HPO/ea.py#L972)) — HPO fits a **bare** estimator, so the
base preprocessor must fully numericize *and* scale: `StandardScaler` on numeric columns +
`OneHotEncoder` on categoricals.

### The `evolve()` loop

`evolve(gens, checkpoint_dir)` ([ea.py:226](Source/HPO/ea.py#L226)) is the whole run:

1. **Initialize** — `initialize_population()` ([ea.py:318](Source/HPO/ea.py#L318)) creates
   `pop_size` `HPOIndividual`s from `param_space.generate_random_parameters(self.rng)`, each tagged
   `construction = "random"`.
2. **Evaluate initial pop** — `evaluation(...)` (below).
3. **Record** — `record_history(evaluated, generation=-1)` ([ea.py:628](Source/HPO/ea.py#L628))
   adds every individual to `eval_archive` and writes back `archive_id`.
4. **Prune + track** — `_drop_errored` removes errored individuals from the breeding pool (they
   stay archived); `update_best_seen` advances `best_perf`.
5. **Checkpoint** (optional) — `checkpoint_best_seen(generation=-1, ...)`.
6. **Per generation** (`g` in `range(gens)`):
   - `variation_order(pop_size, crossover_prob)` → per-offspring `'m'`/`'c'` operators + parent
     count.
   - `parent_selection(...)` ([ea.py:336](Source/HPO/ea.py#L336)) — tournament selection over
     validation performance, tournament size clamped to the (possibly pruned) pool.
   - `generate_offspring(...)` ([ea.py:537](Source/HPO/ea.py#L537)) — fits TPE from the archive
     (if `tpe_prob > 0`) and calls `mutate`/`crossover` per the variation order.
   - `evaluation` → `record_history(generation=g)` → `_drop_errored` → `update_best_seen` →
     `checkpoint_best_seen`.
7. **Finish** — draw the best via `_select_best`, run `model_test_evaluation`, store
   `best_result`. If checkpointing ran, the final checkpoint already made that draw and stashed it,
   so it's reused verbatim.

### `evaluation()` — dedup, dispatch, finalize

`evaluation(candidates)` ([ea.py:363](Source/HPO/ea.py#L363)):

1. **Dedup** — `_resolve_duplicates` ([ea.py:505](Source/HPO/ea.py#L505)): any genome already in
   the archive is filled in place from the stored scores (`_reuse_archived_performance`) and
   excluded; genomes repeated within the batch collapse to one "pending" sighting. Only distinct,
   unseen genomes are hard-evaluated. This is sound because evaluation is genome-deterministic
   under the fixed seed + folds.
2. **Dispatch** — for each pending individual, `param_space.eval_parameters(genotype,
   random_state=self.seed)` produces scikit-learn kwargs (seed folded in), then one
   `self.ray_train_func.remote(...)` task is launched **per fold** over `get_cv_splits()`.
3. **Collect** — `ray.wait` drains results in waves of `self.cores`, accumulating per-fold
   train/val scores per individual. A fold with `status < 0` flags the individual errored.
4. **Finalize** — once all folds for an individual are in, its performance is the **mean** across
   folds — unless any fold errored, in which case both train and val are forced to `error_penalty`
   (a complete penalty, not a competitive-looking average). Results are written via the write-once
   setters and `eval_error` is set.
5. **Fill in-batch duplicates** — collapsed siblings copy their pending representative's scores.
6. `hard_eval_count += len(pending)`.

### Offspring construction

`generate_offspring` fits TPE once per generation (`self.tpe.fit(self._tpe_samples(),
self.param_space, self.rng)`) then dispatches to:

- `mutate(parent)` ([ea.py:789](Source/HPO/ea.py#L789)) — one parent.
- `crossover(parent_a, parent_b)` ([ea.py:813](Source/HPO/ea.py#L813)) — two parents, uniform
  per-gene recombination (`_uniform_crossover`), then an offspring-level mutation gate (probability
  `mut_prob`).

Both route through `_tpe_or_explore` ([ea.py:872](Source/HPO/ea.py#L872)), the shared
explore/exploit switch:

- **With probability `tpe_prob` (exploit):** generate `num_offspring` candidates via a **small
  local shift** (`mutate_parameters_shift`, variance `mut_var`), encode each with
  `param_space.tpe_parameters`, and keep the one `tpe.suggest_one` ranks best. Its acquisition
  score becomes the offspring's `ei`; tagged `construction = "tpe"`.
- **Otherwise (explore):** one candidate via an **unbiased random resample**
  (`mutate_parameters_random`); tagged `construction = "random"`, `ei = -inf`.

`_build_offspring` ([ea.py:842](Source/HPO/ea.py#L842)) wraps the chosen genotype, stamps its
`construction`/`operation`/`parent_ids` (a mutation cites its single parent twice), and — for TPE
offspring — its `ei`. All of this is consumed later by `record_history`.

`_tpe_samples()` ([ea.py:594](Source/HPO/ea.py#L594)) builds the TPE fitting set from the archive:
one entry per **unique** genome key, each rebuilt as an `HPOIndividual` whose genotype is passed
through `param_space.tpe_parameters` and whose validation performance is **negated** (TPE
minimizes; the EA maximizes). Errored individuals negate their penalty into the "bad" region.

### Selection, checkpoints, final test

`update_best_seen` ([ea.py:662](Source/HPO/ea.py#L662)) tracks only the scalar `best_perf`; the
winning individual is recovered from the archive on demand.

`_select_best` ([ea.py:678](Source/HPO/ea.py#L678)) draws `eval_archive.best(self.rng)` (random
tie-break), rebuilds it, and asserts its validation equals `best_perf`.

`checkpoint_best_seen` ([ea.py:704](Source/HPO/ea.py#L704)) — if `checkpoint_dir` is set, after
each generation it draws the best via the **same** `_select_best`, evaluates it on the test set
(skipping the refit when the drawn genome is unchanged from the last checkpoint), appends a row to
`checkpoints.csv`, writes a `results_eval_N.json` snapshot (N = candidates considered =
`len(eval_archive)`), and stashes the draw in `best_result` so the final result matches the last
checkpoint exactly.

`model_test_evaluation` ([ea.py:990](Source/HPO/ea.py#L990)) fits the base preprocessor on the full
training split, builds the estimator via `_build_classifier`/`_build_regressor` (each re-derives
kwargs through `eval_parameters(..., random_state=self.seed)`), fits on the full training set, and
scores train + held-out test (ROC-AUC or R²).

`save_results` ([ea.py:1111](Source/HPO/ea.py#L1111)) writes `best_results.json` and calls
`save_archive` to dump `eval_archive.to_records()` as `archive.json`.

---

## Interaction summary

```
initialize_population ── param_space.generate_random_parameters ─▶ HPOIndividual (construction="random")
        │
evaluation ── _resolve_duplicates ◀── HPOArchive.compute_key / entries_for   (reuse seen genomes)
        │         │
        │         └─ param_space.eval_parameters(seed) ─▶ cv_evaluation.cv_* Ray tasks (per fold)
        │                                                        │
        │◀──────────────────── mean fold scores / error penalty ┘
        ▼
record_history ─▶ HPOArchive.add(...) ─▶ writes archive_id back onto each HPOIndividual
        │
generate_offspring ── tpe.fit(_tpe_samples() from archive) ─▶ mutate / crossover
        │                 └─ _tpe_or_explore: shift+TPE (exploit) vs random resample (explore)
        ▼
_select_best ◀── HPOArchive.best(rng) ─▶ model_test_evaluation ─▶ best_results.json + archive.json
```

The recurring pattern: **the archive remembers every hyperparameter set, dedup avoids re-fitting
them, `cv_evaluation` scores the new ones in parallel across folds, and TPE (fit from the archive)
biases which mutations survive.**
