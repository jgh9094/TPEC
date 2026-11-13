# CASH EA

This document explains how [ea.py](Source/CASH/ea.py), [individual.py](Source/CASH/individual.py),
the CV-evaluation Ray tasks, and [archive.py](Source/CASH/archive.py) interact across one run of the
CASH `EA.evolve()`. It covers only CASH. The TPE surrogate is described separately in
[tpe.md](Source/CASH/tpe.md); the abstract base contract these files implement is in
[Source/Base/ea.md](Source/Base/ea.md).

> **Note on `cv_evaluation.py`:** CASH has no separate `cv_evaluation.py` file. Because a CASH
> individual builds a *whole scikit-learn `Pipeline`* (not a single estimator), its per-fold CV
> tasks — `cv_pipeline_classification` and `cv_pipeline_regression` — live at the top of
> [ea.py](Source/CASH/ea.py) alongside the pipeline-assembly helpers. This section documents them
> as CASH's CV-evaluation layer.

## What CASH optimizes

CASH = **Combined Algorithm Selection and Hyperparameter optimization**. It evolves a full
four-node pipeline, choosing both *which component* sits at each node and *its hyperparameters*:

```
feature_scaling → feature_engineering → feature_selection → predictor
```

Each of the first three nodes may be `passthrough` (skipped); the predictor is mandatory. The
genotype is therefore *structural* (which component per node) **and** *parametric* (that
component's params). Objective: 5-fold CV ROC-AUC (classification) or R² (regression).

---

## The collaborators

| File | Class / functions | Role in CASH |
|------|-------------------|--------------|
| [ea.py](Source/CASH/ea.py) | `EA(BaseEA)` | Orchestrates the loop; also holds pipeline assembly + CV Ray tasks. |
| [individual.py](Source/CASH/individual.py) | `CASHIndividual(Individual)` | Genotype = nested `{node: {"name", "params"}}` pipeline config. |
| [ea.py](Source/CASH/ea.py) CV tasks | `cv_pipeline_classification` / `_regression` | Build + score a full `Pipeline` on one CV fold, in parallel. |
| [archive.py](Source/CASH/archive.py) | `CASHArchive(Archive)` | Keyed history of every evaluated pipeline config. |

---

## `CASHIndividual` — the pipeline genotype

[individual.py](Source/CASH/individual.py). The genotype is a nested dict, one entry per node:

```python
{
  "feature_scaling":     {"name": "StandardScaler", "params": {}},
  "feature_engineering": {"name": "passthrough",    "params": {}},
  "feature_selection":   {"name": "SelectPercentile","params": {"percentile": 42}},
  "predictor":           {"name": "RF", "params": {"n_estimators": 300, "max_depth": 12, ...}},
}
```

`_validate_pipeline` guards this structure at construction (each entry must have exactly
`{"name", "params"}`, a string name, a dict of string-keyed params) so malformed genotypes fail
loudly here rather than deep inside TPE. A parameter-free component — including `"passthrough"` —
carries an empty `params` dict. `get_architecture()` returns just the node→component-name mapping
(the structural choice, no params), which the archive and TPE use as the pipeline's "shape."

---

## `CASHArchive` — the history log

[archive.py](Source/CASH/archive.py). Implements the two abstract `Archive` hooks:

- `compute_key(individual)` serializes the **whole nested genotype** canonically, so two pipelines
  match iff every node picks the same component with the same parameter values.
- `build_individual(entry)` rebuilds a fresh `CASHIndividual` from a stored genome.

This is CASH's single source of history — feeding dedup, TPE fitting, best selection, and
`archive.json`.

---

## Pipeline assembly + CV Ray tasks

Because a CASH genotype is a whole pipeline, several module-level helpers turn a genotype into a
runnable scikit-learn `Pipeline`:

- `PIPELINE_NODES` ([ea.py:77](Source/CASH/ea.py#L77)) — the ordered nodes and which registry
  (`SCALERS`, `TRANSFORMERS`, `SELECTORS`, `CLASSIFIERS`) supplies each, plus whether `passthrough`
  is allowed. Order defines pipeline execution order *and* the TPE architecture-key order.
- `build_estimator(component_name, eval_kwargs, classification)`
  ([ea.py:137](Source/CASH/ea.py#L137)) — instantiate one node's scikit-learn estimator from its
  `eval_parameters` kwargs. Handles the few special cases: `MLP` recombines `layer_1..layer_5` into
  `hidden_layer_sizes`; `KSVC` adds `probability=True`; `XGB` sets `n_jobs=1` (one Ray task per fold,
  so xgboost gets one thread). The predictor table is chosen by `classification`, and its XGB
  classifier is `AnyLabelXGBClassifier`, which accepts any label coding.
- `assemble_steps(pipeline_steps, scale_cols, classification)`
  ([ea.py:181](Source/CASH/ea.py#L181)) — turn resolved `(node, component, kwargs)` triples into
  `(name, estimator)` steps, **dropping any `passthrough` node** (no identity step inserted), and
  applying **School-B scaling** at the `feature_scaling` node: if the matrix has protected columns
  (one-hot dummies), the scaler is wrapped in a `ColumnTransformer` that scales only the numeric
  indices in `scale_cols` and passes the rest through.

`cv_pipeline_classification` ([ea.py:227](Source/CASH/ea.py#L227)) and `cv_pipeline_regression`
([ea.py:288](Source/CASH/ea.py#L288)) are the `@ray.remote` per-fold tasks. Each: `assemble_steps`
→ `Pipeline(steps)` → `fit` on the fold's (already preprocessed) training partition → score
(ROC-AUC via `predict_proba`, or R² via `predict`). Both return `(id, train, val, status)` with
`status = -1.0` on any exception. A CASH pipeline can *legitimately* fail on a fold (e.g. a
`VarianceThreshold` that removes every feature); that fold returns `0.0/0.0` and flags the
individual, and any fold failure penalizes the whole pipeline completely.

---

## `EA` — the orchestrator

### Construction and data loading

`EA.__init__` ([ea.py:349](Source/CASH/ea.py#L349)) stores the EA/TPE knobs plus CASH-specific
ones: `component_mut_prob` (probability a node's *component* is resampled during mutation) and
per-step allow-lists (`scalers`/`transformers`/`selectors`/`predictors`; empty = use the full
registry). The pipeline space, operators, and TPE are **deferred** to `load_data_pd`.

The TPE and checkpoint arguments: `tpe_offspring` (probability an offspring is chosen by the TPE from
`num_offspring` candidate siblings) and `checkpoint_every_generation`
(default `True`; see [per-generation checkpoints](#per-generation-test-checkpoints)).

`load_data_pd` ([ea.py:471](Source/CASH/ea.py#L471)) calls the base loader, then:

1. Builds a `DataContext` (with `n_samples` capped at the smallest CV fold).
2. Computes `self.scale_cols` — the numeric column index range the evolved scaler may touch. The
   CASH preprocessor lays out numeric columns first; if one-hot dummies exist, `scale_cols =
   [0, n_numeric)` (School-B); if the whole matrix is numeric, `scale_cols = None` (scaler applies
   bare).
3. `_build_pipeline_space()` ([ea.py:520](Source/CASH/ea.py#L520)) — for each node, instantiate
   every registered `ModelParams` with the `DataContext`, filter by the allow-list, add
   `passthrough` where permitted. Produces `self.operators` (live `ModelParams` objects, `None` for
   passthrough) and `self.pipeline_space` (`{node: {component: param_space}}` metadata for TPE).
4. `self.tpe = CASH_TPE(gamma, pipeline_space, factorize_structure, joint_weight,
   component_fallback)` (the last three come from the EA's `tpe_factorize_structure`,
   `tpe_joint_weight`, `tpe_component_fallback` arguments; see [tpe.md](Source/CASH/tpe.md)) and
   `self.eval_archive = CASHArchive()`.

`_build_preprocessor` ([ea.py:1157](Source/CASH/ea.py#L1157)) — CASH only makes data numeric:
one-hot the categoricals, **pass numerics through UNSCALED** (scaling is an evolved decision).
Numerics are listed first so the scaler can target them by index.

### The `evolve()` loop

`evolve(gens)` ([ea.py:709](Source/CASH/ea.py#L709)) mirrors HPO's loop, with two additions: the TPE
is fitted once per generation before parent selection, and a test checkpoint is recorded after the
initial population and after every generation (see [per-generation checkpoints](#per-generation-test-checkpoints)).

1. `initialize_population()` ([ea.py:791](Source/CASH/ea.py#L791)) — `pop_size` random pipelines
   via `_random_candidate` (one random component + sampled params per node), tagged
   `construction = "random"`.
2. `evaluation(...)` → `record_history(generation=-1)` → `_drop_errored` → `update_best_seen` →
   `_checkpoint(-1)`.
3. Per generation `g`:
   - `_fit_tpe()` — if `tpe_offspring > 0`, fits the TPE on the archive (`tpe_ready` records whether
     the fit succeeded).
   - `variation_order` → `parent_selection` (fitness tournaments; see
     [parent selection](#parent-selection)) → `generate_offspring` →
     `evaluation` → `record_history(generation=g)` → `_drop_errored` → `update_best_seen` →
     `_checkpoint(g)`.
4. Finish: `total_time_minutes` is recorded **without** checkpoint time. The held best's result comes
   from `_select_best`, and its train/test scores come from the last checkpoint row, so the best is not
   refit a second time. If checkpoints are off, or that refit failed, `model_test_evaluation` runs as
   before. The result is stored as `best_result` (train/val/test performance plus complexity).

### `evaluation()` — dedup, dispatch, finalize

`evaluation(candidates)` ([ea.py:1209](Source/CASH/ea.py#L1209)):

1. **Dedup** — `_resolve_duplicates` reuses archived scores for seen pipelines and collapses
   in-batch repeats; only distinct, unseen genomes are hard-evaluated (evaluation is
   genome-deterministic).
2. **Resolve** — `_resolve_pipeline_steps(genotype)` ([ea.py:1182](Source/CASH/ea.py#L1182)) turns
   each pending genotype into ordered `(node, component, eval_kwargs)` triples, calling each live
   operator's `eval_parameters(params, random_state=self.seed)`; passthrough nodes carry the
   `PASSTHROUGH` marker.
3. **Dispatch** — pick `cv_pipeline_classification`/`_regression` by task, launch one task **per
   fold** per pending candidate over `get_cv_splits()`, passing `pipeline_steps` and `scale_cols`.
4. **Collect + finalize** — mean fold scores per pipeline; any errored fold forces the complete
   penalty (`0.0` for AUC, `-inf` for R²). Write via the write-once setters, set `eval_error`.
5. Fill in-batch duplicates; `hard_eval_count += len(pending)`.

### Offspring construction — structural + parametric

The TPE is fitted once per generation by `_fit_tpe`, at the start of the generation and before parent
selection (`self.tpe_ready = self.tpe.fit(self._tpe_samples(), self.rng)`). `generate_offspring`
([ea.py:832](Source/CASH/ea.py#L832)) no longer fits it. Unlike HPO, the pipeline-aware `fit`
**returns a bool**: `tpe_ready` gates whether TPE guidance is used this generation (it's `False` when
history can't be split into good/bad groups). Then per offspring it calls `mutate` or `crossover`.

`mutate` ([ea.py:882](Source/CASH/ea.py#L882)) and `crossover` ([ea.py:908](Source/CASH/ea.py#L908))
route through `_tpe_or_explore` ([ea.py:675](Source/CASH/ea.py#L675)):

- **Exploit (prob `tpe_offspring`, and `tpe_ready`):** generate `num_offspring` candidates, score the
  **nested genotypes directly** with `tpe.suggest_one` (no per-parameter re-encoding — CASH_TPE
  reads the nested dicts), keep the best; its score becomes `ei`; tag `construction = "tpe"`.
  With `tpe_offspring == 0` the roll is skipped entirely, so no random draw is made.
- **Explore (otherwise):** one candidate; tag `construction = "random"`, `ei = -inf`.

**Redundant-offspring filtering** (`dedup_offspring=True`, the default). A pipeline is redundant
if it is already in the archive or was already produced earlier in the same generation
(`self._generation_keys`, reset at the top of `generate_offspring`). Without filtering, copies of
the parent are common among siblings and the TPE tends to rank them highly (they sit exactly on a
good history point), so an offspring slot is spent re-reading an archived score.

- Exploit: redundant siblings are dropped **before** ranking; the TPE picks among the new ones.
- Explore: a redundant candidate is redrawn, up to `num_offspring` attempts in total (the same
  candidate budget as a TPE roll), so the TPE and non-TPE paths follow the same rule.
- Initial population: drawn without repeats.
- If every attempt is redundant, one is kept anyway and counted in `self.redundant_offspring`
  (printed at the end of `evolve`). So with filtering on,
  `hard_eval_count == len(eval_archive) - redundant_offspring`.

Helpers: `_genome_key` (same encoding as the archive key), `_is_novel`, `_claim`. Set
`dedup_offspring=False` for the old behaviour.

The candidate factories are what make CASH's variation structural + parametric:

- `_mutate_genotype(genotype, use_tpe)` ([ea.py:591](Source/CASH/ea.py#L591)) — per node,
  probability `component_mut_prob` does **structural** mutation (resample the component and draw
  fresh params via `_random_node_entry`); otherwise **parametric** mutation keeps the component and
  either shift-mutates (`use_tpe=True`, local, variance `mut_var`) or random-resamples
  (`use_tpe=False`, global) its params.
- `_crossover_child` ([ea.py:624](Source/CASH/ea.py#L624)) — `_uniform_crossover`
  ([ea.py:652](Source/CASH/ea.py#L652)) recombines **per node as a unit** (component + params kept
  together, so a child never pairs one component's name with another's params), then an
  offspring-level mutation gate (prob `mut_prob`) may mutate the child.

`_build_offspring` ([ea.py:937](Source/CASH/ea.py#L937)) stamps `construction`/`operation`/
`parent_ids`/`ei` for `record_history`.

`_tpe_samples()` ([ea.py:967](Source/CASH/ea.py#L967)) builds the TPE fitting set from the archive:
one `CASHIndividual` per unique genome key, each with **negated** validation performance (TPE
minimizes; the EA maximizes). Errored pipelines negate into the "bad" group.

### Parent selection

Tournaments are decided by validation fitness. For each of the `num_parents` tournaments,
`parent_selection` draws `k = min(tournament_size, len(population))` competitors with `rng.choice`, takes
the highest validation score, and breaks ties with `rng.choice`.

A `tpe_selection` knob (the probability that a tournament compared TPE scores instead of fitness) was
tried in the HIV TwoKnob sweep and removed: it weakened selection pressure without improving
offspring (see `Data-Tools/HIV-TwoKnob/REPORT.md`; the code is in git history up to commit `d268a07`).
At `tpe_selection = 0` that code drew exactly the same random numbers as the current one, so the
reference fixture `Test/fixtures/reference_cash_archive.json` still reproduces
(`Test/test_cash_parent_selection.py`).

The TPE is fitted once per generation by `_fit_tpe`, before any offspring is built, and only when
`tpe_offspring > 0`.
`CASH_TPE.fit` consumes no random draws (every KDE it builds has a bandwidth floor, so the jitter
path that uses `rng` never runs). Fitting earlier than the old per-offspring point therefore leaves
the search unchanged.

### Selection and final test

`update_best_seen` ([ea.py:1026](Source/CASH/ea.py#L1026)), `_select_best`
([ea.py:1079](Source/CASH/ea.py#L1079)), and `model_test_evaluation`
([ea.py:1333](Source/CASH/ea.py#L1333)) parallel HPO but operate on pipelines: the final test refit
fits the base preprocessor on the full train split, assembles the winning pipeline via
`_resolve_pipeline_steps` + `assemble_steps` (same steps as CV, passthrough omitted, scaler
numeric-aware), fits, and scores train + test. It also returns the fitted Pipeline's
`calculate_model_complexity` score ([Source/ML/complexity.py](Source/ML/complexity.py)): the
predictor's parameter count, plus 1 per transformer/scaler step and 0 per selector; the fixed base
preprocessor is not counted. `best_results.json` additionally records the winning `architecture`
(via `get_architecture()`), `complexity`, `best_eval`, and `best_generation` (as in HPO).
`save_results` ([ea.py:1402](Source/CASH/ea.py#L1402)) writes `best_results.json` and `archive.json`,
plus `checkpoints.csv` and `checkpoint_runtime_minutes.txt` when checkpoints exist.

### Per-generation test checkpoints

With `checkpoint_every_generation` on (the default), `BaseEA._checkpoint`
([Source/Base/base_ea.py](Source/Base/base_ea.py)) refits the best-so-far pipeline after the initial
population and after every generation, and writes one row to `checkpoints.csv`. Columns, in order:

| Column | Meaning |
|---|---|
| `generation` | −1 for the initial population, then 0 … gens − 1 |
| `evaluations` | `len(eval_archive)`: candidates considered, which equals `pop_size * (generation + 2)` |
| `hard_evaluations` | distinct pipelines evaluated so far |
| `best_id`, `best_eval` | archive id of the best entry, and `best_id + 1` (as in `best_results.json`) |
| `best_generation` | generation that produced the best entry |
| `train_performance`, `validation_performance`, `test_performance` | the best's train AUC (refit on the full training set), its CV validation score, and its test AUC |
| `complexity` | `calculate_model_complexity` of the refit pipeline |
| `refit` | True if the pipeline was refit at this checkpoint |
| `refit_error` | empty unless the refit failed |
| `search_minutes` | search time so far, **excluding** checkpoint time |
| `tpe_ready` | whether the TPE was fitted this generation (CASH only; False at generation −1) |

Rules:

- **The search never reads these numbers.** Nothing in the search reads `self.checkpoints` or the
  cached scores; test scores are for reporting only. `Test/test_checkpoints.py` replaces the test
  evaluation with random numbers from a separate generator and confirms the archive is unchanged.
- **Refit only when the best changes.** The best is refit on the full training set only when its
  archive entry differs from the last checkpoint's. Otherwise the cached scores are reused, which is
  valid because a fit is deterministic for a fixed genome. `refit` is True exactly on those rows.
- **A failed refit never ends the run.** It is recorded (NaN train and test, `complexity = -1`,
  `refit_error = "<exception>"`) and cached for that best entry, so it is not retried every
  generation. The end of `evolve` retries the refit once, so a real failure still raises, as it did
  before checkpoints existed.
- **Budget prefix.** A run with budget N and the same seed produces the same rows as the first rows
  of a longer run, in every column except `search_minutes`. The row whose `evaluations` equals N is
  therefore the state an N-budget run would have reached. One max-budget run replaces the separate
  budget runs.
- **Timing.** Checkpoint time is measured and kept out of `total_runtime_minutes.txt` and
  `search_minutes`. It is written to `checkpoint_runtime_minutes.txt`.
- **Final result.** With checkpoints on, `best_result` is read from the last row, which holds the same
  best entry. `best_results.json` therefore has the same content as before.
- A checkpoint is skipped (no row is written) until some pipeline has a finite validation score. This
  only happens for regression, when every initial individual errors (R² = −inf).

---

## Interaction summary

```
initialize_population ── _random_candidate (per node: pick component + sample params) ─▶ CASHIndividual
        │
evaluation ── _resolve_duplicates ◀── CASHArchive.compute_key / entries_for   (reuse seen pipelines)
        │         │
        │         └─ _resolve_pipeline_steps (eval_parameters, seed) ─▶ cv_pipeline_* Ray tasks (per fold)
        │                                                                    │  assemble_steps → Pipeline → fit/score
        │◀──────────────────────────── mean fold scores / error penalty ────┘
        ▼
record_history ─▶ CASHArchive.add(...) ─▶ writes archive_id back onto each CASHIndividual
        │
_fit_tpe (start of generation) ── tpe_ready = tpe.fit(_tpe_samples() from archive)
        │
parent_selection ── fitness tournaments
        │
generate_offspring ── mutate / crossover ── TPE-chosen offspring (tpe_offspring)
        │                 └─ structural (component) + parametric (shift/random) variation
        ▼
_checkpoint(g) ── refit best only if it changed ─▶ checkpoints.csv (reporting only, never read by search)
        │
_select_best ◀── CASHArchive.best(rng) ─▶ final result from last checkpoint ─▶ best_results.json + archive.json
```

The recurring pattern: **the archive remembers every full pipeline, dedup avoids re-fitting them,
the in-`ea.py` `cv_pipeline_*` tasks build and score whole pipelines in parallel across folds, and
the pipeline-aware TPE (fit from the archive) biases both the structural and parametric moves that
survive.**
