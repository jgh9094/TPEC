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
  `hidden_layer_sizes`; `KSVC` adds `probability=True`; the predictor table is chosen by
  `classification`.
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
4. `self.tpe = CASH_TPE(gamma, pipeline_space)` and `self.eval_archive = CASHArchive()`.

`_build_preprocessor` ([ea.py:1157](Source/CASH/ea.py#L1157)) — CASH only makes data numeric:
one-hot the categoricals, **pass numerics through UNSCALED** (scaling is an evolved decision).
Numerics are listed first so the scaler can target them by index.

### The `evolve()` loop

`evolve(gens, checkpoint_dir)` ([ea.py:712](Source/CASH/ea.py#L712)) mirrors HPO's loop:

1. `initialize_population()` ([ea.py:791](Source/CASH/ea.py#L791)) — `pop_size` random pipelines
   via `_random_candidate` (one random component + sampled params per node), tagged
   `construction = "random"`.
2. `evaluation(...)` → `record_history(generation=-1)` → `_drop_errored` → `update_best_seen` →
   `checkpoint_best_seen(generation=-1)`.
3. Per generation `g`: `variation_order` → `parent_selection` (tournament over validation) →
   `generate_offspring` → `evaluation` → `record_history(generation=g)` → `_drop_errored` →
   `update_best_seen` → `checkpoint_best_seen`.
4. Finish: `_select_best` → `model_test_evaluation` → `best_result` (reused from the last
   checkpoint if checkpointing ran).

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

`generate_offspring` ([ea.py:832](Source/CASH/ea.py#L832)) fits TPE once per generation:
`self.tpe_ready = self.tpe.fit(self._tpe_samples(), self.rng)`. Unlike HPO, the pipeline-aware
`fit` **returns a bool** — `tpe_ready` gates whether TPE guidance is used this generation (it's
`False` when history can't be split into good/bad groups). Then per offspring it calls `mutate` or
`crossover`.

`mutate` ([ea.py:882](Source/CASH/ea.py#L882)) and `crossover` ([ea.py:908](Source/CASH/ea.py#L908))
route through `_tpe_or_explore` ([ea.py:675](Source/CASH/ea.py#L675)):

- **Exploit (prob `tpe_prob`, and `tpe_ready`):** generate `num_offspring` candidates, score the
  **nested genotypes directly** with `tpe.suggest_one` (no per-parameter re-encoding — CASH_TPE
  reads the nested dicts), keep the best; its score becomes `ei`; tag `construction = "tpe"`.
- **Explore (otherwise):** one candidate; tag `construction = "random"`, `ei = -inf`.

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

### Selection, checkpoints, final test

`_select_best` ([ea.py:1049](Source/CASH/ea.py#L1049)), `checkpoint_best_seen`
([ea.py:1075](Source/CASH/ea.py#L1075)), and `model_test_evaluation`
([ea.py:1393](Source/CASH/ea.py#L1393)) parallel HPO but operate on pipelines: the final test refit
fits the base preprocessor on the full train split, assembles the winning pipeline via
`_resolve_pipeline_steps` + `assemble_steps` (same steps as CV, passthrough omitted, scaler
numeric-aware), fits, and scores train + test. Checkpoints and `best_results.json` additionally
record the winning `architecture` (via `get_architecture()`). `save_results`
([ea.py:1452](Source/CASH/ea.py#L1452)) writes `best_results.json` and `archive.json`.

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
generate_offspring ── tpe_ready = tpe.fit(_tpe_samples() from archive) ─▶ mutate / crossover
        │                 └─ structural (component) + parametric (shift/random) variation
        ▼
_select_best ◀── CASHArchive.best(rng) ─▶ model_test_evaluation ─▶ best_results.json + archive.json
```

The recurring pattern: **the archive remembers every full pipeline, dedup avoids re-fitting them,
the in-`ea.py` `cv_pipeline_*` tasks build and score whole pipelines in parallel across folds, and
the pipeline-aware TPE (fit from the archive) biases both the structural and parametric moves that
survive.**
