# CASH TPE surrogate

This document explains how the CASH-specific, **pipeline-aware** TPE surrogate in
[tpe.py](Source/CASH/tpe.py) is set up and, with a worked example, how it fits history and scores
candidate pipelines. It covers only CASH. The shared machinery it inherits (`MultivariateKDE`,
`CategoricalPMF`, `ParamGroupModel`, `BaseTPE`) is documented in
[Source/Base/tpe.md](Source/Base/tpe.md); read that first.

## The idea, CASH-specific

A CASH candidate is a *whole pipeline*: a choice of component at each node **plus** each chosen
component's hyperparameters. `CASH_TPE` scores a candidate as the sum of three kinds of good/bad
log-ratio evidence:

1. **Per-node structure terms.** One smoothed PMF pair per node over that node's components (for
   example, the predictor node's PMFs over `RF, ET, KSVC, GB, KNN, MLP`). Evidence is **shared
   across architectures**: if RF does well, every pipeline that uses RF benefits, including
   pipelines whose full architecture has never been evaluated.
2. **Joint structure term.** One smoothed PMF pair over the joint choice of every node's component
   (the full architecture), multiplied by `joint_weight`. Because it is smoothed over every
   expressible architecture (1,440 in the current CASH sweep), it is roughly 0 for unseen
   architectures and only moves the score for architectures actually seen in history. With the
   per-node terms on, it therefore acts as an **interaction correction**: it captures
   combinations that do better or worse than their parts suggest (e.g. a scaler and selector that
   error together).
3. **Parameter terms.** For each node, the log-ratio of the **most specific** parameter model that
   exists:
   - the `(architecture, node)` model, fit only from pipelines with exactly this architecture;
   - otherwise the `(node, component)` model, fit from **every** pipeline that used this component
     at this node, whatever the other nodes were (the *component fallback*);
   - otherwise nothing: the node adds a log-ratio of `0.0` (a neutral `l/g = 1`, not `log(0)`).

Why three terms? With 1,440 architectures and budgets of 100–500 evaluations, almost every
architecture is seen 0–1 times. A joint-only model has too little data per architecture: unseen
architectures all tie, and parameter models almost never fit. Per-node terms and pooled parameter
models give the surrogate usable evidence from the first generation, while the joint term and the
architecture-level parameter models keep the interaction information where it exists.

The whole class is **fully generic**: the nodes, components, and params all come from
`pipeline_space`, so there are no scaler/predictor-specific branches. Adding a node is a metadata
change only.

---

## Key types

[tpe.py:13-22](Source/CASH/tpe.py#L13):

- `PipelineSpace` — `{node: {component: {param: spec}}}`. Passed at construction; the metadata TPE
  models against. A parameter-free component (e.g. `passthrough`, `MinMaxScaler`) has `{}`.
- `Architecture` — an ordered tuple of `(node, component)` pairs following `pipeline_space` node
  order. Hashable, so it keys `parameter_models`.
- `Candidate` — the nested `{node: {"name", "params"}}` genotype (exactly a `CASHIndividual`'s
  genome).

---

## Class setup

`CASH_TPE(BaseTPE)` ([tpe.py:26](Source/CASH/tpe.py#L26)).
`__init__(gamma, pipeline_space, factorize_structure=True, joint_weight=1.0, component_fallback=True, log_ratio_clip=5.0)`
([tpe.py:53](Source/CASH/tpe.py#L53)):

| Argument | Default | Meaning |
|---|---|---|
| `gamma` | — | Fraction of history treated as "good". |
| `pipeline_space` | — | `{node: {component: {param: spec}}}` metadata. |
| `factorize_structure` | `True` | Add the per-node component log-ratios (term 1). |
| `joint_weight` | `1.0` | Weight on the joint-architecture log-ratio (term 2). `0.0` drops it. Must be `> 0` if `factorize_structure=False`, otherwise there is no structure evidence at all. |
| `component_fallback` | `True` | Use `(node, component)` parameter models when no `(architecture, node)` model exists (term 3). |
| `log_ratio_clip` | `5.0` | Cap on each node's parameter log-ratio (see [Source/Base/tpe.md](Source/Base/tpe.md)). `None` disables it. |

**Legacy settings.** `factorize_structure=False, joint_weight=1.0, component_fallback=False`
reproduces the original joint-architecture-only surrogate exactly (this is tested). Use it as the
ablation baseline.

The constructor also:

- stashes `pipeline_space` and `node_names`;
- precomputes `self.all_architectures = self._enumerate_architectures()`, the **Cartesian
  product** of each node's component choices ([tpe.py:99](Source/CASH/tpe.py#L99)). This is the
  support for the joint PMFs, so an unseen-but-legal architecture still gets a smoothed non-zero
  probability;
- declares the fit state:
  - `architecture_l`, `architecture_g` — joint-architecture PMF pair;
  - `node_l`, `node_g` — `{node: CategoricalPMF}` per-node component PMF pairs;
  - `parameter_models` — `{(architecture, node): ParamGroupModel}`;
  - `component_models` — `{(node, component): ParamGroupModel}` (empty when
    `component_fallback=False`).

Helper `get_architecture_key(candidate)` extracts a candidate's architecture tuple;
`_arch_to_str(architecture)` encodes that tuple as one string (e.g.
`"feature_scaling=StandardScaler|...|predictor=RF"`) because `CategoricalPMF` categories are
scalars.

### EA arguments

The CASH EA exposes the three options as `tpe_factorize_structure`, `tpe_joint_weight`, and
`tpe_component_fallback` (same defaults) and forwards them when it builds the TPE in
`load_data_pd`. The experiment runners do not pass them yet, so runs use the defaults.

---

## `fit(samples, rng) -> bool`

[tpe.py:124](Source/CASH/tpe.py#L124). Steps:

1. If `< 2` samples → return `False`.
2. `good, bad = split_samples(samples)`; if either group is empty → return `False`.
3. **Joint-architecture PMFs.** Map each good/bad candidate to its architecture string and fit
   `architecture_l` / `architecture_g` over the full `all_architectures` support.
4. **Per-node PMFs.** For each node, fit `node_l[node]` / `node_g[node]` from the good/bad
   candidates' component names at that node, over that node's own components.
5. **Architecture-level parameter models.** For each **observed** architecture and each node whose
   component has params, gather that architecture's good/bad param dicts at that node and call
   `self._fit_param_group(...)`. Store any non-`None` result under `(architecture, node)`.
6. **Component-level parameter models** (only if `component_fallback`). For each node and each
   **observed** component with params, gather the good/bad param dicts of **every** pipeline that
   used that component there and call `self._fit_param_group(...)`. Store any non-`None` result
   under `(node, component)`.
7. Return `True`.

Steps 3 and 4 always run (they are cheap); the scoring flags decide whether their terms are used.

`_fit_param_group` only returns a model with evidence: the numeric KDE pair needs **more than `d`
observations in both** the good and bad groups (`d` = number of numeric params), and the
categorical PMF pairs need at least one observation in each group. Pooling in step 6 is what makes
the numeric KDE reachable in practice: RF (`d = 4`) needs 5 good and 5 bad RF pipelines in total,
instead of 5 + 5 with one exact architecture.

The `bool` return is the gate the EA reads as `tpe_ready`. Parameter-model maturity does **not**
affect it; only whether non-empty good and bad groups exist.

---

## `score_candidates(candidates) -> np.ndarray`

[tpe.py:228](Source/CASH/tpe.py#L228). For each candidate:

```
architecture = get_architecture_key(candidate)

# structure score (_structure_score, tpe.py:205)
score = 0
if factorize_structure:
    for node in node_names:
        score += log node_l[node].pmf(component) - log node_g[node].pmf(component)
if joint_weight > 0:
    score += joint_weight * (log architecture_l.pmf(arch_str) - log architecture_g.pmf(arch_str))

# parameter score (_parameter_model_for, tpe.py:218)
for node in node_names:
    model = parameter_models.get((architecture, node))
    if model is None and component_fallback:
        model = component_models.get((node, component))
    if model is not None:
        score += model.log_ratio(candidate[node]["params"])   # else adds 0.0
```

The more specific model always wins when both exist, even if the pooled model has a numeric KDE
and the architecture model has only categorical PMFs.

---

## Worked example

Everything below was computed with the repo's `CategoricalPMF` (Laplace smoothing, `alpha = 1`).
The structure numbers are checked by `test_worked_example_structure_scores` in
`Test/test_cash_tpe.py`.

### Setup

- **History:** the initial 25 pipelines, all different. With `gamma = 0.3`, the 8 with the highest
  AUC are "good" and the other 17 are "bad".
- **Parent:** the tournament picked **(StandardScaler → PCA → passthrough → KNN)**, one of the 8
  good pipelines.
- **Siblings:** mutating the parent produced, among others:

| Sibling | Scaler | Feature eng. | Feature sel. | Predictor | Predictor params |
|---|---|---|---|---|---|
| **A1** | StandardScaler | PCA | passthrough | KNN | weights=distance, algorithm=brute |
| **A2** | StandardScaler | PCA | passthrough | KNN | weights=uniform, algorithm=kd_tree |
| **B** | StandardScaler | PCA | passthrough | **RF** | criterion=entropy, class_weight=balanced |
| **C** | **Normalizer** | PCA | passthrough | KNN | weights=distance, algorithm=brute |
| **D** | StandardScaler | PCA | **VarianceThreshold** | KNN | weights=distance, algorithm=brute |

A1 and A2 keep the parent's architecture. B, C and D each had one structural mutation. A1/A2's
architecture was seen once, in the good group; D's was seen once, in the bad group (it errored);
B's and C's have never been seen.

### Per-node evidence

| Node | Component | Good (of 8) | Bad (of 17) | Log-ratio |
|---|---|---|---|---|
| Scaler | StandardScaler | 3 | 2 | +0.784 |
| Scaler | Normalizer | 0 | 5 | −1.295 |
| Feature eng. | PCA | 2 | 1 | +0.811 |
| Feature sel. | passthrough | 4 | 5 | +0.377 |
| Feature sel. | VarianceThreshold | 0 | 5 | −1.232 |
| Predictor | RF | 3 | 2 | +0.784 |
| Predictor | KNN | 1 | 4 | −0.420 |

Summing over each sibling's four nodes:

- A1, A2: 0.784 + 0.811 + 0.377 − 0.420 = **+1.552**
- B: 0.784 + 0.811 + 0.377 + 0.784 = **+2.756**
- C: −1.295 + 0.811 + 0.377 − 0.420 = **−0.527**
- D: 0.784 + 0.811 − 1.232 − 0.420 = **−0.057**

### Joint term

`log((good count + 1)/(8 + 1440)) − log((bad count + 1)/(17 + 1440))`:

- A1, A2 (1 good, 0 bad): **+0.699**
- B, C (unseen): **+0.006**
- D (0 good, 1 bad): **−0.687**

### Parameter terms

No `(architecture, node)` model exists for any sibling: A1/A2's architecture has 1 good and 0 bad
observations, and a model needs both groups. So each falls back to the pooled predictor model.

- **KNN** (pooled over 1 good, 4 bad KNN pipelines):
  - `weights`: `distance` +0.693, `uniform` −0.693;
  - `algorithm`: `brute` +0.560, `kd_tree` −0.539.
- **RF** (pooled over 3 good, 2 bad RF pipelines):
  - `criterion=entropy` +0.916;
  - `class_weight=balanced` +0.916.

For readability, the scaler and PCA parameter terms are left out here; they work the same way.

### Totals

| Sibling | Per-node | Joint | Parameters | **Score** | Legacy score |
|---|---|---|---|---|---|
| **B** | +2.756 | +0.006 | +1.833 | **+4.595** | +0.006 |
| A1 | +1.552 | +0.699 | +1.253 | +3.505 | +0.699 |
| A2 | +1.552 | +0.699 | −1.232 | +1.020 | +0.699 |
| C | −0.527 | +0.006 | +1.253 | +0.732 | +0.006 |
| D | −0.057 | −0.687 | +1.253 | +0.509 | −0.687 |

What the new terms change:

1. **A1 vs A2 is no longer a coin flip.** Legacy ties them. The pooled KNN model separates them by
   about 2.5.
2. **The surrogate can leave the parent's architecture.** B was never seen as a whole, but its
   components each do well, so it ranks first. Legacy gives it about 0 and always prefers A1/A2.
3. **Unseen architectures are no longer all equal.** Legacy ties B and C. Now C is held back by
   Normalizer's 0 good / 5 bad record.
4. **The joint correction still works.** D's architecture errored once, and together with
   VarianceThreshold's poor per-node record, D ranks last.

### Caveats

- **Small counts give large weights.** RF's +1.833 parameter term rests on 3 good and 2 bad
  observations, and it is larger than any single structure term. Each node's parameter log-ratio is
  clipped to ±5 (`log_ratio_clip`, see [Source/Base/tpe.md](Source/Base/tpe.md)), which bounds this
  but doesn't remove it.
- **No numeric parameter evidence yet on generation 0.** KDEs need more than `d` *unique* good and
  bad settings of a component; pooling reaches that after a few generations. The pooled models are
  also where repeated settings are most common (different pipelines inheriting one component's
  exact parameters), which is why the base KDE deduplicates vectors and floors the bandwidth.
- **Pooling ignores context.** C's KNN parameters get +1.253 from KNN pipelines that did not use
  Normalizer. Parameters whose meaning depends on an earlier step (Binarizer/VarianceThreshold
  thresholds after a scaler, KNN's `n_neighbors` after PCA) are blurred across contexts until an
  `(architecture, node)` model exists and takes over.

---

## How the EA drives it

Inside `CASH.EA.generate_offspring` ([ea.py:854](Source/CASH/ea.py#L854)) →
`_tpe_or_explore` ([ea.py:692](Source/CASH/ea.py#L692)); see [ea.md](Source/CASH/ea.md):

```python
# once per generation, from the deduplicated archive history (negated val performance):
self.tpe_ready = self.tpe.fit(self._tpe_samples(), self.rng)   # bool gate

# per TPE-guided offspring (only when tpe_ready, with probability tpe_prob):
candidate_offspring = [tpe_candidate() for _ in range(self.num_offspring)]  # same operator as non-TPE
genotypes = [g for g, _ in candidate_offspring]
idx = self.tpe.suggest_one(genotypes, self.rng)   # nested dicts scored directly
ei  = float(self.tpe.score_candidates([genotypes[idx]])[0])    # -> offspring.ei
```

Key points specific to CASH (contrast with HPO):

- **Three-part score.** Per-node structure + joint structure + per-node parameter evidence with a
  pooled fallback, versus HPO's single flat group.
- **`fit` returns a `bool`.** The EA gates TPE use on `tpe_ready`; HPO's `fit` returns `None`.
- **No `tpe_parameters` encoding.** `score_candidates` reads the nested `Candidate` dicts directly;
  HPO re-encodes each flat genotype through `param_space.tpe_parameters` first.
- **Negated objective, ranking only.** `_tpe_samples` ([ea.py:982](Source/CASH/ea.py#L982))
  negates validation so the base `split_samples` treats maximization as minimization. TPE only
  ranks EA-generated candidates and never samples.
- **Diagnostics.** `model_availability()` reports which `(architecture, node)` parameter models
  exist and whether each has a numeric KDE; `component_model_availability()` does the same for the
  pooled `(node, component)` models. Both are useful for watching guidance mature across
  generations.
