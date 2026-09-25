# CASH TPE surrogate

This document explains how the CASH-specific, **pipeline-aware** TPE surrogate in
[tpe.py](Source/CASH/tpe.py) is set up and, with small worked examples, how it fits history and
scores candidate pipelines. It covers only CASH. The shared machinery it inherits
(`MultivariateKDE`, `CategoricalPMF`, `ParamGroupModel`, `BaseTPE`) is documented in
[Source/Base/tpe.md](Source/Base/tpe.md); read that first.

## The idea, CASH-specific

A CASH candidate is a *whole pipeline*: a choice of component at each node **plus** each chosen
component's hyperparameters. `CASH_TPE` models that in two layers:

1. **Architecture layer.** The joint choice of one component per node (scaler +
   feature-engineering + feature-selection + predictor) is a single categorical. One smoothed PMF
   pair (`architecture_l` good / `architecture_g` bad) is fit over **every expressible
   architecture**.
2. **Parameter layer.** For each *observed* architecture and each node whose component has
   parameters, an **architecture-conditioned** `ParamGroupModel` is fit from just that
   architecture's observations at that node.

A candidate's score = the architecture log-ratio **plus** the parameter log-ratio of every node
whose `(architecture, node)` model exists. A missing parameter model contributes a **log-ratio of
`0.0`** — it is skipped, adding nothing to the sum (equivalently, a neutral density ratio `l/g = 1`,
since `log(1) = 0`; this is *not* `log(0)`). So a partially-mature architecture still ranks against
a fully-mature one.

The whole class is **fully generic** — the nodes, components, and params all come from
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

`CASH_TPE(BaseTPE)` ([tpe.py:25](Source/CASH/tpe.py#L25)). `__init__(gamma, pipeline_space)`:

- stashes `pipeline_space` and `node_names`,
- precomputes `self.all_architectures = self._enumerate_architectures()` — the **Cartesian
  product** of each node's component choices ([tpe.py:69](Source/CASH/tpe.py#L69)). This is the
  full categorical support for the architecture PMFs, so an unseen-but-legal architecture still
  gets a smoothed non-zero probability.
- declares fit state: `architecture_l`, `architecture_g` (the PMF pair), and
  `parameter_models: {(architecture, node): ParamGroupModel}`.

Helper `get_architecture_key(candidate)` extracts a candidate's architecture tuple;
`_arch_to_str(architecture)` encodes that tuple as one string (e.g.
`"feature_scaling=StandardScaler|...|predictor=RF"`) because `CategoricalPMF` categories are
scalars.

---

## `fit(samples, rng) -> bool`

[tpe.py:94](Source/CASH/tpe.py#L94). Steps:

1. If `< 2` samples → return `False`.
2. `good, bad = split_samples(samples)`; if either group is empty → return `False`.
3. **Architecture PMFs.** Map each good/bad candidate to its architecture string and fit
   `architecture_l` / `architecture_g` over the full `all_architectures` support (smoothed).
4. **Conditioned parameter models.** For each **observed** architecture (those actually seen in
   good ∪ bad) and each node whose component has params, gather that architecture's good/bad param
   dicts at that node and call `self._fit_param_group(...)`. Store any non-`None` result under
   `(architecture, node)`.
5. Return `True`.

The `bool` return is the gate the EA reads as `tpe_ready` — parameter-model maturity does **not**
affect it; only whether architecture-level good/bad groups exist.

### Worked fit example

Suppose only two nodes for illustration — `feature_selection ∈ {SelectPercentile, passthrough}`
and `predictor ∈ {RF, KNN}` — so there are `2 × 2 = 4` expressible architectures. History has 8
evaluated pipelines; with `gamma = 0.25`, `split_samples` gives 2 good / 6 bad (on negated
objective).

- Say the 2 good pipelines are both `(feature_selection=SelectPercentile, predictor=RF)`, and the
  6 bad ones are a mix. The architecture PMFs are fit over all 4 architecture strings with Laplace
  smoothing, so:
  - `architecture_l("...SelectPercentile|predictor=RF")` is high (2 of 2 good),
  - `architecture_g` for that same string is low (few of 6 bad),
  - the other 3 architectures still get small non-zero probabilities.
- **Parameter models** are fit only for *observed* architectures. For the
  `(SelectPercentile, RF)` architecture: at the `feature_selection` node, `SelectPercentile` has a
  `percentile` param → a `ParamGroupModel` may fit if `n > d` holds in both groups; at the
  `predictor` node, `RF`'s numeric params similarly. `passthrough` nodes have no params, so they're
  skipped.

---

## `score_candidates(candidates) -> np.ndarray`

[tpe.py:147](Source/CASH/tpe.py#L147). For each candidate:

```
architecture = get_architecture_key(candidate)
score  = log architecture_l.pmf(arch_str) - log architecture_g.pmf(arch_str)   # always present after fit
for node in node_names:
    model = parameter_models.get((architecture, node))
    if model is not None:
        score += model.log_ratio(candidate[node]["params"])                     # else skipped: adds 0.0
```

Returns one score per candidate. The architecture term is always available after a successful
`fit`; parameter terms are added only where a conditioned model exists.

### Worked scoring example

Score a candidate `{feature_selection: {"name": "SelectPercentile", "params": {"percentile": 40}},
predictor: {"name": "RF", "params": {"n_estimators": 300, ...}}}` against the fit above:

```
architecture term:  log l("...SelectPercentile|predictor=RF") - log g(same)
                  =  log(0.60) - log(0.10)                                  = +1.792
feature_selection:  RF-conditioned SelectPercentile param model.log_ratio({"percentile": 40})
                  =  (+0.35)                                                 (numeric KDE + any PMFs)
predictor:          (SelectPercentile,RF)-conditioned RF param model.log_ratio({...})
                  =  (+0.80)
------------------------------------------------------------------------------------------------
score            =  1.792 + 0.35 + 0.80                                     = +2.942
```

A candidate on a rarely-good architecture (say `(passthrough, KNN)`) would get a **negative**
architecture term and, likely, no conditioned parameter models (each adding `0.0`), so it scores
lower. `suggest_one` returns the index of the higher scorer — the pipeline the EA keeps.

---

## How the EA drives it

Inside `CASH.EA.generate_offspring` → `_tpe_or_explore` (see [ea.md](Source/CASH/ea.md)):

```python
# once per generation, from the deduplicated archive history (negated val performance):
self.tpe_ready = self.tpe.fit(self._tpe_samples(), self.rng)   # bool gate

# per TPE-guided offspring (only when tpe_ready):
cands = [self._mutate_genotype(parent, use_tpe=True) for _ in range(num_offspring)]  # nested candidates
idx   = self.tpe.suggest_one(cands, self.rng)     # nested dicts scored directly — no re-encoding
ei    = float(self.tpe.score_candidates([cands[idx]])[0])       # -> offspring.ei
child = cands[idx]
```

Key points specific to CASH (contrast with HPO):

- **Two-layer model.** Architecture PMFs + per-`(architecture, node)` parameter groups, versus
  HPO's single flat group.
- **`fit` returns a `bool`.** The EA gates TPE use on `tpe_ready`; HPO's `fit` returns `None`.
- **No `tpe_parameters` encoding.** `score_candidates` reads the nested `Candidate` dicts directly;
  HPO re-encodes each flat genotype through `param_space.tpe_parameters` first.
- **Negated objective, ranking only.** Same as everywhere: `_tpe_samples` negates validation so the
  base `split_samples` treats maximization as minimization; TPE only ranks EA-generated candidates,
  never samples.
- `model_availability()` ([tpe.py:179](Source/CASH/tpe.py#L179)) is a diagnostic reporting which
  `(architecture, node)` parameter models currently carry evidence — useful for watching guidance
  mature across generations.
