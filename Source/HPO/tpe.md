# HPO TPE surrogate

This document explains how the HPO-specific TPE surrogate in [tpe.py](Source/HPO/tpe.py) is set up
and, with small worked examples, how it fits history and scores candidates. It covers only HPO. The
shared machinery it inherits (`MultivariateKDE`, `CategoricalPMF`, `ParamGroupModel`, `BaseTPE`) is
documented in [Source/Base/tpe.md](Source/Base/tpe.md); read that first.

## The idea, HPO-specific

HPO tunes one estimator, so the genotype is a **single flat** hyperparameter dict. `HPO_TPE` models
that entire genotype as **one parameter group**:

- one multivariate Gaussian KDE pair over **all numeric genes** jointly (good `l` / bad `g`), and
- one smoothed PMF pair per **categorical/bool gene**.

A candidate's score is the log density ratio `log l(x) − log g(x)` — higher means "looks more like
the historically good configs than the bad ones." The EA generates several candidate mutations and
uses this score to keep the most promising one.

---

## Class setup

`HPO_TPE(BaseTPE)` ([tpe.py:10](Source/HPO/tpe.py#L10)). `__init__(gamma)` just calls the base with
`gamma` and declares two fields filled at fit time:

- `self.param_space` — the flat `ModelParams` being optimized (stashed so scoring needs no extra
  argument).
- `self._model` — the fitted `ParamGroupModel` for the whole space, or `None` when history is too
  thin to fit anything.

The class is deliberately small: all density math is inherited from `BaseTPE._fit_param_group` and
`ParamGroupModel.log_ratio`. HPO's only job is "treat the whole genotype as one group."

---

## `fit(samples, param_space, rng)`

[tpe.py:48](Source/HPO/tpe.py#L48). Steps:

1. Stash `param_space`.
2. `good, bad = self.split_samples(samples)` — sort by (negated) objective, take the top `gamma`
   fraction as good (see [Base tpe.md](Source/Base/tpe.md)).
3. Extract each group's genotype dicts.
4. `self._model = self._fit_param_group(good_dicts, bad_dicts, param_space.param_space, rng)` —
   builds the one group model (numeric KDE pair + per-categorical PMF pairs), or `None` if no
   evidence could be fit.

Returns `None`.

### Worked fit example

Say we optimize an RF with a tiny history of 8 evaluated configs, and for simplicity look at just
two genes: numeric `max_features ∈ (0,1)` and categorical `criterion ∈ {gini, entropy, log_loss}`.
The EA feeds **negated** validation AUC (TPE minimizes). With `gamma = 0.25` and 8 samples,
`split_samples` puts the 2 best-AUC configs in "good", the other 6 in "bad".

- Numeric part: `max_features` is 1-dimensional. The KDE pair needs `n > d`, i.e. `> 1` observation
  per group — good has 2, bad has 6, so both KDEs fit. `multi_l` is a KDE over the 2 good
  `max_features` values; `multi_g` over the 6 bad ones.
- Categorical part: `criterion` gets a smoothed PMF from each group's observed values over the full
  3-value support.

If instead the good group had only **1** sample (e.g. `gamma` tiny or few samples), the numeric KDE
could not fit (`n=1` is not `> d=1`), so `multi_l/multi_g` stay `None` and only the categorical PMF
contributes — the model is *partially mature* but still usable.

---

## `score_candidates(candidates)`

[tpe.py:70](Source/HPO/tpe.py#L70). For each candidate dict:

- If `self._model is None` (no evidence yet), **every** candidate scores `0.0` — neutral — so
  `suggest_one` falls back to a uniform random pick instead of crashing.
- Otherwise, return `self._model.log_ratio(params)` per candidate.

Returns a NumPy array of one score per candidate.

### Worked scoring example

Continuing the RF example, suppose the fitted model gives, for a candidate
`{'max_features': 0.35, 'criterion': 'gini', ...}`:

```
numeric term:     log l_num(0.35) - log g_num(0.35)          = log(2.1) - log(0.8)  = +0.965
criterion term:   log pmf_l('gini') - log pmf_g('gini')      = log(0.6) - log(0.2)  = +1.099
--------------------------------------------------------------------------------------------
log_ratio (score) = 0.965 + 1.099                            = +2.064
```

(Other genes add their own terms the same way; the numbers here are illustrative.) A second
candidate `{'max_features': 0.9, 'criterion': 'log_loss', ...}` might score `−1.3`. Given both,
`suggest_one` returns the index of the `+2.064` candidate — that's the mutation the EA keeps.

---

## How the EA drives it

Inside `HPO.EA.generate_offspring` → `_tpe_or_explore` (see [ea.md](Source/HPO/ea.md)), one TPE
exploit step looks like:

```python
# once per generation, from the deduplicated archive history (negated val performance):
tpe.fit(self._tpe_samples(), self.param_space, self.rng)

# per TPE-guided offspring:
cands   = [mutate_parameters_shift(parent, mut_var, mut_prob, rng) for _ in range(num_offspring)]
encoded = [param_space.tpe_parameters(c) for c in cands]   # HPO encoding (identity by default)
idx     = tpe.suggest_one(encoded, rng)                    # best of the batch (rng tie-break)
ei      = float(tpe.score_candidates([encoded[idx]])[0])   # its acquisition score -> offspring.ei
child   = cands[idx]                                        # the offspring the EA keeps
```

Key points specific to HPO:

- **One group, whole genotype.** Unlike CASH there is no architecture and no per-node
  conditioning — a single `ParamGroupModel` covers the flat space.
- **`tpe_parameters` encoding.** Candidates are passed through `param_space.tpe_parameters` before
  scoring; the default is an identity deep copy, but a `ModelParams` subclass may override it (e.g.
  to expose a structured gene as numeric dimensions the KDE can model).
- **Negated objective.** `_tpe_samples` sets each sample's val performance to `val * -1.0` so the
  base `split_samples` (ascending sort = best first) treats maximization as minimization. Errored
  configs (penalized to the mode's worst score) negate into the "bad" group.
- **Ranking only.** `HPO_TPE` never samples new candidates — the EA's mutation operators generate
  them and TPE only ranks.
