# Base TPE surrogate

This document explains how the TPE base class in [tpe.py](Source/Base/tpe.py) is set up: the
flavor-independent machinery every concrete TPE surrogate inherits. The two concrete surrogates —
`HPO_TPE` ([Source/HPO/tpe.py](Source/HPO/tpe.py)) and `CASH_TPE`
([Source/CASH/tpe.py](Source/CASH/tpe.py)) — are documented in their own `tpe.md` files; read this
first to understand the shared foundation.

## What TPE does here (in one paragraph)

TPE (Tree-structured Parzen Estimator) is used only to **rank** candidates the EA already
generated — it never samples new ones. Given the evaluated history, it splits observations into a
"good" group (best objective values) and a "bad" group, fits a density model `l(x)` over the good
group and `g(x)` over the bad group, and scores each candidate by the ratio `l(x) / g(x)` (higher
= looks more like the good solutions than the bad ones). In this codebase the ratio is computed in
**log space** (`log l − log g`) for numerical stability. The EA generates a handful of candidate
offspring, asks TPE which one is best, and keeps that one.

[tpe.py](Source/Base/tpe.py) contains four pieces: two density primitives (`MultivariateKDE`,
`CategoricalPMF`), a bundle of them for one parameter group (`ParamGroupModel`), and the abstract
surrogate (`BaseTPE`).

---

## `MultivariateKDE` — density over numeric parameters

[tpe.py:32](Source/Base/tpe.py#L32). A thin wrapper over `scipy.stats.gaussian_kde` that models the
*joint* density of a group's numeric parameters.

- Constructed from `data` of shape `(d, n)` — `d` dimensions (numeric params) × `n` samples.
- **Requires `n > d`** (`raise ValueError` otherwise): you need strictly more observations than
  dimensions or the covariance is singular. This requirement is why numeric guidance only appears
  after enough history accumulates.
- Bandwidth uses Silverman's rule, scaled by `bw_factor`.
- **Optional `min_bandwidth`** (one positive std per dimension). When given, the kernel covariance
  is `data_covariance × factor² + diag(min_bandwidth²)` (via `_FlooredGaussianKDE`,
  [tpe.py:9](Source/Base/tpe.py#L9)). The floor is added rather than max-ed so correlations the data
  do show are kept, and the covariance is always positive definite. `_fit_param_group` always passes
  a floor (see below).
- Without a floor: if the Cholesky factorization fails (near-singular data), it jitters the data
  with noise scaled to each feature's spread and retries once; a second failure raises
  `ValueError`. Note this only catches *outright* failures: data that are constant in one dimension
  can still produce a covariance with eigenvalues around 1e-20, a KDE of zero width in that
  direction, and log-densities around −1e20 for any point off that line. The floor exists for this.
- `pdf(vec)` floors the density at `eps = 1e-12` so `log(pdf)` is always finite.
- `sample(rng, n)` exists but is unused by the ranking-only surrogates.

**Example.** Two numeric params (`C`, `gamma`) observed over 5 good configs:

```python
good = np.array([[0.5, 0.8, 1.2, 0.6, 0.9],    # C values      (dimension 1)
                 [0.01, 0.02, 0.015, 0.03, 0.02]])  # gamma values (dimension 2)
kde_good = MultivariateKDE(good, rng)   # n=5 > d=2, OK
density = kde_good.pdf([0.7, 0.018])[0]  # joint density at C=0.7, gamma=0.018
```

---

## `CategoricalPMF` — density over a categorical parameter

[tpe.py:86](Source/Base/tpe.py#L86). A smoothed probability mass function over a discrete
parameter's allowed values.

- Constructed from the observed `values` and the full `all_categories` support.
- Uses **Laplace (add-`alpha`) smoothing** (`alpha = 1.0`) so a category never seen still gets a
  non-zero probability: `P(c) = (count(c) + alpha) / Σ(count + alpha)`.
- `pmf(x)` returns the smoothed probability, or `eps` for a value outside the support.

**Example.** A `kernel` param with support `('linear', 'poly', 'rbf', 'sigmoid')`, observed as
`['rbf', 'rbf', 'linear']` in the good group:

```python
pmf_good = CategoricalPMF(['rbf', 'rbf', 'linear'],
                          all_categories=('linear', 'poly', 'rbf', 'sigmoid'))
# counts: rbf=2, linear=1, poly=0, sigmoid=0; total = (2+1)+(1+1)+(0+1)+(0+1) = 7
pmf_good.pmf('rbf')     # (2+1)/7 = 0.4286
pmf_good.pmf('poly')    # (0+1)/7 = 0.1429  <- smoothing keeps it non-zero
```

---

## `ParamGroupModel` — good/bad models for one parameter group

[tpe.py:193](Source/Base/tpe.py#L193). Bundles, for **one group of observations over a flat
parameter sub-space**, an optional numeric KDE pair (`multi_l` good / `multi_g` bad) plus an
optional PMF pair per categorical parameter (`cat_l` / `cat_g`). Any component can be missing when
there isn't enough evidence.

`log_ratio(params)` ([tpe.py:233](Source/Base/tpe.py#L233)) is the scoring workhorse — it sums the
available good/bad log-density ratios:

- If the numeric KDEs exist, add one joint term: `log l_num − log g_num`, evaluated at the
  candidate's numeric values.
- For each categorical PMF, add `log pmf_l(value) − log pmf_g(value)`.
- Missing components simply contribute nothing — a **log-ratio of `0.0`** ("no evidence", neither
  reward nor penalty; equivalently a density ratio `l/g = 1`, since `log(1) = 0` — this is *not*
  `log(0)`, which would be `−∞`). So a partially-mature group is still rankable.
- **Clipping.** The group's total is clipped to `[−clip, clip]` (`clip` comes from
  `BaseTPE.log_ratio_clip`, default **5.0**; `None` disables it). So one group's parameter evidence
  (one CASH node, or HPO's whole genotype) can never outweigh everything else, and huge and tiny
  terms are never summed in floating point (where the tiny ones would vanish). ±5 is a density
  ratio of about 150 to 1.

**Example.** Suppose a group has numeric KDEs for `(C, gamma)` and a PMF for `kernel`. For a
candidate `{'C': 0.7, 'gamma': 0.018, 'kernel': 'rbf'}`:

```
score = (log l_num([0.7, 0.018]) - log g_num([0.7, 0.018]))   # joint numeric term
      + (log pmf_l_kernel('rbf')  - log pmf_g_kernel('rbf'))    # categorical term
```

A positive `score` means the candidate resembles the good group more than the bad group.

This same `ParamGroupModel` is reused by both surrogates: `HPO_TPE` fits **one** group over the
whole flat genotype; `CASH_TPE` fits **one per `(architecture, node)`**, plus (by default) a
pooled fallback **one per `(node, component)`** (see [Source/CASH/tpe.md](Source/CASH/tpe.md)).

---

## `BaseTPE` — the abstract surrogate

[tpe.py:259](Source/Base/tpe.py#L259). Holds only the flavor-independent parts: the good/bad split
and the candidate-ranking helpers. Concrete density modeling is deferred to subclasses through two
abstract methods.

### Construction and the good/bad split

`__init__(gamma, log_ratio_clip=5.0)` stores `gamma ∈ (0, 1)`, the fraction of history treated as
"good", and the clip applied to every parameter group's log-ratio (see `ParamGroupModel`). Both
`HPO_TPE` and `CASH_TPE` accept `log_ratio_clip` and pass it through.

`split_samples(samples)` ([tpe.py:287](Source/Base/tpe.py#L287)):

- Requires at least 2 samples.
- Sorts by `get_val_performance()` **ascending** (lowest first). Note: the surrogate treats the
  objective as a **minimization**, so the EA feeds it *negated* validation scores (it maximizes,
  TPE minimizes) — see the domain `ea.md`/`tpe.md`. After negation, "lowest" = "best".
- The good group is the first `ceil(len(samples) * gamma)` samples (at least 1); the rest are bad.

**Example.** `gamma = 0.25`, 8 samples → `split_idx = ceil(8 * 0.25) = 2`: the 2 best-objective
samples are "good", the other 6 "bad".

### Ranking helpers

Both call the abstract `score_candidates` and break ties with the EA's `rng`:

- `suggest_one(candidates, rng)` ([tpe.py:308](Source/Base/tpe.py#L308)) — index of the highest
  scorer, uniformly random among ties. This is what the EA calls to pick the best of its
  `num_offspring` proposals.
- `suggest_top_k(candidates, k, rng)` ([tpe.py:330](Source/Base/tpe.py#L330)) — the top `k`
  indices, sampling without replacement across the tie boundary.

### The shared fitting helper

`_fit_param_group(good_param_dicts, bad_param_dicts, param_specs, rng)`
([tpe.py:380](Source/Base/tpe.py#L380)) builds one `ParamGroupModel` from a group's good/bad
observations and its `{param: spec}` metadata:

- Numeric params (`int`/`float`) get a joint `MultivariateKDE` pair, with three safeguards for
  small, repetitive histories:
  1. **Unique vectors.** Each group's numeric vectors (on the modeled scale, log for `log=True`)
     are deduplicated before fitting. Many history entries can carry one component's exact
     settings (crossover copies whole nodes; mutation leaves most genes untouched); counting one
     setting several times would concentrate the density on it.
  2. **`n > d` on unique vectors.** The pair is fit only if **both groups have strictly more
     unique vectors than numeric dimensions**; otherwise the numeric part is unavailable.
  3. **Minimum bandwidth** per dimension: `range / min(100, 1 + n_unique)`, where `range` is the
     parameter's bound width on the modeled scale. This is Optuna's TPE rule: wide kernels when there
     are few points, narrowing as evidence accumulates.
- Categorical params (`cat`/`bool`) get a `CategoricalPMF` pair whenever both groups are non-empty.
  PMFs count every observation (not deduplicated).
- The model's log-ratio is clipped at `self.log_ratio_clip`.
- Returns the model, or `None` if no evidence at all could be fit.

**Example (a real collapse these safeguards fix).** In a CASH run, RobustScaler's good group was
three pipelines that all carried the same `(q_min, q_max) = (8.65, 87.58)`, against 4 distinct bad
settings. Before the safeguards the good KDE had zero width, so a candidate at `(15.0, 72.9)`
scored −9 × 10²¹. Now:

- With deduplication, the good group is **1** unique vector, which is not more than `d = 2`, so
  there is no numeric model and the term is 0 (honestly: there is only one good setting).
- With only the floor (`30 / (1 + 3) = 7.5` per dimension), the same candidates score:
  - the repeated point: +2.05;
  - `(8.65, 71.8)`: +2.02;
  - `(15.0, 72.9)`: 0.00;
  - near the bad points: −1.49.

An RF good group with `max_features` constant at 0.9948 shows why deduplication alone isn't enough:
9 unique vectors still have zero spread in that dimension. The floor fixes that, and the clip caps
the remaining strong evidence at −5 instead of −1.9 × 10³¹. Both cases are tests in
`Test/test_tpe_kde_safeguards.py`.

Both concrete surrogates call this exact helper — that's the point of the base class.

### Abstract hooks

- `fit(*args, **kwargs)` — fit the good/bad models from history. Signature is flavor-specific: the
  flat HPO variant takes `(samples, param_space, rng)` and returns `None`; the pipeline-aware CASH
  variant takes `(samples, rng)` and returns a `bool` (whether usable guidance exists).
- `score_candidates(candidates)` — one acquisition score per candidate, higher = more promising.

### Deliberate non-feature: no sampling

`BaseTPE` has **no `sample()`** method. Classic TPE samples new candidates from `l(x)`; here the EA
owns candidate generation (mutation/crossover) and TPE is a pure ranker. This is the key design
choice that lets the surrogate stay simple and share `ParamGroupModel` across flavors.

---

## End-to-end (base-level view)

```
EA generates N candidate offspring
        │
        ▼
tpe.fit(history_samples, ...)          # split_samples → _fit_param_group(s)
        │
        ▼
idx = tpe.suggest_one(candidates, rng) # score_candidates → argmax (rng tie-break)
        │
        ▼
EA keeps candidates[idx]; its score becomes the offspring's expected-improvement (ei)
```

The concrete "fit" and "score" steps — one flat group for HPO, architecture + per-node groups for
CASH — are covered in [Source/HPO/tpe.md](Source/HPO/tpe.md) and
[Source/CASH/tpe.md](Source/CASH/tpe.md).
