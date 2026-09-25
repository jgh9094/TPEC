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

[tpe.py:9](Source/Base/tpe.py#L9). A thin wrapper over `scipy.stats.gaussian_kde` that models the
*joint* density of a group's numeric parameters.

- Constructed from `data` of shape `(d, n)` — `d` dimensions (numeric params) × `n` samples.
- **Requires `n > d`** (`raise ValueError` otherwise): you need strictly more observations than
  dimensions or the covariance is singular. This requirement is why numeric guidance only appears
  after enough history accumulates.
- Bandwidth uses Silverman's rule, scaled by `bw_factor`.
- If the Cholesky factorization fails (near-singular data), it jitters the data with noise scaled
  to each feature's spread and retries once; a second failure raises `ValueError`.
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

[tpe.py:141](Source/Base/tpe.py#L141). Bundles, for **one group of observations over a flat
parameter sub-space**, an optional numeric KDE pair (`multi_l` good / `multi_g` bad) plus an
optional PMF pair per categorical parameter (`cat_l` / `cat_g`). Any component can be missing when
there isn't enough evidence.

`log_ratio(params)` ([tpe.py:169](Source/Base/tpe.py#L169)) is the scoring workhorse — it sums the
available good/bad log-density ratios:

- If the numeric KDEs exist, add one joint term: `log l_num − log g_num`, evaluated at the
  candidate's numeric values.
- For each categorical PMF, add `log pmf_l(value) − log pmf_g(value)`.
- Missing components simply contribute nothing — a **log-ratio of `0.0`** ("no evidence", neither
  reward nor penalty; equivalently a density ratio `l/g = 1`, since `log(1) = 0` — this is *not*
  `log(0)`, which would be `−∞`). So a partially-mature group is still rankable.

**Example.** Suppose a group has numeric KDEs for `(C, gamma)` and a PMF for `kernel`. For a
candidate `{'C': 0.7, 'gamma': 0.018, 'kernel': 'rbf'}`:

```
score = (log l_num([0.7, 0.018]) - log g_num([0.7, 0.018]))   # joint numeric term
      + (log pmf_l_kernel('rbf')  - log pmf_g_kernel('rbf'))    # categorical term
```

A positive `score` means the candidate resembles the good group more than the bad group.

This same `ParamGroupModel` is reused by both surrogates: `HPO_TPE` fits **one** group over the
whole flat genotype; `CASH_TPE` fits **one per `(architecture, node)`**.

---

## `BaseTPE` — the abstract surrogate

[tpe.py:189](Source/Base/tpe.py#L189). Holds only the flavor-independent parts: the good/bad split
and the candidate-ranking helpers. Concrete density modeling is deferred to subclasses through two
abstract methods.

### Construction and the good/bad split

`__init__(gamma)` stores `gamma ∈ (0, 1)`, the fraction of history treated as "good".

`split_samples(samples)` ([tpe.py:214](Source/Base/tpe.py#L214)):

- Requires at least 2 samples.
- Sorts by `get_val_performance()` **ascending** (lowest first). Note: the surrogate treats the
  objective as a **minimization**, so the EA feeds it *negated* validation scores (it maximizes,
  TPE minimizes) — see the domain `ea.md`/`tpe.md`. After negation, "lowest" = "best".
- The good group is the first `ceil(len(samples) * gamma)` samples (at least 1); the rest are bad.

**Example.** `gamma = 0.25`, 8 samples → `split_idx = ceil(8 * 0.25) = 2`: the 2 best-objective
samples are "good", the other 6 "bad".

### Ranking helpers

Both call the abstract `score_candidates` and break ties with the EA's `rng`:

- `suggest_one(candidates, rng)` ([tpe.py:235](Source/Base/tpe.py#L235)) — index of the highest
  scorer, uniformly random among ties. This is what the EA calls to pick the best of its
  `num_offspring` proposals.
- `suggest_top_k(candidates, k, rng)` ([tpe.py:257](Source/Base/tpe.py#L257)) — the top `k`
  indices, sampling without replacement across the tie boundary.

### The shared fitting helper

`_fit_param_group(good_param_dicts, bad_param_dicts, param_specs, rng)`
([tpe.py:307](Source/Base/tpe.py#L307)) builds one `ParamGroupModel` from a group's good/bad
observations and its `{param: spec}` metadata:

- Numeric params (`int`/`float`) get a joint `MultivariateKDE` pair — **only if both groups have
  strictly more observations than numeric dimensions** (the `n > d` rule); a singular fit even
  after jitter leaves the numeric part unavailable.
- Categorical params (`cat`/`bool`) get a `CategoricalPMF` pair whenever both groups are non-empty.
- Returns the model, or `None` if no evidence at all could be fit.

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
