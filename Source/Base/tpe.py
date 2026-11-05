import numpy as np
from abc import ABC, abstractmethod
from .individual import Individual
from collections import Counter
from scipy.stats import gaussian_kde
from typing import Tuple, Dict, List, Union, Iterable, Optional, Any
from typeguard import typechecked

class _FlooredGaussianKDE(gaussian_kde):
    """
    ``scipy.stats.gaussian_kde`` whose kernel covariance has a per-dimension minimum bandwidth.

    The kernel covariance is ``data_covariance * factor**2 + diag(min_bandwidth**2)``. Adding the
    floor (rather than taking an elementwise max) keeps any correlation the data does show and
    guarantees a positive-definite covariance, even when the data are constant in some dimension
    (where the plain KDE collapses to zero width and its log-density off that line is ~ -1e20).
    """
    def __init__(self, dataset: np.ndarray, min_bandwidth: np.ndarray, bw_method: str = "silverman"):
        # must exist before super().__init__, which calls _compute_covariance
        self._min_bandwidth = np.asarray(min_bandwidth, dtype=float)
        super().__init__(dataset, bw_method=bw_method)

    def _compute_covariance(self):
        self.factor = self.covariance_factor()
        data_cov = np.atleast_2d(np.cov(self.dataset, rowvar=1, bias=False, aweights=self.weights))
        self.covariance = data_cov * self.factor ** 2 + np.diag(self._min_bandwidth ** 2)
        self.cho_cov = np.linalg.cholesky(self.covariance).astype(np.float64)
        self.log_det = 2 * np.log(np.diag(self.cho_cov * np.sqrt(2 * np.pi))).sum()


@typechecked
class MultivariateKDE:
    """
    Multivariate Gaussian KDE with floor to prevent zero likelihood.
    This wrapper over 'scipy.stats.gaussian_kde' supports optional bandwidth scaling, an optional
    per-dimension minimum bandwidth, and enforces a minimum density 'eps' to avoid zero likelihood
    values.
    """
    def __init__(self, data: np.ndarray, rng: np.random.Generator, bw_factor: float=1.0, eps: float=1e-12,
                 min_bandwidth: Optional[np.ndarray] = None):
        """
        Parameters:
        - data : ndarray, shape=(d,n)
            'd' (row) is the number of dimensions or features
            'n' (column) is the number of samples
            For example, np.array([[1, 2, 3],     # dimension 1 values
                                   [4, 5, 6]])    # dimension 2 values
        - bw_factor: float
            Scaling factor for the KDE bandwidth.
            The base bandwidth is chosen using Silverman's rule, then scaled by this factor.
        - eps : float
            Minimum floor value for the estimated density. Returned densities will be at least this value.
        - min_bandwidth : ndarray, shape=(d,), optional
            Per-dimension minimum kernel standard deviation, added in quadrature to the Silverman
            bandwidth (see ``_FlooredGaussianKDE``). With a floor the covariance is always positive
            definite, so the jitter fallback below is never needed. ``None`` keeps the plain
            Silverman KDE.
        """
        d, n = data.shape
        # Guaranteed singular covariance if more dimensions than data points
        if n <= d:
            raise ValueError(f"Not enough samples for multivariate KDE: n={n}, d={d}")

        if min_bandwidth is not None:
            assert np.shape(min_bandwidth) == (d,), "min_bandwidth must have one entry per dimension."
            assert np.all(np.asarray(min_bandwidth) > 0), "min_bandwidth entries must be positive."
            # bw_factor scales the data-driven (Silverman) part only; the floor stays absolute
            self.kde = _FlooredGaussianKDE(data, np.asarray(min_bandwidth, dtype=float))
            if bw_factor != 1.0:
                self.kde.set_bandwidth(self.kde.factor * bw_factor)
            self.eps = eps
            return

        # Rank checking is too strict/is a proxy, just try
        try:
            self.kde = gaussian_kde(data, bw_method='silverman')
        except np.linalg.LinAlgError: # Likely from Cholesky
            # Add and scale noise relative to each feature's spread
            per_dim_std = np.std(data, axis=1, ddof=1) # (d, )
            per_dim_std = np.maximum(per_dim_std, 1e-6) # make sure std is not 0
            Z = rng.normal(size=data.shape) # this should be the same shape as data
            noise = Z * (1e-3 * per_dim_std[:, None])
            data = data + noise

            try:
                self.kde = gaussian_kde(data, bw_method='silverman')
            except np.linalg.LinAlgError:
                # We could drop model params that are near-constant here (by checking to see if std is close to 0)
                # but that would require overhauling the system
                # We could also build Univariate KDEs as backup, but again, overhaul
                raise ValueError("Unable to construct Multivariate KDE, even after adding noise.")

        # self.kde = gaussian_kde(data, bw_method='silverman')
        self.kde.set_bandwidth(self.kde.factor * bw_factor)
        self.eps = eps

    def __repr__(self):
        return f"MultivariateKDE(dims={self.kde.d}, samples={self.kde.n})"

    def pdf(self, vec: Union[np.ndarray, List]) -> np.ndarray:
        """
        Evaluate the KDE probability density function at given points.

        Parameters:
        - vec : array-like, shape (d,) or (d, m)
            Points at which to evaluate the density. Can be a single d-dimensional point
            or multiple points as columns in a (d, m) array.
        """
        vec = np.asarray(vec) # reshape to (d, )
        if vec.ndim == 1:
            vec = vec[:, None] # reshape to (d, 1)
        # Returns an array of shape (m,), corresponding to 1 density value per point
        return np.maximum(self.kde.pdf(vec), self.eps)

    def logpdf(self, vec: Union[np.ndarray, List]) -> np.ndarray:
        """
        Evaluate the KDE log probability density at given points.

        Parameters:
        - vec : array-like, shape (d,) or (d, m)
            Points at which to evaluate the log density. Can be a single d-dimensional point
            or multiple points as columns in a (d, m) array.
        """
        vec = np.asarray(vec)
        if vec.ndim == 1:
            vec = vec[:, None]
        return self.kde.logpdf(vec)

    def sample(self, rng: np.random.Generator, n_samples: int = 1) -> np.ndarray:
        """
        Sample 'n_samples' new points from the estimated distribution.
        Returns a matrix of shape (dimensions, n_samples)
        """
        # For reproducibility, pull an int seed from the generator
        return self.kde.resample(size=n_samples, seed=rng) # shape (dimensions, n_samples)

@typechecked
class CategoricalPMF:
    """
    Categorical probability mass function with Laplace smoothing to avoid zero probabilities.
    Computes smoothed category probabilities based on observed frequencies,
    ensuring all categories have non-zero likelihood (with smoothing factor 'alpha').
    """

    def __init__(self, values: Iterable[str | bool | None], all_categories: Iterable[str | bool | None],
                 alpha = 1.0):
        """
        Parameters:
        - values: Iterable[str]
            List or iterable of observed categorical values.
        - all_categories: List[str] | List[bool] | Tuple[str] | Tuple[bool]
            The full list (or tuple) of possible categories to support in the distribution.
        - alpha: float
            Laplace smoothing parameter. Higher values increase the uniformity of the distribution.
        """
        # self.rng = rng

        self.all_categories = all_categories

        # Count the frequency of each category in 'values'
        counts = Counter(values)
        total = sum(counts[c] + alpha for c in all_categories)
        self.prob: Dict = {c: (counts[c] + alpha) / total for c in all_categories}
        self.eps = 1e-12

    def pmf(self, x) -> float:
        """
        Evaluate the smoothed probability of a category 'x' if it was part of
        'all_categories'; otherwise, returns 'self.eps' to avoid zero likelihood.

        Parameters:
        - x: Any
            The category to evaluate.
        """
        return self.prob.get(x, self.eps)

    def sample(self, rng: np.random.Generator, n_samples: int = 1) -> List[str | bool | None]:
        """
        Sample n categories from the categorical PMF.

        Parameters:
        - rng: np.random.Generator
        - n_samples (int): Number of samples to draw.

        Returns:
            List of sampled categories (length = n_samples)
        """
        probabilities = list(self.prob.values())
        samples = rng.choice(self.all_categories, size=n_samples, p=probabilities)
        return samples.tolist()

@typechecked
class ParamGroupModel:
    """
    Density models for ONE group of observations over a flat parameter sub-space.

    Bundles an optional multivariate Gaussian KDE pair over the group's numeric parameters with
    an optional smoothed PMF pair per categorical/bool parameter. Any component may be
    unavailable when there is not enough evidence to fit it (too few observations for the KDE, or
    a group with no observations); ``log_ratio`` simply skips unavailable components, which makes
    their contribution neutral (log-ratio 0) rather than rewarding or penalizing the candidate.

    Used both for a flat single-model space (one global group) and, in the pipeline-aware CASH
    variant, for each ``(architecture, node)`` group. Build instances via
    ``BaseTPE._fit_param_group`` rather than directly.
    """
    def __init__(self, numeric_names: List[str],
                 multi_l: Optional[MultivariateKDE], multi_g: Optional[MultivariateKDE],
                 cat_l: Dict[str, CategoricalPMF], cat_g: Dict[str, CategoricalPMF],
                 log_names: Optional[Iterable[str]] = None,
                 clip: Optional[float] = None):
        self.numeric_names = numeric_names  # order of dimensions in the numeric KDEs
        self.multi_l = multi_l              # good numeric KDE (None if unavailable)
        self.multi_g = multi_g              # bad numeric KDE (None if unavailable)
        self.cat_l = cat_l                  # {param: good PMF}
        self.cat_g = cat_g                  # {param: bad PMF}
        self.log_names = set(log_names or ())  # numeric params modeled on a log scale
        assert clip is None or clip > 0.0, "clip must be positive (or None for no clipping)."
        self.clip = clip                    # |log_ratio| cap for the whole group (None = no cap)

    def numeric_vector(self, params: Dict[str, Any]) -> List[float]:
        """
        Numeric parameter values in KDE dimension order, with ``log``-flagged parameters mapped
        to their natural logarithm so the KDE models them on the scale they are sampled on.
        """
        return [float(np.log(params[name])) if name in self.log_names else float(params[name])
                for name in self.numeric_names]

    def has_evidence(self) -> bool:
        """True if at least one density model (numeric KDE or any categorical PMF) was fit."""
        return (self.multi_l is not None and self.multi_g is not None) or bool(self.cat_l)

    def log_ratio(self, params: Dict[str, Any]) -> float:
        """
        Sum of available good/bad log-density ratios for a candidate's parameter values.

        The numeric KDEs (if available) contribute one joint ``log l - log g`` term, evaluated
        directly in log space so the ratio stays informative far into the density tails; each
        categorical PMF contributes its own ``log l - log g``, with ``pmf`` flooring its output
        at a small epsilon so those logs are always finite.

        When ``clip`` is set, the group's total is clipped to ``[-clip, clip]`` so a single
        group's parameter evidence (one CASH node, or HPO's whole flat genotype) cannot outweigh
        everything else in the score, and huge and tiny terms are never summed in floating point.
        """
        score = 0.0
        if self.multi_l is not None and self.multi_g is not None and self.numeric_names:
            num_vals = self.numeric_vector(params)
            # logpdf returns a length-1 array for a single point; take the scalar.
            score += float(self.multi_l.logpdf(num_vals)[0] - self.multi_g.logpdf(num_vals)[0])
        for name, pmf_l in self.cat_l.items():
            score += float(np.log(pmf_l.pmf(params[name])) - np.log(self.cat_g[name].pmf(params[name])))
        if self.clip is not None:
            score = float(np.clip(score, -self.clip, self.clip))
        return score


@typechecked
class BaseTPE(ABC):
    """
    Abstract base for the Tree-structured Parzen Estimator (TPE) surrogate used to guide the EA.

    Holds only the flavor-independent machinery: the good/bad split and the candidate-ranking
    helpers. Ranking is driven by the abstract ``score_candidates`` acquisition hook (higher =
    more promising), so the concrete density modeling lives entirely in a subclass:

      * ``Source.HPO.tpe.HPO_TPE`` models a single flat ``ModelParams`` space (one global
        multivariate KDE + per-categorical PMFs).
      * the planned pipeline-aware CASH variant models the joint pipeline architecture plus
        architecture-conditioned parameter models (see Source/CASH/TPE_pipeline_EA_plan.md).

    Both fit good/bad density models from evaluated history and score EA-generated candidates by
    how strongly they resemble historically good rather than bad solutions.
    """
    def __init__(self, gamma: float, log_ratio_clip: Optional[float] = 5.0):
        """
        Parameters:
            gamma (float): Fraction of samples considered "good".
            log_ratio_clip (Optional[float]): Cap on the absolute log-ratio of each parameter group
                (see ``ParamGroupModel.log_ratio``). Defaults to 5.0; ``None`` disables clipping.
        """
        assert 0.0 < gamma < 1.0, "gamma must be in (0, 1)"
        assert log_ratio_clip is None or log_ratio_clip > 0.0, "log_ratio_clip must be positive or None."
        self.gamma = gamma # splitting parameter
        self.log_ratio_clip = log_ratio_clip

    def split_samples(self, samples: List[Individual]) -> Tuple[List[Individual], List[Individual]]:
        """
        Splits a given sample set into 'good' and 'bad' groups based on
        the objective.

        Parameters:
            samples (List[Individual]): The sample set to split.

        Returns:
            Tuple[List[Individual], List[Individual]]: a tuple containing the good and bad sample groups.
        """
        if len(samples) < 2:
            raise RuntimeError("Need at least 2 samples before TPE can fit.")

        # Sort population/samples set (lowest/best first)
        samples.sort(key=lambda o: o.get_val_performance())
        split_idx = max(1, int(np.ceil(len(samples) * self.gamma)))
        good_samples = samples[:split_idx]
        bad_samples = samples[split_idx:]
        return good_samples, bad_samples

    def suggest_one(self, candidates: List[Dict], rng: np.random.Generator) -> int:
        """
        Suggest the top candidate based on the acquisition score.

        Parameters:
            candidates (List[Dict]): Candidate parameter dictionaries to rank.
            rng (np.random.Generator): Random number generator for tie-breaking.

        Returns:
            int: Index of the best candidate in the original candidates list.
        """
        scores = self.score_candidates(candidates)

        # find max score from scores
        best_index = int(np.argmax(scores))

        # collect all indices from candidates with the best score
        best_indices = [i for i, score in enumerate(scores) if score == scores[best_index]]

        # randomly select one of the best indices
        return int(rng.choice(best_indices))

    def suggest_top_k(self, candidates: List[Dict], k: int, rng: np.random.Generator) -> List[int]:
        """
        Suggest the top k candidates based on the acquisition score.
        Handles ties by randomly sampling among candidates with equal scores.

        Parameters:
            candidates (List[Dict]): Candidate parameter dictionaries to rank.
            k (int): Number of top candidates to return.
            rng (np.random.Generator): Random number generator for tie-breaking.

        Returns:
            List[int]: Indices of the top k candidates in the original candidates list.
        """
        if k > len(candidates):
            k = len(candidates)

        scores = self.score_candidates(candidates)

        # Create list of (index, score) tuples
        indexed_scores = list(enumerate(scores))

        # Sort by score (descending)
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Find the k-th highest score (or tie boundary)
        if k == len(candidates):
            return [idx for idx, _ in indexed_scores]

        kth_score = indexed_scores[k-1][1]

        # Collect all candidates with scores >= k-th score
        candidates_above_threshold = [idx for idx, score in indexed_scores if score > kth_score]
        candidates_at_threshold = [idx for idx, score in indexed_scores if score == kth_score]

        # If we have exactly k candidates above threshold, return them
        if len(candidates_above_threshold) == k:
            return candidates_above_threshold

        # If we have fewer than k above threshold, we need to sample from the tie
        remaining_slots = k - len(candidates_above_threshold)

        # Randomly sample from candidates at the threshold score
        sampled_at_threshold = rng.choice(
            candidates_at_threshold,
            size=remaining_slots,
            replace=False
        ).tolist()

        return candidates_above_threshold + sampled_at_threshold

    def _fit_param_group(self, good_param_dicts: List[Dict[str, Any]], bad_param_dicts: List[Dict[str, Any]],
                         param_specs: Dict[str, Any], rng: np.random.Generator) -> Optional[ParamGroupModel]:
        """
        Fit the density models for one group of observations over a flat parameter sub-space.

        ``param_specs`` is a ``{param_name: {"type": ..., "bounds": ...}}`` mapping (a
        ``ModelParams.param_space``, or one node/component's parameter dict in the pipeline
        space). Numeric ('int'/'float') parameters are modeled jointly with a MultivariateKDE
        pair, built to stay well-behaved on small, repetitive histories:

          * **Unique vectors.** Each group's numeric vectors are deduplicated before fitting. Many
            history entries can share one component's exact settings (crossover copies whole
            nodes; mutation leaves most genes untouched), and counting one setting several times
            would concentrate the density on it.
          * **n > d on unique vectors.** The KDE pair is fit only when BOTH groups have strictly
            more UNIQUE vectors than numeric dimensions; otherwise numeric evidence is unavailable.
          * **Minimum bandwidth.** Each dimension's kernel std is floored at
            ``range / min(100, 1 + n_unique)`` (Optuna's TPE rule), where ``range`` is the
            parameter's bound width on the scale it is modeled on (log scale for ``log=True``).
            Without it, a dimension that is constant in a group collapses the KDE to zero width
            and log-densities off that line reach ~ -1e20.

        Categorical ('cat'/'bool') parameters get a smoothed PMF pair whenever both groups are
        non-empty (from the raw, non-deduplicated observations). The returned model clips its
        log-ratio at ``self.log_ratio_clip``.

        Returns a ``ParamGroupModel`` carrying whatever evidence could be fit, or ``None`` if no
        evidence at all was available (so callers can skip storing it).
        """
        numeric_names = [n for n, s in param_specs.items() if s["type"] in ("int", "float")]
        cat_names = [n for n, s in param_specs.items() if s["type"] in ("cat", "bool")]
        log_names = {n for n in numeric_names if param_specs[n].get("log", False)}

        def numeric_row(name, dicts):
            return [float(np.log(d[name])) if name in log_names else float(d[name]) for d in dicts]

        def unique_columns(arr):
            # arr is (d, n); keep each distinct numeric vector (column) once
            return np.unique(arr.T, axis=0).T

        def bound_range(name):
            lo, hi = param_specs[name]["bounds"]
            if name in log_names:
                return float(np.log(hi) - np.log(lo))
            return float(hi - lo)

        multi_l = multi_g = None
        if numeric_names and good_param_dicts and bad_param_dicts:
            d = len(numeric_names)
            good_arr = unique_columns(np.array([numeric_row(n, good_param_dicts) for n in numeric_names], dtype=float))
            bad_arr = unique_columns(np.array([numeric_row(n, bad_param_dicts) for n in numeric_names], dtype=float))
            # MultivariateKDE requires strictly more (unique) samples than dimensions in each group.
            if good_arr.shape[1] > d and bad_arr.shape[1] > d:
                ranges = np.array([bound_range(n) for n in numeric_names], dtype=float)
                # guard degenerate bounds (lo == hi) so the floor stays strictly positive
                ranges = np.maximum(ranges, 1e-12)

                def floor(n_unique):
                    return ranges / min(100, 1 + n_unique)

                try:
                    multi_l = MultivariateKDE(good_arr, rng, min_bandwidth=floor(good_arr.shape[1]))
                    multi_g = MultivariateKDE(bad_arr, rng, min_bandwidth=floor(bad_arr.shape[1]))
                except (ValueError, np.linalg.LinAlgError):
                    # Should not happen with a floor (covariance is positive definite), but stay safe.
                    multi_l = multi_g = None

        cat_l: Dict[str, CategoricalPMF] = {}
        cat_g: Dict[str, CategoricalPMF] = {}
        if cat_names and good_param_dicts and bad_param_dicts:
            for n in cat_names:
                bounds = param_specs[n]["bounds"]
                cat_l[n] = CategoricalPMF([gd[n] for gd in good_param_dicts], bounds)
                cat_g[n] = CategoricalPMF([bd[n] for bd in bad_param_dicts], bounds)

        model = ParamGroupModel(numeric_names, multi_l, multi_g, cat_l, cat_g, log_names,
                                clip=self.log_ratio_clip)
        return model if model.has_evidence() else None

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fit the good/bad density models from evaluated history.

        The concrete signature and return type are flavor-specific (see subclass docs): the flat
        HPO variant takes ``(samples, param_space, rng)`` and returns ``None``; the pipeline-aware
        variant returns a ``bool`` indicating whether usable architecture-level guidance exists.
        """
        raise NotImplementedError("Subclasses must implement the 'fit' method.")

    @abstractmethod
    def score_candidates(self, candidates: List[Dict]) -> np.ndarray:
        """
        Return one acquisition score per candidate; higher means more characteristic of
        historically good (rather than bad) configurations. ``suggest_one`` and
        ``suggest_top_k`` rank candidates by this score.
        """
        raise NotImplementedError("Subclasses must implement the 'score_candidates' method.")
