import numpy as np
from .individual import Individual
from .model_param_space import ModelParams
from collections import Counter
from scipy.stats import gaussian_kde
from typing import Tuple, Dict, List, Union, Iterable, Any
from typeguard import typechecked

@typechecked
class MultivariateKDE:
    """
    Multivariate Gaussian KDE with floor to prevent zero likelihood.
    This wrapper over 'scipy.stats.gaussian_kde' supports optional bandwidth scaling
    and enforces a minimum density 'eps' to avoid zero likelihood values.
    """
    def __init__(self, data: np.ndarray, rng: np.random.Generator, bw_factor: float=1.0, eps: float=1e-12):
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
        """
        # self.rng = rng

        d, n = data.shape
        # Guaranteed singular covariance if more dimensions than data points
        if n <= d:
            raise ValueError(f"Not enough samples for multivariate KDE: n={n}, d={d}")

        # If there is only 1 sample (n=1), KDE will fail
        # if n == 1:
        #     data = np.hstack([data, data + 1e-3]) # duplicate it slightly

        # cov = np.cov(data, rowvar=True, bias=False)
        # rank = np.linalg.matrix_rank(cov)

        # Another full rank check - add small noise to escape colinearity and constant dimensions
        # if rank < d:
        #     # Scale noise relative to each feature's spread
        #     per_dim_std = np.std(data, axis=1, ddof=1) # (d, )
        #     per_dim_std = np.maximum(per_dim_std, 1e-6) # make sure std is not 0
        #     Z = rng.normal(size=data.shape) # this should be the same shape as data
        #     noise = Z * (1e-3 * per_dim_std[:, None])            
        #     data = data + noise

        #     # Recheck rank
        #     cov = np.cov(data, rowvar=True, bias=False)
        #     if np.linalg.matrix_rank(cov) < d:
        #         raise ValueError("Still rank-deficient after adding noise.")

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
        # self.kde.set_bandwidth(self.kde.factor * bw_factor)
        # self.eps = eps

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
class TPE:
    """
    Tree-structured Parzen Estimator (TPE) Solver for hyperparameter optimization.
    This class supports both categorical and numeric parameters,
    and uses kernel density estimation (KDE) and probability mass functions (PMFs)
    to model the likelihood of good and bad configurations.
    """
    def __init__(self, gamma: float):
        """
        Parameters:
            gamma (float): Fraction of samples considered "good".
        """

        self.gamma = gamma # splitting parameter

        # Fitted distributions
        self.multi_l: MultivariateKDE = None # good, numeric
        self.multi_g: MultivariateKDE = None  # bad, numeric
        self.cat_l: Dict[str, CategoricalPMF] = {} # good, categorical
        self.cat_g: Dict[str, CategoricalPMF] = {} # bad, categorical

    def sample(self, num_samples: int, param_space: ModelParams, rng: np.random.Generator) -> List[Dict]:
        """
        Returns 'num_samples' Individuals.
        For each Individual's params, sample from the "good" MultivariateKDE and CategoricalPMFs separately,
        then reassemble into a full set of hyperparameters.
        """
        numeric_params = {
            **param_space.get_params_by_type('int'),
            **param_space.get_params_by_type('float'),
        }
        categorical_params = {
            **param_space.get_params_by_type('cat'),
            **param_space.get_params_by_type('bool')
        }

        # Validate that fitted models exist for parameter types present in the space
        if numeric_params:
            assert self.multi_l is not None and self.multi_g is not None, \
                "MultivariateKDE models must be fitted when numeric parameters exist."

        if categorical_params:
            assert len(self.cat_l) > 0 and len(self.cat_g) > 0, \
                "CategoricalPMF models must be fitted when categorical parameters exist."

        numeric_params_names = list(numeric_params.keys())

        # Sample from the good numeric distribution
        multi_samples = self.multi_l.sample(rng=rng, n_samples=num_samples) # shape (dimensions, n_samples)
        assert multi_samples.shape[0] == len(numeric_params_names)
        assert multi_samples.shape[1] == num_samples

        # Align numeric parameter names to each dimension
        params = {
            name: list(multi_samples[i])
            for i, name in enumerate(numeric_params_names)
        }

        # Make sure parameters are rounded if int and within bounds
        for name, info in param_space.param_space.items():
            if info["type"] == "int":
                params[name] = [
                    int(np.clip(val, *info['bounds']))
                    for val in params[name]
                    ]
            if info["type"] == "float":
                params[name] = [
                    float(np.clip(val, *info['bounds']))
                    for val in params[name]
                ]

        # Sample from the good categorical distribution
        for name, dist in self.cat_l.items():
            params[name] = dist.sample(rng=rng, n_samples=num_samples)

        assert all(len(v) == num_samples for v in params.values())

        samples: List[Dict] = []
        for i in range(num_samples):
            # Map name to a single value
            ind_params = {name: params[name][i] for name in param_space.param_space}
            samples.append(ind_params)

        assert(len(samples) == num_samples)
        return samples

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
        split_idx = max(1, int(len(samples) * self.gamma))
        good_samples = samples[:split_idx]
        bad_samples = samples[split_idx:]
        return good_samples, bad_samples

    def fit(self, samples: List[Individual], param_space: ModelParams, rng: np.random.Generator) -> None:
        """
        Fit probabilistic models (KDEs and PMFs) to the good and bad sample groups.

        Parameters:
        """
        good_samples, bad_samples = self.split_samples(samples)

        numeric_params = {
            **param_space.get_params_by_type('int'),
            **param_space.get_params_by_type('float'),
        }
        categorical_params = {
            **param_space.get_params_by_type('cat'),
            **param_space.get_params_by_type('bool')
        }

        # For each sample set, extract values of numeric hyperparameters
        # Format shape (n_params, n_samples): [[value11, value12,...], [value21, value22, ...], ...]
        # Each parameter has its own row
        good_num_samples = np.array([[o.get_params()[param_name] for o in good_samples]
                            for param_name in numeric_params])
        bad_num_samples = np.array([[o.get_params()[param_name] for o in bad_samples]
                            for param_name in numeric_params])

        # Fit Multivariate KDEs
        self.multi_l = MultivariateKDE(good_num_samples, rng)
        self.multi_g = MultivariateKDE(bad_num_samples, rng)

        # Fit independent PMFs
        self.cat_l.clear()
        self.cat_g.clear()
        # Construct 2 PMFs (good and bad) for each categorical parameter
        # Format: {param_name: CategoricalPMF}
        self.cat_l = {
            param_name: CategoricalPMF(
                # Extract categorical values from samples in (d, n) format
                values = [o.get_params()[param_name] for o in good_samples],
                all_categories = info["bounds"]
            )
            for param_name, info in categorical_params.items()
        }

        self.cat_g = {
            param_name : CategoricalPMF(
                values = [o.get_params()[param_name] for o in bad_samples],
                all_categories = info["bounds"]
            )
            for param_name, info in categorical_params.items()
        }
        return

    def expected_improvement(self, param_space: ModelParams, candidates: List[Dict]) -> np.ndarray:
        """
        Compute the expected improvement (EI) for a list of candidate individuals.

        Parameters:
        - candidates: List[Individual]
            Candidate individuals to evaluate

        Returns an array of EI scores, one per candidate.
        """
        ei_scores = []

        numeric_params = {
            **param_space.get_params_by_type('int'),
            **param_space.get_params_by_type('float'),
        }

        for params in candidates:
            # Numeric contribution (multivariate)
            if numeric_params:
                num_vals = [params[param_name] for param_name in numeric_params] # (, d_num)
                # 'num_vals' gets reshaped into (d_num, 1) here
                l_num = float(self.multi_l.pdf(num_vals)) # a single density value
                g_num = float(self.multi_g.pdf(num_vals))
            else: # If no numeric parameters exist, no contribution
                l_num = g_num = 1.0

            # Categorical contribution (product of per-dim PMFs)
            l_cat = g_cat = 1.0
            # PMFs are univariate
            for param_name, pmf_l in self.cat_l.items():
                lx = pmf_l.pmf(params[param_name]) # a single density
                gx = self.cat_g[param_name].pmf(params[param_name])
                l_cat *= lx
                g_cat *= gx

            # To avoid NaN
            if (g_num * g_cat) <= 1e-12:
                ei_scores.append(0.0)
            else:
                ei_scores.append((l_num * l_cat) / (g_num * g_cat))
        return np.asarray(ei_scores)

    def suggest_one(self, param_space: ModelParams, candidates: List[Dict], rng: np.random.Generator) -> int:
        """
        Suggest the top candidate based on expected improvement.

        Parameters:
            candidates (List[Individual]): Candidate individuals to rank.
            num_top_cand (int): Number of top candidates to return.

        Returns:
            int: Index of the best candidate in the original candidates list.
        """
        scores = self.expected_improvement(param_space, candidates)

        # find max score from scores
        best_index = int(np.argmax(scores))

        # collect all indices from candidates with the best score
        best_indices = [i for i, score in enumerate(scores) if score == scores[best_index]]

        # randomly select one of the best indices
        return int(rng.choice(best_indices))

    def suggest_top_k(self, param_space: ModelParams, candidates: List[Dict], k: int, rng: np.random.Generator) -> List[int]:
        """
        Suggest the top k candidates based on expected improvement scores.
        Handles ties by randomly sampling among candidates with equal scores.

        Parameters:
            param_space (ModelParams): The parameter space definition.
            candidates (List[Dict]): Candidate parameter dictionaries to rank.
            k (int): Number of top candidates to return.
            rng (np.random.Generator): Random number generator for tie-breaking.

        Returns:
            List[int]: Indices of the top k candidates in the original candidates list.
        """
        if k > len(candidates):
            k = len(candidates)

        scores = self.expected_improvement(param_space, candidates)

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