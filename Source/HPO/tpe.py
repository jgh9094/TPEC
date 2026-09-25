import numpy as np
from typing import Dict, List, Optional
from typeguard import typechecked

from Source.Base.individual import Individual
from Source.Base.model_param_space import ModelParams
from Source.Base.tpe import BaseTPE, ParamGroupModel


@typechecked
class HPO_TPE(BaseTPE):
    """
    TPE surrogate for single-model hyperparameter optimization over a flat ``ModelParams`` space.

    Models the whole (flat) genotype as ONE parameter group: a multivariate Gaussian KDE pair over
    all numeric genes plus a smoothed PMF pair per categorical/bool gene, bundled in a
    ``ParamGroupModel`` (the same shared machinery the pipeline-aware CASH variant uses per
    ``(architecture, node)`` group). Candidate ranking uses the standard TPE l/g density ratio,
    returned in LOG space (``log l - log g``) for numerical stability -- identical ranking to the
    raw ratio, but products of many densities can't underflow to a common zero and collapse the
    ranking.

    When there is too little history to fit any density model (each numeric KDE needs strictly more
    observations than dimensions in both the good and bad groups), the group model is unavailable
    and every candidate scores 0 -- neutral, "no evidence", so ``suggest_one``/``suggest_top_k``
    fall back to a uniform random pick rather than crashing. A partially-mature group (e.g.
    categorical PMFs available but the numeric KDE not yet) contributes only the evidence it has.

    The pipeline-aware CASH variant instead conditions parameter models on the joint pipeline
    architecture; see Source/CASH/TPE_pipeline_EA_plan.md. Sampling (generating new candidates from
    the fitted "good" distribution) is intentionally absent -- the EA generates candidates and TPE
    only ranks them.
    """
    def __init__(self, gamma: float):
        """
        Parameters:
            gamma (float): Fraction of samples considered "good".
        """
        super().__init__(gamma)

        # The space being modeled -- stashed at fit() so scoring needs no extra argument.
        self.param_space: Optional[ModelParams] = None

        # Fitted density models for the whole flat space (None until fit(); may stay None when
        # history is too thin for any model, in which case scoring is neutral).
        self._model: Optional[ParamGroupModel] = None

    def fit(self, samples: List[Individual], param_space: ModelParams, rng: np.random.Generator) -> None:
        """
        Fit the good/bad density models for the flat parameter space.

        Parameters:
            samples (List[Individual]): Evaluated history to split into good/bad and model.
            param_space (ModelParams): The flat parameter space being optimized. Stored on the
                instance so ``score_candidates`` can read parameter types without re-passing it.
            rng (np.random.Generator): Random generator (used when a KDE needs jitter to fit).
        """
        self.param_space = param_space
        good_samples, bad_samples = self.split_samples(samples)

        good_param_dicts = [o.get_genotype() for o in good_samples]
        bad_param_dicts = [o.get_genotype() for o in bad_samples]

        # One group over the entire flat space; numeric genes get a joint KDE pair (only when both
        # groups have strictly more observations than numeric dimensions), categoricals get PMF
        # pairs. Returns None if no evidence at all could be fit.
        self._model = self._fit_param_group(good_param_dicts, bad_param_dicts, param_space.param_space, rng)
        return

    def score_candidates(self, candidates: List[Dict]) -> np.ndarray:
        """
        Compute the log expected-improvement (``log l - log g``) for each candidate.

        Uses the flat space stashed on ``self.param_space`` at ``fit()`` time. Higher scores
        indicate candidates more characteristic of the "good" group. When no density model matured
        (too little history), every candidate scores 0 (neutral), so ranking helpers pick uniformly
        at random.

        Parameters:
            candidates (List[Dict]): Candidate parameter dictionaries to score.

        Returns:
            np.ndarray: One log-ratio score per candidate.
        """
        assert self.param_space is not None, "fit() must be called before score_candidates()."

        if self._model is None:
            # No evidence -> neutral score for every candidate.
            return np.zeros(len(candidates), dtype=float)

        return np.asarray([self._model.log_ratio(params) for params in candidates])
