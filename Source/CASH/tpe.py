import itertools
import numpy as np
from typing import Dict, List, Tuple, Any
from typeguard import typechecked

from Source.Base.individual import Individual
from Source.Base.tpe import BaseTPE, CategoricalPMF, ParamGroupModel


# A pipeline_space maps each node -> component -> that component's flat parameter space.
# A component's parameter space is a {param_name: {"type": ..., "bounds": ...}} dict (a
# ModelParams.param_space), or {} for parameter-free components (e.g. "passthrough", MinMaxScaler).
PipelineSpace = Dict[str, Dict[str, Dict[str, Any]]]

# An architecture is the joint choice of one component per node, as an ordered tuple of
# (node_name, component_name) pairs following pipeline_space's node order. It is hashable, so it
# keys both parameter_models and (via a string encoding) the architecture PMFs.
Architecture = Tuple[Tuple[str, str], ...]

# A candidate is the nested pipeline representation:
#   {node_name: {"name": component_name, "params": {param: value, ...}}, ...}
Candidate = Dict[str, Dict[str, Any]]


@typechecked
class CASH_TPE(BaseTPE):
    """
    Pipeline-aware TPE surrogate for the CASH problem (see Source/CASH/TPE_pipeline_EA_plan.md).

    Models the COMPLETE pipeline architecture jointly (one smoothed categorical PMF pair over the
    joint choice of scaler + feature-engineering + feature-selection + predictor), then conditions
    parameter evidence on both the architecture and the pipeline node. A candidate's acquisition
    score is the architecture good/bad log-ratio plus, for each node whose ``(architecture, node)``
    parameter model has been fit, that model's parameter log-ratio. Unavailable parameter models
    contribute nothing (log 0), so partially-mature architectures remain rankable against fully
    mature ones -- an unavailable model means "no evidence", not "average" or "bad".

    The implementation is fully generic: the allowed nodes, components, and component parameters
    all come from ``pipeline_space`` at construction, so there are no scaler/transformer/predictor
    specific branches here. Adding a node (e.g. making feature scaling an evolved decision) is a
    metadata change only. Candidate generation lives in the EA; this class only fits history and
    ranks EA-generated candidates.
    """
    def __init__(self, gamma: float, pipeline_space: PipelineSpace):
        """
        Parameters:
            gamma (float): Fraction of history considered "good".
            pipeline_space (PipelineSpace): {node: {component: {param: spec}}} metadata describing
                every allowed pipeline node, its component choices, and each component's flat
                parameter space. Node order is preserved and defines architecture-key order.
        """
        super().__init__(gamma)

        self.pipeline_space = pipeline_space
        self.node_names: Tuple[str, ...] = tuple(pipeline_space.keys())

        # Every architecture the space can express (Cartesian product of per-node component
        # choices). Used as the categorical support for the architecture PMFs so unseen-but-legal
        # architectures still receive smoothed (non-zero) probability.
        self.all_architectures: List[Architecture] = self._enumerate_architectures()

        # Fitted state (populated by fit()).
        self.architecture_l: CategoricalPMF = None  # good architecture PMF
        self.architecture_g: CategoricalPMF = None  # bad architecture PMF
        self.parameter_models: Dict[Tuple[Architecture, str], ParamGroupModel] = {}

    # ------------------------------------------------------------------ helpers

    def _enumerate_architectures(self) -> List[Architecture]:
        """Enumerate all architectures as the Cartesian product of each node's component choices."""
        per_node_choices = [
            [(node, component) for component in self.pipeline_space[node].keys()]
            for node in self.node_names
        ]
        return [tuple(combo) for combo in itertools.product(*per_node_choices)]

    def get_architecture_key(self, candidate: Candidate) -> Architecture:
        """Extract a candidate's joint architecture as an ordered (node, component) tuple."""
        return tuple((node, candidate[node]["name"]) for node in self.node_names)

    @staticmethod
    def _arch_to_str(architecture: Architecture) -> str:
        """
        Encode an architecture tuple as a single string.

        CategoricalPMF's categories are scalars (str/bool/None), so the architecture -- naturally a
        tuple of pairs -- is joined into a stable string to serve as a PMF category. The structured
        tuple is still used everywhere else (parameter_models keys, per-node lookups).
        """
        return "|".join(f"{node}={component}" for node, component in architecture)

    # ------------------------------------------------------------------ fit / score

    def fit(self, samples: List[Individual], rng: np.random.Generator) -> bool:
        """
        Fit the joint-architecture PMFs and the architecture-conditioned parameter models.

        Splits ``samples`` (evaluated history, each exposing its nested pipeline candidate through
        ``get_genotype()`` and its objective through ``get_val_performance()``) into good/bad,
        fits the architecture PMF pair over the full architecture support, then -- for each
        OBSERVED architecture and each node whose component has parameters -- opportunistically
        fits an architecture-conditioned parameter model from that architecture's observations.

        Returns True when architecture-level guidance was built (the EA can then score candidates),
        or False when history cannot yield non-empty good and bad groups (the EA should fall back
        to its normal evaluation policy). Parameter-model maturity does not affect the return value.
        """
        if len(samples) < 2:
            return False

        good_samples, bad_samples = self.split_samples(samples)
        if len(good_samples) == 0 or len(bad_samples) == 0:
            return False

        good_candidates = [o.get_genotype() for o in good_samples]
        bad_candidates = [o.get_genotype() for o in bad_samples]

        # --- Joint architecture PMFs (smoothed over every expressible architecture) ---
        all_arch_strs = [self._arch_to_str(a) for a in self.all_architectures]
        good_arch_strs = [self._arch_to_str(self.get_architecture_key(c)) for c in good_candidates]
        bad_arch_strs = [self._arch_to_str(self.get_architecture_key(c)) for c in bad_candidates]
        self.architecture_l = CategoricalPMF(good_arch_strs, all_arch_strs)
        self.architecture_g = CategoricalPMF(bad_arch_strs, all_arch_strs)

        # --- Architecture-conditioned parameter models (fit opportunistically) ---
        self.parameter_models = {}
        good_keys = [self.get_architecture_key(c) for c in good_candidates]
        bad_keys = [self.get_architecture_key(c) for c in bad_candidates]
        observed_architectures = set(good_keys) | set(bad_keys)

        for architecture in observed_architectures:
            component_by_node = dict(architecture)
            for node in self.node_names:
                param_specs = self.pipeline_space[node][component_by_node[node]]
                if not param_specs:
                    continue  # parameter-free component (passthrough, default-only scaler, ...)

                good_params = [c[node]["params"] for c, k in zip(good_candidates, good_keys) if k == architecture]
                bad_params = [c[node]["params"] for c, k in zip(bad_candidates, bad_keys) if k == architecture]

                model = self._fit_param_group(good_params, bad_params, param_specs, rng)
                if model is not None:
                    self.parameter_models[(architecture, node)] = model

        return True

    def score_candidates(self, candidates: List[Candidate]) -> np.ndarray:
        """
        Score EA-generated candidates by their good/bad log density ratio (higher = more promising).

        Each score is the architecture log-ratio plus the parameter log-ratio of every node whose
        ``(architecture, node)`` model is available; nodes without a fitted model contribute 0.
        ``fit()`` must have returned True before calling this.
        """
        assert self.architecture_l is not None and self.architecture_g is not None, \
            "fit() must be called (and return True) before score_candidates()."

        scores = []
        for candidate in candidates:
            architecture = self.get_architecture_key(candidate)
            arch_str = self._arch_to_str(architecture)

            # Architecture-level evidence -- always available after a successful fit().
            score = float(np.log(self.architecture_l.pmf(arch_str)) - np.log(self.architecture_g.pmf(arch_str)))

            # Architecture-conditioned parameter evidence -- missing models contribute log 0.
            for node in self.node_names:
                model = self.parameter_models.get((architecture, node))
                if model is None:
                    continue
                score += model.log_ratio(candidate[node]["params"])

            scores.append(score)

        return np.asarray(scores)

    # ------------------------------------------------------------------ diagnostics

    def model_availability(self) -> Dict[Tuple[Architecture, str], Dict[str, Any]]:
        """
        Report which architecture-conditioned parameter models currently carry evidence.

        Returns ``{(architecture, node): {"numeric": bool, "categorical": [param names]}}`` for
        every fitted ``(architecture, node)`` model -- useful for debugging and for tracking how
        TPE guidance matures across EA generations.
        """
        availability: Dict[Tuple[Architecture, str], Dict[str, Any]] = {}
        for key, model in self.parameter_models.items():
            availability[key] = {
                "numeric": model.multi_l is not None,
                "categorical": list(model.cat_l.keys()),
            }
        return availability
