import itertools
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
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
    Pipeline-aware TPE surrogate for the CASH problem (see Source/CASH/tpe.md).

    A candidate's acquisition score is the sum of three kinds of good/bad log-ratio evidence:

      1. **Per-node structure terms** (``factorize_structure``): one smoothed categorical PMF pair
         per node over that node's components. Evidence is shared across architectures -- "RF does
         well" helps every architecture that uses RF, seen or not.
      2. **Joint structure term**: one smoothed PMF pair over the joint choice of every node's
         component (the full architecture), weighted by ``joint_weight``. With the per-node terms
         on, this acts as an interaction correction: it is ~0 for unseen architectures and only
         moves the score for architectures actually observed in history (e.g. a scaler/selector
         pair that errors together).
      3. **Parameter terms**: for each node, the log-ratio of the most specific parameter model
         that exists -- the ``(architecture, node)`` model if one was fit, else (with
         ``component_fallback``) the ``(node, component)`` model pooled over every architecture
         that used that component. A node with neither contributes nothing (log-ratio 0).

    Setting ``factorize_structure=False, joint_weight=1.0, component_fallback=False`` reproduces
    the original joint-architecture-only surrogate exactly.

    The implementation is fully generic: the allowed nodes, components, and component parameters
    all come from ``pipeline_space`` at construction, so there are no scaler/transformer/predictor
    specific branches here. Candidate generation lives in the EA; this class only fits history and
    ranks EA-generated candidates.
    """
    def __init__(self, gamma: float, pipeline_space: PipelineSpace,
                 factorize_structure: bool = True,
                 joint_weight: float = 1.0,
                 component_fallback: bool = True,
                 log_ratio_clip: Optional[float] = 5.0):
        """
        Parameters:
            gamma (float): Fraction of history considered "good".
            pipeline_space (PipelineSpace): {node: {component: {param: spec}}} metadata describing
                every allowed pipeline node, its component choices, and each component's flat
                parameter space. Node order is preserved and defines architecture-key order.
            factorize_structure (bool): Add one per-node component log-ratio per node to the
                structure score. Defaults to True.
            joint_weight (float): Weight on the joint-architecture log-ratio. 1.0 adds it in full
                (as an interaction correction when ``factorize_structure`` is on); 0.0 drops it.
                Must be > 0 when ``factorize_structure`` is False, otherwise there is no structure
                evidence at all. Defaults to 1.0.
            component_fallback (bool): When a node has no ``(architecture, node)`` parameter model,
                fall back to a ``(node, component)`` model pooled across architectures. Defaults
                to True.
            log_ratio_clip (Optional[float]): Cap on each node's parameter log-ratio (see
                ``BaseTPE``). Defaults to 5.0; ``None`` disables clipping.
        """
        super().__init__(gamma, log_ratio_clip=log_ratio_clip)
        assert joint_weight >= 0.0, "joint_weight must be non-negative."
        assert factorize_structure or joint_weight > 0.0, \
            "With factorize_structure=False, joint_weight must be > 0 (otherwise no structure evidence)."

        self.pipeline_space = pipeline_space
        self.node_names: Tuple[str, ...] = tuple(pipeline_space.keys())
        self.factorize_structure = factorize_structure
        self.joint_weight = joint_weight
        self.component_fallback = component_fallback

        # Every architecture the space can express (Cartesian product of per-node component
        # choices). Used as the categorical support for the architecture PMFs so unseen-but-legal
        # architectures still receive smoothed (non-zero) probability.
        self.all_architectures: List[Architecture] = self._enumerate_architectures()

        # Fitted state (populated by fit()).
        self.architecture_l: CategoricalPMF = None  # good joint-architecture PMF
        self.architecture_g: CategoricalPMF = None  # bad joint-architecture PMF
        self.node_l: Dict[str, CategoricalPMF] = {}  # {node: good per-node component PMF}
        self.node_g: Dict[str, CategoricalPMF] = {}  # {node: bad per-node component PMF}
        self.parameter_models: Dict[Tuple[Architecture, str], ParamGroupModel] = {}
        self.component_models: Dict[Tuple[str, str], ParamGroupModel] = {}

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
        Fit the structure PMFs and the parameter models from evaluated history.

        Splits ``samples`` (evaluated history, each exposing its nested pipeline candidate through
        ``get_genotype()`` and its objective through ``get_val_performance()``) into good/bad, then
        fits:
          * the joint-architecture PMF pair over the full architecture support (always -- it is
            cheap, and ``score_candidates`` skips it when ``joint_weight == 0``);
          * one per-node component PMF pair per node (always; used when ``factorize_structure``);
          * an ``(architecture, node)`` parameter model for each OBSERVED architecture and each
            node whose component has parameters;
          * with ``component_fallback``, a ``(node, component)`` parameter model for each OBSERVED
            component with parameters, pooled over every architecture that used it.
        Parameter models are stored only when they carry evidence (see ``_fit_param_group``).

        Returns True when structure-level guidance was built (the EA can then score candidates),
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

        # --- Per-node component PMFs (smoothed over each node's own components) ---
        self.node_l, self.node_g = {}, {}
        for node in self.node_names:
            components = list(self.pipeline_space[node].keys())
            self.node_l[node] = CategoricalPMF([c[node]["name"] for c in good_candidates], components)
            self.node_g[node] = CategoricalPMF([c[node]["name"] for c in bad_candidates], components)

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

        # --- Component-level parameter models (pooled across architectures; fallback) ---
        self.component_models = {}
        if self.component_fallback:
            for node in self.node_names:
                observed_components = {c[node]["name"] for c in good_candidates + bad_candidates}
                for component in observed_components:
                    param_specs = self.pipeline_space[node][component]
                    if not param_specs:
                        continue
                    good_params = [c[node]["params"] for c in good_candidates if c[node]["name"] == component]
                    bad_params = [c[node]["params"] for c in bad_candidates if c[node]["name"] == component]
                    model = self._fit_param_group(good_params, bad_params, param_specs, rng)
                    if model is not None:
                        self.component_models[(node, component)] = model

        return True

    def _structure_score(self, candidate: Candidate, architecture: Architecture) -> float:
        """Per-node component log-ratios (if enabled) plus the weighted joint-architecture log-ratio."""
        score = 0.0
        if self.factorize_structure:
            for node in self.node_names:
                component = candidate[node]["name"]
                score += float(np.log(self.node_l[node].pmf(component)) - np.log(self.node_g[node].pmf(component)))
        if self.joint_weight > 0.0:
            arch_str = self._arch_to_str(architecture)
            score += self.joint_weight * float(
                np.log(self.architecture_l.pmf(arch_str)) - np.log(self.architecture_g.pmf(arch_str)))
        return score

    def _parameter_model_for(self, candidate: Candidate, architecture: Architecture, node: str):
        """
        The most specific parameter model available for ``node``: the ``(architecture, node)``
        model, else (with ``component_fallback``) the ``(node, component)`` model, else None.
        """
        model = self.parameter_models.get((architecture, node))
        if model is None and self.component_fallback:
            model = self.component_models.get((node, candidate[node]["name"]))
        return model

    def score_candidates(self, candidates: List[Candidate]) -> np.ndarray:
        """
        Score EA-generated candidates by their good/bad log density ratio (higher = more promising).

        Each score is the structure score (per-node component terms + weighted joint-architecture
        term) plus, for every node, the log-ratio of the most specific available parameter model;
        nodes without one contribute 0. ``fit()`` must have returned True before calling this.
        """
        assert self.architecture_l is not None and self.architecture_g is not None, \
            "fit() must be called (and return True) before score_candidates()."

        scores = []
        for candidate in candidates:
            architecture = self.get_architecture_key(candidate)
            score = self._structure_score(candidate, architecture)

            for node in self.node_names:
                model = self._parameter_model_for(candidate, architecture, node)
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
        return {key: self._describe(model) for key, model in self.parameter_models.items()}

    def component_model_availability(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Report which component-level (fallback) parameter models currently carry evidence.

        Returns ``{(node, component): {"numeric": bool, "categorical": [param names]}}``. Empty when
        ``component_fallback`` is False.
        """
        return {key: self._describe(model) for key, model in self.component_models.items()}

    @staticmethod
    def _describe(model: ParamGroupModel) -> Dict[str, Any]:
        return {
            "numeric": model.multi_l is not None,
            "categorical": list(model.cat_l.keys()),
        }
