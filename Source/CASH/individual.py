from typeguard import typechecked
from typing import Dict, Any
import copy as cp

from Source.Base.individual import Individual


@typechecked
class CASHIndividual(Individual):
    """
    Individual for the CASH (Combined Algorithm Selection and Hyperparameter
    optimization) problem.

    The genotype is a nested pipeline configuration -- one entry per evolved
    pipeline node -- stored as::

        {node_name: {"name": component_name, "params": {param_name: value, ...}}, ...}

    This is exactly the ``Candidate`` representation consumed by ``Source/CASH/tpe.py``
    (``get_architecture_key`` reads ``genotype[node]["name"]`` and the parameter
    models read ``genotype[node]["params"]``). A parameter-free component -- including
    the ``"passthrough"`` choice that means the node/step is skipped -- carries an
    empty ``params`` dict.
    """
    def __init__(self, pipeline: Dict[str, Dict[str, Any]]):
        """
        Parameters:
            pipeline (Dict[str, Dict[str, Any]]): The nested pipeline configuration,
                mapping each node to a ``{"name": component, "params": {...}}`` entry.
        """
        super().__init__()

        self._validate_pipeline(pipeline)

        # the genotype is the nested pipeline configuration for this individual
        self.genotype: Dict[str, Dict[str, Any]] = pipeline
        return

    @staticmethod
    def _validate_pipeline(pipeline: Dict[str, Dict[str, Any]]) -> None:
        """
        Guard the nested pipeline structure so malformed genotypes fail loudly here
        rather than deep inside CASH_TPE.fit()/score_candidates().

        Every node entry must be a mapping with a string ``"name"`` and a dict
        ``"params"`` whose keys are all strings.
        """
        assert len(pipeline) > 0, "Pipeline must contain at least one node."
        for node, entry in pipeline.items():
            assert isinstance(node, str), f"Node key must be a string, got {type(node).__name__}."
            assert isinstance(entry, dict), f"Node '{node}' entry must be a dict, got {type(entry).__name__}."
            assert set(entry.keys()) == {"name", "params"}, \
                f"Node '{node}' entry must have exactly keys {{'name', 'params'}}, got {sorted(entry.keys())}."
            assert isinstance(entry["name"], str), \
                f"Node '{node}' component name must be a string, got {type(entry['name']).__name__}."
            assert isinstance(entry["params"], dict), \
                f"Node '{node}' params must be a dict, got {type(entry['params']).__name__}."
            for param_name in entry["params"]:
                assert isinstance(param_name, str), \
                    f"Node '{node}' param names must be strings, got {type(param_name).__name__}."

    def get_architecture(self) -> Dict[str, str]:
        """Return the node -> component-name mapping (the structural choice, no params)."""
        assert self.genotype is not None, "Genotype has not been set yet."
        return {node: entry["name"] for node, entry in self.genotype.items()}

    def __repr__(self) -> str:
        components = ", ".join(f"{node}={entry['name']}" for node, entry in self.genotype.items())
        return f"CASHIndividual({components})"
