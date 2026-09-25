from typeguard import typechecked
from typing import Dict, Any
import copy as cp

from Source.Base.individual import Individual

@typechecked
class HPOIndividual(Individual):
    """
    Individual for hyperparameter optimization (HPO).

    The genotype is a set of hyperparameters (the "params") for a single model
    type, stored as a ``{parameter_name: value}`` mapping.
    """
    def __init__(self, params: Dict[str, Any], model_type: str):
        """
        Parameters:
            params (Dict[str, Any]): The hyperparameters ({parameter_name: value}) for this individual.
            model_type (str): The model type these hyperparameters belong to (e.g., 'rf', 'mlp').
        """
        super().__init__()

        # the genotype is the set of hyperparameters for this individual
        self.genotype: Dict[str, Any] = params
        self.model_type = model_type
        return

    def __repr__(self) -> str:
        return f"HPOIndividual(model_type={self.model_type}, params={self.genotype})"