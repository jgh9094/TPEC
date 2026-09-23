from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
import numpy as np
from typeguard import typechecked
from typing import Tuple, Dict, List, TypedDict, Any, Literal, Union, Optional

# Defining custom type alias
# Value must be exactly one of the specified literals
class IntParam(TypedDict):
    bounds: Tuple[int, int]
    type: Literal["int"]

class FloatParam(TypedDict):
    bounds: Tuple[float, float]
    type: Literal["float"]
    log: bool  # if True, sample/optimize on a log scale (values must be strictly positive)

class CatParam(TypedDict):
    bounds: Tuple[str | None, ...]
    type: Literal["cat"]

class BoolParam(TypedDict):
    bounds: Tuple[bool, bool]
    type: Literal["bool"]

@dataclass(frozen=True)
class DataContext:
    """
    Dataset-derived context that a parameter space needs at construction time to shape its
    search space (as opposed to the ``random_state`` used only at evaluation time).

    Passed uniformly to every ModelParams subclass ``__init__`` so a pipeline/operator factory
    can build any operator as ``cls(ctx)`` without special-casing which ones need context.
    Operators that need none of these fields simply ignore the context.

      * ``n_samples``  -- rows in the training data (bounds n_quantiles, Nystroem components).
      * ``n_features`` -- feature count of the CURRENT representation reaching the operator
                          (bounds n_components / n_clusters for decomposition/agglomeration).
      * ``n_classes``  -- number of target classes (selects the legal ``loss`` set for
                          GradientBoosting: 'exponential' is binary-only).
    """
    n_samples: int
    n_features: int
    n_classes: int

# ParamSpec can be one of IntParam, FloatParam, CatParam, and BoolParam
ParamSpec = Union[IntParam, FloatParam, CatParam, BoolParam]
# Dictionary where each key is a parameter name, and each value is exactly one of the 3 kinds of ParamSpecs
ParamSpace = Dict[str, ParamSpec] # {parameter_name: {"bounds": Tuple, "type": Literal["int", "float", "cat", "bool"]}}

@typechecked
class ModelParams(ABC):
    """
    Abstract base class for a model's hyperparameter space.

    Encapsulates the parameter space and provides shared helper methods for random
    sampling and mutation. Concrete classifier parameter spaces (e.g., RandomForest,
    MLP) live in ``Source.ML.classifiers`` and implement the abstract methods below.
    """
    def __init__(self, param_space: ParamSpace):
        self.param_space = param_space

    def generate_random_parameters(self, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Generates a random set of parameter values based on the defined parameter space.
        Each parameter is sampled independently according to its type. Should be ready for
        hard evaluation. Format {parameter_name: value}.

        Parameters:
            rng (np.random.Generator): A NumPy random generator instance.
        Returns:
            Dict[str, Any]: A dictionary of randomly generated parameters.
        """
        rand_genotype = {}
        for param_name, spec in self.param_space.items():
            if spec["type"] == "int":
                rand_genotype[param_name] = int(rng.integers(*spec["bounds"]))
            elif spec["type"] == "float":
                rand_genotype[param_name] = self.sample_float_parameter(spec, rng)
            elif spec["type"] == "cat":
                rand_genotype[param_name] = rng.choice(spec["bounds"])
            else:
                raise ValueError(f"Unsupported parameter type: {spec['type']}")
        return rand_genotype

    def get_params_by_type(self, type: str) -> Dict:
        """
        Retrieves a subset of parameters of a given type.
        Parameters
            type (str): The type of parameters to retrieve ('int', 'float', 'cat', 'bool').
        Returns:
            Dict: A dictionary of parameters matching the specified type.
        """
        assert self.param_space is not None, "Parameter space is not defined."
        assert type in ['int', 'float', 'cat', 'bool'], f"Unsupported parameter type: {type}"
        return {name: info for name, info in self.param_space.items() if info['type'] == type}

    def shift_float_parameter(self, cur_value: float, min: float, max: float, var: float, rng: np.random.Generator) -> float:
        """
        Shifts a float parameter either up or down within bounds.
        68% of increases/decreases will be within var% of the current value
        95% of increases/decreases will be within 2*var% of the current value
        99.7% of increases/decreases will be within 3*var% of the current value
        """
        value = float(cur_value * rng.normal(1.0, var))

        # ensure the value is within the bounds, clip to safe boundaries
        eps = 1e-12
        return np.clip(value, min + eps, max - eps)

    def shift_int_parameter(self, cur_value: int, min: int, max: int, var: float, rng: np.random.Generator) -> int:
        """
        Shifts a integer parameter either up or down within bounds.
        68% of increases/decreases will be within var% of the current value
        95% of increases/decreases will be within 2*var% of the current value
        99.7% of increases/decreases will be within 3*var% of the current value
        """
        value = int(cur_value * rng.normal(1.0, var))

        # ensure the value is within the bounds
        if value < min:
            return min
        elif value > max:
            return max
        else:
            return value

    def sample_float_parameter(self, spec: FloatParam, rng: np.random.Generator) -> float:
        """
        Draws a random float within the spec's bounds.

        When the spec is flagged ``log=True`` the value is sampled uniformly in log-space
        (so small and large magnitudes are equally likely); otherwise it is sampled
        uniformly on the linear scale. Log-scaled bounds must be strictly positive.
        """
        lo, hi = spec["bounds"]
        if spec.get("log", False):
            assert lo > 0.0 and hi > 0.0, "Log-scaled float bounds must be strictly positive."
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return float(rng.uniform(lo, hi))

    def pick_categorical_parameter(self, choices: List | Tuple, rng: np.random.Generator):
        """
        Picks a random value from a list of categorical choices.
        Parameters
            choices (List | Tuple): A list or tuple of possible categorical values.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        assert len(choices) > 0, "Choices list cannot be empty."
        return rng.choice(choices)

    def mutate_parameters_shift(self, model_params: Dict[str, Any], var: float, mut_rate: float, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Mutates a given set of hyperparameters in-place by shifting each selected gene
        away from its current value. Each gene is independently mutated with probability
        ``mut_rate``. Should be ready for hard evaluation.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to mutate.
            var (float): Variation factor for the shift.
            mut_rate (float): Probability of mutating each parameter.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        for name, spec in self.param_space.items():
            if rng.uniform() < mut_rate:
                if spec["type"] == "int":
                    model_params[name] = self.shift_int_parameter(int(model_params[name]), spec['bounds'][0], spec['bounds'][1], var, rng)
                elif spec["type"] == "float":
                    model_params[name] = self.shift_float_parameter(float(model_params[name]), spec['bounds'][0], spec['bounds'][1], var, rng)
                elif spec["type"] == "cat":
                    model_params[name] = self.pick_categorical_parameter(spec['bounds'], rng)
                else:
                    raise ValueError(f"Unsupported parameter type: {spec['type']}")
        return model_params

    def mutate_parameters_random(self, model_params: Dict[str, Any], mut_rate: float, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Mutates a given set of hyperparameters in-place by resampling each selected gene
        uniformly from its full range of available options (rather than shifting from its
        current value). Each gene is independently resampled with probability ``mut_rate``.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to mutate.
            mut_rate (float): Probability of resampling each parameter.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        for name, spec in self.param_space.items():
            if rng.uniform() < mut_rate:
                if spec["type"] == "int":
                    model_params[name] = int(rng.integers(*spec["bounds"]))
                elif spec["type"] == "float":
                    model_params[name] = self.sample_float_parameter(spec, rng)
                elif spec["type"] == "cat":
                    model_params[name] = self.pick_categorical_parameter(spec['bounds'], rng)
                else:
                    raise ValueError(f"Unsupported parameter type: {spec['type']}")
        return model_params

    def tpe_parameters(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a copy of 'model_params' adjusted for compatibility with the TPE optimizer.

        The default is an identity deep copy (parameters are already TPE-compatible). Subclasses
        may override this to apply model-specific encoding for the TPE optimizer.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to adjust.

        Returns:
            Dict[str, Any]: A copy of 'model_params' adjusted for TPE optimization.
        """
        return copy.deepcopy(model_params)

    @abstractmethod
    def get_model_type(self) -> str:
        """
        Returns the model type as a string.
        """
        pass

    @abstractmethod
    def eval_parameters(self, model_params: Dict[str, Any], random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Fixes a set of parameters for hard evaluation with scikit-learn.

        ``random_state`` is an evaluation-time-only input (it does not shape the search space):
        subclasses whose estimator accepts a ``random_state`` fold it into the returned kwargs
        so the caller can build the estimator generically as ``Estimator(**eval_parameters(...))``;
        subclasses whose estimator is deterministic accept and ignore it.
        """
        pass
