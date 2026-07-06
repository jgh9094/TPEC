from abc import ABC, abstractmethod
import copy
import numpy as np
from typeguard import typechecked
from typing import Tuple, Dict, List, TypedDict, Any, Literal, Union

# Defining custom type alias
# Value must be exactly one of the specified literals
class IntParam(TypedDict):
    bounds: Tuple[int, int]
    type: Literal["int"]

class FloatParam(TypedDict):
    bounds: Tuple[float, float]
    type: Literal["float"]

class CatParam(TypedDict):
    bounds: Tuple[str | None, ...]
    type: Literal["cat"]

class BoolParam(TypedDict):
    bounds: Tuple[bool, bool]
    type: Literal["bool"]

# ParamSpec can be one of IntParam, FloatParam, CatParam, and BoolParam
ParamSpec = Union[IntParam, FloatParam, CatParam, BoolParam]
# Dictionary where each key is a parameter name, and each value is exactly one of the 3 kinds of ParamSpecs
ParamSpace = Dict[str, ParamSpec] # {parameter_name: {"bounds": Tuple, "type": Literal["int", "float", "cat", "bool"]}}

@typechecked
class ModelParams(ABC):
    """
    This class encapsulates the parameter space and provides
    helper methods for mutation and random sampling.
    """
    def __init__(self, param_space: ParamSpace):
        self.param_space = param_space

    def get_parameter_space(self) -> ParamSpace:
        """
        Returns the parameter space.
        """
        assert self.param_space is not None, "Parameter space is not defined."
        return self.param_space

    @abstractmethod
    def generate_random_parameters(self, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Returns a random set of parameters.
        Format {parameter_name: value}
        """
        pass

    def get_param_type(self, key: str) -> str:
        """
        Returns the type of a given parameter.
        """
        assert key in self.param_space, f"Parameter '{key}' not found in parameter space."
        return self.param_space[key]['type']

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

    def pick_categorical_parameter(self, choices: List | Tuple, rng: np.random.Generator):
        """
        Picks a random value from a list of categorical choices.
        Parameters
            choices (List | Tuple): A list or tuple of possible categorical values.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        assert len(choices) > 0, "Choices list cannot be empty."
        return rng.choice(choices)

    @abstractmethod
    def mutate_parameters(self, model_params: Dict[str, Any], var: float, mut_rate: float, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Mutates a given set of hyperparameters in-place.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to mutate.
            var (float): Variation factor for mutation.
            mut_rate (float): Probability of mutating each parameter.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        pass

    @abstractmethod
    def variation_fix_parameters(self, model_params: Dict[str, Any], rng: np.random.Generator) -> None:
        """
        Fixes parameters (in-place) that do not align with scikit-learn's requirements.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to fix.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """
        Returns the model type as a string.
        """
        pass

    @abstractmethod
    def tpe_parameters(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a modified copy of 'model_params'. This ensures parameters are adjusted for
        compatibility with the TPE optimizer.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to adjust.

        Returns:
            Dict[str, Any]: A copy of 'model_params' adjusted for TPE optimization.
        """
        pass

    @abstractmethod
    def eval_parameters(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """ Fixes a set of parameter returned by TPE for hard evaluation. """
        pass

@typechecked
class RandomForestParams(ModelParams):
    def __init__(self, offset: float = 1.0e-4):
        super().__init__(param_space={
            'n_estimators': IntParam(bounds=(100, 1000), type='int'),
            'criterion': CatParam(bounds=('gini', 'entropy', 'log_loss'), type='cat'),
            'max_depth': IntParam(bounds=(1, 30), type='int'),
            'max_features': FloatParam(bounds=(0.0 + offset, 1.0 - offset), type='float'),
            'max_samples': FloatParam(bounds=(0.0 + offset, 1.0 - offset), type='float'),
            'class_weight': CatParam(bounds=('None', 'balanced', 'balanced_subsample'), type='cat')
        })

    def generate_random_parameters(self, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Generates a random set of parameter values based on the defined parameter space.
        Should be ready for hard evaluation.

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
                rand_genotype[param_name] = float(rng.uniform(*spec["bounds"]))
            elif spec["type"] == "cat":
                rand_genotype[param_name] = rng.choice(spec["bounds"])
            else:
                raise ValueError(f"Unsupported parameter type: {spec['type']}")
        # Fix the parameters to ensure they are valid
        self.variation_fix_parameters(rand_genotype, rng)
        return rand_genotype

    def mutate_parameters(self, model_params: Dict[str, Any], var: float, mut_rate: float, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Mutates the model parameters (genotype) in-place with a given mutation rate.
        Should be ready for hard evaluation.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to mutate.
            mut_rate (float): Probability of mutating each parameter.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        # Per-gene mutation
        for name, spec in self.param_space.items():
            # Coin flip to decide whether to mutate each parameter
            if rng.uniform() < mut_rate:
                if spec["type"] == "int":
                    model_params[name] = self.shift_int_parameter(int(model_params[name]), spec['bounds'][0], spec['bounds'][1], var, rng)
                elif spec["type"] == "float":
                    model_params[name] = self.shift_float_parameter(float(model_params[name]), spec['bounds'][0], spec['bounds'][1], var, rng)
                elif spec["type"] == "cat":
                    model_params[name] = self.pick_categorical_parameter(spec['bounds'], rng)
                else:
                    raise ValueError(f"Unsupported parameter type: {spec['type']}")

        # Fix parameters in case of mutation errors
        self.variation_fix_parameters(model_params, rng)
        return model_params

    def variation_fix_parameters(self, model_params: Dict[str, Any], rng: np.random.Generator) -> None:
        """
        Fixes parameters (in-place) that do not align with scikit-learn's requirements.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to fix.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        return

    def tpe_parameters(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a modified copy of 'model_params', which are parameters adjusted for compatibility
        with the TPE optimizer.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to adjust.

        Returns:
            Dict[str, Any]: A copy of 'model_params' adjusted for TPE optimization.
        """
        # If 'bootstrap' is True, 'max_samples' must have a numeric value within bounds
        if model_params['bootstrap'] is True:
            bounds = self.param_space['max_samples']['bounds']
            assert(model_params['max_samples'] is not None
                   and bounds[0] <= model_params['max_samples'] <= bounds[1])

        # If 'bootstrap' is False, 'max_samples' must be None
        if model_params['bootstrap'] is False:
            assert(model_params['max_samples'] is None)

        model_params_copy = copy.deepcopy(model_params)

        if model_params_copy['max_samples'] is None:
            # if bootstrap is False, set max_samples to 1.0 (100% of data being used)
            model_params_copy['max_samples'] = 1.0

        return model_params_copy

    def eval_parameters(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """ Fixes parameters (in-place) that do not align with scikit-learn's requirements. """
        # make sure if 'bootstrap' is True, 'max_samples' must have a numeric value within bounds
        return {'n_estimators': model_params['n_estimators'],
                'criterion': model_params['criterion'],
                'max_depth': model_params['max_depth'],
                'max_features': model_params['max_features'],
                'max_samples': model_params['max_samples'],
                'class_weight': model_params['class_weight'] if model_params['class_weight'] != 'None' else None}

    def get_model_type(self) -> str:
        """
        Returns the model type as a string.
        """
        return "RF"

@typechecked
class KernelSVCParams(ModelParams):
    def __init__(self, offset: float = 1.0e-4):
        super().__init__(param_space = {
            'C': FloatParam(bounds=(0.0 + offset, 1e2), type='float'),
            'kernel': CatParam(bounds=('linear', 'poly', 'rbf', 'sigmoid'), type='cat'),
            'max_iter': IntParam(bounds=(10000, 100000), type='int'),
            'class_weight': CatParam(bounds=('None', 'balanced'), type='cat'),
            'decision_function_shape': CatParam(bounds=('ovo', 'ovr'), type='cat')
        })

    def generate_random_parameters(self, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Generates a random set of parameter values based on the defined parameter space.
        Should be ready for hard evaluation.

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
                rand_genotype[param_name] = float(rng.uniform(*spec["bounds"]))
            elif spec["type"] == "cat":
                rand_genotype[param_name] = rng.choice(spec["bounds"])
            else:
                raise ValueError(f"Unsupported parameter type: {spec['type']}")
        # Fix the parameters to ensure they are valid
        self.variation_fix_parameters(rand_genotype, rng)
        return rand_genotype

    def mutate_parameters(self, model_params: Dict[str, Any], var: float, mut_rate: float, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Mutates the model parameters (genotype) in-place with a given mutation rate.
        Should be ready for hard evaluation.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to mutate.
            mut_rate (float): Probability of mutating each parameter.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        # Per-gene mutation
        for name, spec in self.param_space.items():
            # Coin flip to decide whether to mutate each parameter
            if rng.uniform() < mut_rate:
                if spec["type"] == "int":
                    model_params[name] = self.shift_int_parameter(int(model_params[name]), spec['bounds'][0], spec['bounds'][1], var, rng)
                elif spec["type"] == "float":
                    model_params[name] = self.shift_float_parameter(float(model_params[name]), spec['bounds'][0], spec['bounds'][1], var, rng)
                elif spec["type"] == "cat":
                    model_params[name] = self.pick_categorical_parameter(spec['bounds'], rng)
                else:
                    raise ValueError(f"Unsupported parameter type: {spec['type']}")

        # Fix parameters in case of mutation errors
        self.variation_fix_parameters(model_params, rng)
        return model_params

    def variation_fix_parameters(self, model_params: Dict[str, Any], rng: np.random.Generator) -> None:
        """
        Fixes parameters (in-place) that do not align with scikit-learn's requirements.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to fix.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        return

    def tpe_parameters(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a deep copy of 'model_params', which are parameters adjusted for the TPE optimizer.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to adjust.

        Returns:
            Dict[str, Any]: A copy of 'model_params' adjusted for TPE optimization.
        """
        model_params_copy = copy.deepcopy(model_params)
        return model_params_copy

    def eval_parameters(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """ Fixes parameters (in-place) that do not align with scikit-learn's requirements. """
        return {'C': model_params['C'],
                'kernel': model_params['kernel'],
                'max_iter': model_params['max_iter'],
                'class_weight': model_params['class_weight'] if model_params['class_weight'] != 'None' else None,
                'decision_function_shape': model_params['decision_function_shape']}

    def get_model_type(self) -> str:
        """
        Returns the model type as a string.
        """
        return "KSVC"

@typechecked
class GradientBoostParams(ModelParams):
    def __init__(self, binary_class: bool, offset: float = 1.0e-4):
        super().__init__(param_space = {
            'loss': CatParam(bounds=('log_loss', 'exponential'), type='cat') if binary_class else CatParam(bounds=('log_loss',), type='cat'),
            'learning_rate': FloatParam(bounds=(0.0 + offset, 1.0), type='float'),
            'n_estimators': IntParam(bounds=(100, 1000), type='int'),
            'subsample': FloatParam(bounds=(0.0 + offset, 1.0 - offset), type='float'),
            'criterion': CatParam(bounds=('friedman_mse', 'squared_error'), type='cat'),
            'max_depth': IntParam(bounds=(1, 30), type='int'),
            'max_features': FloatParam(bounds=(0.0 + offset, 1.0 - offset), type='float')
        })

    def generate_random_parameters(self, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Generates a random set of parameter values based on the defined parameter space.
        Should be ready for hard evaluation.

        Parameters:
            rng (np.random.default_rng): A NumPy random generator instance.
        Returns:
            Dict[str, Any]: A dictionary of randomly generated parameters.
        """
        rand_genotype = {}
        for param_name, spec in self.param_space.items():
            if spec["type"] == "int":
                rand_genotype[param_name] = int(rng.integers(*spec["bounds"]))
            elif spec["type"] == "float":
                rand_genotype[param_name] = float(rng.uniform(*spec["bounds"]))
            elif spec["type"] == "cat":
                rand_genotype[param_name] = rng.choice(spec["bounds"])
            else:
                raise ValueError(f"Unsupported parameter type: {spec['type']}")
        # Fix the parameters to ensure they are valid
        self.variation_fix_parameters(rand_genotype, rng)
        return rand_genotype

    def mutate_parameters(self, model_params: Dict[str, Any], var: float, mut_rate: float, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Mutates the model parameters (genotype) in-place with a given mutation rate.
        Should be ready for hard evaluation.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to mutate.
            mut_rate (float): Probability of mutating each parameter.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        # Per-gene mutation
        for name, spec in self.param_space.items():
            # Coin flip to decide whether to mutate each parameter
            if rng.uniform() < mut_rate:
                if spec["type"] == "int":
                    model_params[name] = self.shift_int_parameter(int(model_params[name]), spec['bounds'][0], spec['bounds'][1], var, rng)
                elif spec["type"] == "float":
                    model_params[name] = self.shift_float_parameter(float(model_params[name]), spec['bounds'][0], spec['bounds'][1], var, rng)
                elif spec["type"] == "cat":
                    model_params[name] = self.pick_categorical_parameter(spec['bounds'], rng)
                else:
                    raise ValueError(f"Unsupported parameter type: {spec['type']}")

        # Fix parameters in case of mutation errors
        self.variation_fix_parameters(model_params, rng)
        return model_params

    def variation_fix_parameters(self, model_params: Dict[str, Any], rng: np.random.Generator) -> None:
        """
        Fixes parameters (in-place) that do not align with scikit-learn's requirements.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to fix.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        return

    def tpe_parameters(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a deep copy of 'model_params', which are parameters adjusted for the TPE optimizer.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to adjust.

        Returns:
            Dict[str, Any]: A copy of 'model_params' adjusted for TPE optimization.
        """

        model_params_copy = copy.deepcopy(model_params)
        return model_params_copy

    def eval_parameters(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """ Fixes parameters (in-place) that do not align with scikit-learn's requirements. """
        return {'n_estimators': model_params['n_estimators'],
                'learning_rate': model_params['learning_rate'],
                'subsample': model_params['subsample'],
                'criterion': model_params['criterion'],
                'max_depth': model_params['max_depth'],
                'max_features': model_params['max_features'],
                'loss': model_params['loss']}

    def get_model_type(self) -> str:
        """
        Returns the model type as a string.
        """
        return "GB"

@typechecked
class KNeighborsClassifierParams(ModelParams):
    def __init__(self, offset: float = 1.0e-4):
        super().__init__(param_space = {
            'n_neighbors': IntParam(bounds=(1, 100), type='int'),
            'weights': CatParam(bounds=('uniform', 'distance'), type='cat'),
            'algorithm': CatParam(bounds=('ball_tree', 'kd_tree', 'brute'), type='cat'),
            'leaf_size': IntParam(bounds=(1, 50), type='int'),
            'p': IntParam(bounds=(1, 5), type='int')
        })

    def generate_random_parameters(self, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Generates a random set of parameter values based on the defined parameter space.
        Should be ready for hard evaluation.

        Parameters:
            rng (np.random.default_rng): A NumPy random generator instance.
        Returns:
            Dict[str, Any]: A dictionary of randomly generated parameters.
        """
        rand_genotype = {}
        for param_name, spec in self.param_space.items():
            if spec["type"] == "int":
                rand_genotype[param_name] = int(rng.integers(*spec["bounds"]))
            elif spec["type"] == "cat":
                rand_genotype[param_name] = rng.choice(spec["bounds"])
            else:
                raise ValueError(f"Unsupported parameter type: {spec['type']}")
        # Fix the parameters to ensure they are valid
        self.variation_fix_parameters(rand_genotype, rng)
        return rand_genotype

    def mutate_parameters(self, model_params: Dict[str, Any], var: float, mut_rate: float, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Mutates the model parameters (genotype) in-place with a given mutation rate.
        Should be ready for hard evaluation.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to mutate.
            mut_rate (float): Probability of mutating each parameter.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        # Per-gene mutation
        for name, spec in self.param_space.items():
            # Coin flip to decide whether to mutate each parameter
            if rng.uniform() < mut_rate:
                if spec["type"] == "int":
                    model_params[name] = self.shift_int_parameter(int(model_params[name]), spec['bounds'][0], spec['bounds'][1], var, rng)
                elif spec["type"] == "cat":
                    model_params[name] = self.pick_categorical_parameter(spec['bounds'], rng)
                else:
                    raise ValueError(f"Unsupported parameter type: {spec['type']}")

        # Fix parameters in case of mutation errors
        self.variation_fix_parameters(model_params, rng)
        return model_params

    def variation_fix_parameters(self, model_params: Dict[str, Any], rng: np.random.Generator) -> None:
        """
        Fixes parameters (in-place) that do not align with scikit-learn's requirements.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to fix.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        return

    def tpe_parameters(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a deep copy of 'model_params', which are parameters adjusted for the TPE optimizer.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to adjust.

        Returns:
            Dict[str, Any]: A copy of 'model_params' adjusted for TPE optimization.
        """

        model_params_copy = copy.deepcopy(model_params)
        return model_params_copy

    def eval_parameters(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """ Fixes parameters (in-place) that do not align with scikit-learn's requirements. """
        return {'n_neighbors': model_params['n_neighbors'],
                'weights': model_params['weights'],
                'algorithm': model_params['algorithm'],
                'leaf_size': model_params['leaf_size'],
                'p': model_params['p']}

    def get_model_type(self) -> str:
        """
        Returns the model type as a string.
        """
        return "KNN"

@typechecked
class MLPClassifierParams(ModelParams):
    def __init__(self, offset: float = 1.0e-4):
        super().__init__(param_space = {
            'layer_1': IntParam(bounds=(10, 100), type='int'),
            'layer_2': IntParam(bounds=(10, 100), type='int'),
            'layer_3': IntParam(bounds=(10, 100), type='int'),
            'layer_4': IntParam(bounds=(10, 100), type='int'),
            'layer_5': IntParam(bounds=(10, 100), type='int'),
            'activation': CatParam(bounds=('identity', 'logistic', 'tanh', 'relu'), type='cat'),
            'solver': CatParam(bounds=('lbfgs', 'sgd', 'adam'), type='cat'),
            'max_iter': IntParam(bounds=(10000, 100000), type='int')
        })

    def generate_random_parameters(self, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Generates a random set of parameter values based on the defined parameter space.
        Should be ready for hard evaluation.

        Parameters:
            rng (np.random.default_rng): A NumPy random generator instance.
        Returns:
            Dict[str, Any]: A dictionary of randomly generated parameters.
        """
        rand_genotype = {}
        for param_name, spec in self.param_space.items():
            if spec["type"] == "int":
                rand_genotype[param_name] = int(rng.integers(*spec["bounds"]))
            elif spec["type"] == "cat":
                rand_genotype[param_name] = rng.choice(spec["bounds"])
            else:
                raise ValueError(f"Unsupported parameter type: {spec['type']}")
        # Fix the parameters to ensure they are valid
        self.variation_fix_parameters(rand_genotype, rng)
        return rand_genotype

    def mutate_parameters(self, model_params: Dict[str, Any], var: float, mut_rate: float, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Mutates the model parameters (genotype) in-place with a given mutation rate.
        Should be ready for hard evaluation.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to mutate.
            mut_rate (float): Probability of mutating each parameter.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        # Per-gene mutation
        for name, spec in self.param_space.items():
            # Coin flip to decide whether to mutate each parameter
            if rng.uniform() < mut_rate:
                if spec["type"] == "int":
                    model_params[name] = self.shift_int_parameter(int(model_params[name]), spec['bounds'][0], spec['bounds'][1], var, rng)
                elif spec["type"] == "cat":
                    model_params[name] = self.pick_categorical_parameter(spec['bounds'], rng)
                else:
                    raise ValueError(f"Unsupported parameter type: {spec['type']}")

        # Fix parameters in case of mutation errors
        self.variation_fix_parameters(model_params, rng)
        return model_params

    def variation_fix_parameters(self, model_params: Dict[str, Any], rng: np.random.Generator) -> None:
        """
        Fixes parameters (in-place) that do not align with scikit-learn's requirements.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to fix.
            rng (np.random.Generator): A NumPy random generator instance.
        """
        return

    def tpe_parameters(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a deep copy of 'model_params', which are parameters adjusted for the TPE optimizer.

        Parameters:
            model_params (Dict[str, Any]): The set of hyperparameters to adjust.

        Returns:
            Dict[str, Any]: A copy of 'model_params' adjusted for TPE optimization.
        """

        model_params_copy = copy.deepcopy(model_params)
        return model_params_copy

    def eval_parameters(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """ Fixes parameters (in-place) that do not align with scikit-learn's requirements. """
        return {'layer_1': model_params['layer_1'],
                'layer_2': model_params['layer_2'],
                'layer_3': model_params['layer_3'],
                'layer_4': model_params['layer_4'],
                'layer_5': model_params['layer_5'],
                'activation': model_params['activation'],
                'solver': model_params['solver'],
                'max_iter': model_params['max_iter']}

    def get_model_type(self) -> str:
        """
        Returns the model type as a string.
        """
        return "MLP"