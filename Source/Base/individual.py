from typeguard import typechecked
from typing import Dict, Any
import copy as cp

@typechecked
class Individual:
    """
    This class encapsulates a set of hyperparameters (the "params") and stores additional information,
    such as the performance (accuracy), expected improvement (ei), and cross-validation training score (cv_train_score).
    """
    def __init__(self, params: Dict[str, Any], model_type: str):
        """
        Parameters:
            param_space (ModelParams): This object encapsulates the parameter space and allows you to sample from it.
            performance (float): Objective function value, or the cross-validation score.
            ei (float): Expected improvement score.
            cv_train_score (float): Final accuracy on the training set.
        """

        # train_performance
        self.train_performance = None
        # validation performance
        self.val_performance = None
        # test performance
        self.test_performance = None
        # expected improvement
        self.ei = None
        # upper confidence bound
        self.ucb = None
        # probability of improvement
        self.pi = None

        # Initialize params, a set of random hyperparameters ({parameter_name: value})
        self.params: Dict[str, Any] = params
        self.model_type = model_type
        return

    def __repr__(self):
        return f"Individual(params={self.params}"

    def set_train_performance(self, f: float) -> None:
        assert self.train_performance is None, "Train performance has already been set."
        self.train_performance = f

    def get_train_performance(self) -> float:
        assert self.train_performance is not None, "Train performance has not been set yet."
        return self.train_performance

    def set_val_performance(self, f: float) -> None:
        assert self.val_performance is None, "Validation performance has already been set."
        self.val_performance = f

    def get_val_performance(self) -> float:
        assert self.val_performance is not None, "Validation performance has not been set yet."
        return self.val_performance

    def set_test_performance(self, f: float) -> None:
        assert self.test_performance is None, "Test performance has already been set."
        self.test_performance = f

    def get_test_performance(self) -> float:
        assert self.test_performance is not None, "Test performance has not been set yet."
        return self.test_performance

    def set_ei(self, ei: float) -> None:
        assert self.ei is None, "Expected Improvement has already been set."
        self.ei = ei

    def set_ucb(self, ucb: float) -> None:
        assert self.ucb is None, "Upper Confidence Bound has already been set."
        self.ucb = ucb

    def get_ucb(self) -> float:
        assert self.ucb is not None, "Upper Confidence Bound has not been set yet."
        return self.ucb

    def set_pi(self, pi: float) -> None:
        assert self.pi is None, "Probability of Improvement has already been set."
        self.pi = pi

    def get_pi(self) -> float:
        assert self.pi is not None, "Probability of Improvement has not been set yet."
        return self.pi

    def get_ei(self) -> float:
        assert self.ei is not None, "Expected Improvement has not been set yet."
        return self.ei

    def get_params(self) -> Dict[str, Any]:
        # return a deep copy to avoid accidental modifications
        return cp.deepcopy(self.params)