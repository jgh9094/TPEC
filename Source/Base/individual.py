from abc import ABC, abstractmethod
from typeguard import typechecked
from typing import Any, Optional, Tuple
import copy as cp

@typechecked
class Individual(ABC):
    """
    Abstract base class for an individual in the search.

    An Individual encapsulates a genotype (e.g., hyperparameter values or an ML
    pipeline configuration) together with additional information such as the
    performance (train/validation/test) and expected improvement (ei).

    Subclasses are responsible for defining how the genotype is represented and
    retrieved by implementing :meth:`get_genotype` and :meth:`__repr__`.
    """
    def __init__(self):
        # train_performance
        self.train_performance = None
        # validation performance
        self.val_performance = None
        # test performance
        self.test_performance = None
        # expected improvement
        self.ei = None
        # how this individual was constructed ("random" or "tpe" offspring construction); set by
        # the EA at creation time and consumed when the individual is recorded in the archive
        self.construction: Optional[str] = None
        # which variation operator produced this offspring ("mutation", "crossover", or
        # "crossover_mutation"); set by the EA at creation time and consumed by the archive.
        # None for the initial population (not produced by a variation operator).
        self.operation: Optional[str] = None
        # archive ids of the two parents that produced this offspring (identical for a
        # mutation-only offspring; two, possibly-equal, ids for a crossover offspring). None for
        # the initial population, which has no parents.
        self.parent_ids: Optional[Tuple[int, int]] = None
        # id assigned to this individual when it is recorded in the provenance archive; lets a
        # later offspring reference this individual as a parent by its archive id.
        self.archive_id: Optional[int] = None
        # whether any CV fold errored while evaluating this individual; set during evaluation
        self.eval_error: Optional[bool] = None
        # genotype for the individual (e.g., hyperparameter values or ML pipeline configuration)
        self.genotype = None
        return

    def get_genotype(self) -> Any:
        """Return the individual's genotype."""
        assert self.genotype is not None, "Genotype has not been set yet."
        return cp.deepcopy(self.genotype)

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

    def get_ei(self) -> float:
        assert self.ei is not None, "Expected Improvement has not been set yet."
        return self.ei

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the individual."""
        raise NotImplementedError