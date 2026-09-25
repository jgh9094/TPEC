##########################################################################################
#
# Abstract base class for an archive of evaluated individuals.
#
# An Archive is a log of an individual an EA evaluates.
# Each individual is represented by a stored entry that records:
# - the genome
# - the performances
# - the generation it was produced,
# - whether it came from random or TPE-guided offspring construction
# - the expected-improvement value that construction assigned
#
# The domain-specific pieces -- how a genome collapses to a lookup key and how a stored
# genome is rebuilt into a concrete Individual -- are left abstract and implemented by the
# CASH and HPO archives (Source/CASH/archive.py, Source/HPO/archive.py).
#
##########################################################################################

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typeguard import typechecked
from typing import Any, Dict, List, Optional, Tuple
import copy as cp
import math
import numpy as np


def _json_safe_float(value: Optional[float]) -> Optional[float]:
    """Collapse non-finite performances (e.g. the -inf regression error penalty) to None so
    that json.dump emits valid JSON instead of the non-standard ``-Infinity`` token."""
    if value is None or not math.isfinite(value):
        return None
    return value
import json

from Source.Base.individual import Individual


# offspring-construction provenance tags stored on every archive entry
RANDOM_CONSTRUCTION = "random"
TPE_CONSTRUCTION = "tpe"
VALID_CONSTRUCTIONS = frozenset({RANDOM_CONSTRUCTION, TPE_CONSTRUCTION})

# variation-operator provenance tags: how an offspring was generated:
# -``MUTATION_OPERATION``: is a single parent mutated
# -``CROSSOVER_OPERATION``: is two parents recombined with no subsequent mutation
# -``CROSSOVER_MUTATION_OPERATION``: is a recombination that was then mutated.
# The initial population is not produced by a variation operator, so its entries
# carry ``operation=None`` (and ``parent_ids=None``).
MUTATION_OPERATION = "mutation"
CROSSOVER_OPERATION = "crossover"
CROSSOVER_MUTATION_OPERATION = "crossover_mutation"
VALID_OPERATIONS = frozenset({
    MUTATION_OPERATION, CROSSOVER_OPERATION, CROSSOVER_MUTATION_OPERATION,
})


@dataclass(frozen=True)
class ArchiveEntry:
    """
    One immutable record in an :class:`Archive`.

    Attributes:
        id (int): Unique, archive-assigned identifier (monotonic insertion order, starting
            at 0). Doubles as the entry's position in the archive.
        key (str): Canonical lookup key for the genome (see :meth:`Archive.compute_key`).
            Two entries with identical genomes share a key, which is how the archive answers
            "have we evaluated this pipeline before?".
        generation (int): Generation index that produced the individual (e.g. -1 for the
            initial population, 0..gens-1 for evolved offspring).
        construction (str): How the individual was constructed -- ``RANDOM_CONSTRUCTION`` or
            ``TPE_CONSTRUCTION``.
        operation (Optional[str]): Which variation operator produced the offspring --
            ``MUTATION_OPERATION``, ``CROSSOVER_OPERATION``, or ``CROSSOVER_MUTATION_OPERATION``.
            ``None`` for the initial population (not produced by a variation operator).
        parent_ids (Optional[Tuple[int, int]]): Archive ids of the two parents that produced the
            offspring. Both ids are identical for a mutation-only offspring (single parent); for a
            crossover offspring they are the two parents' ids (which may coincide if one individual
            was selected as both parents). ``None`` for the initial population.
        ei (float): Expected-improvement value assigned at construction. ``-inf`` for random
            offspring, which carry no surrogate score.
        genome (Any): Deep-copied genotype of the individual.
        train_performance (Optional[float]): Mean CV train score, or None if the evaluation
            errored.
        val_performance (Optional[float]): Mean CV validation score, or None if the
            evaluation errored.
        error (bool): True if any fold errored during evaluation, else False.
    """
    id: int
    key: str
    generation: int
    construction: str
    operation: Optional[str]
    parent_ids: Optional[Tuple[int, int]]
    ei: float
    genome: Any
    train_performance: Optional[float]
    val_performance: Optional[float]
    error: bool

    def to_record(self) -> Dict[str, Any]:
        """Return a plain-dict view of this entry (for logging / serialization)."""
        return {
            "id": self.id,
            "key": self.key,
            "generation": self.generation,
            "construction": self.construction,
            "operation": self.operation,
            "parent_ids": list(self.parent_ids) if self.parent_ids is not None else None,
            "ei": self.ei,
            "genome": cp.deepcopy(self.genome),
            "train_performance": _json_safe_float(self.train_performance),
            "val_performance": _json_safe_float(self.val_performance),
            "error": self.error,
        }


@typechecked
class Archive(ABC):
    """
    Abstract archive of evaluated individuals.

    Concrete subclasses (one per search domain) implement the two domain-specific hooks:
      * :meth:`compute_key` -- collapse an individual's genome to a canonical lookup key.
      * :meth:`build_individual` -- rebuild a concrete Individual from a stored entry.

    Everything else -- id assignment, keyed storage, duplicate detection, and best-so-far
    lookup -- is provided here and shared across domains.
    """

    def __init__(self) -> None:
        # every entry, in insertion order (index == ArchiveEntry.id)
        self._entries: List[ArchiveEntry] = []
        # key -> ids of the entries carrying that key (multiple, since duplicates are kept)
        self._by_key: Dict[str, List[int]] = {}
        # next id to assign (monotonic)
        self._next_id: int = 0
        return

    # ------------------------------------------------------------------ #
    # domain-specific hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def compute_key(self, individual: Individual) -> str:
        """
        Return a canonical, hashable lookup key for ``individual``'s genome.

        Two individuals with identical genomes must produce the same key, and two with
        differing genomes must (barring hash collisions) produce different keys. This is
        used for O(1) "have we seen this pipeline before?" checks via :meth:`contains`.

        Note: keys compare genomes exactly. Two pipelines that differ only by a
        floating-point epsilon in one parameter are treated as distinct, since their stored
        values differ; the archive does not fuzz-match near-duplicates.
        """
        raise NotImplementedError

    @abstractmethod
    def build_individual(self, entry: ArchiveEntry) -> Individual:
        """
        Rebuild a concrete Individual from a stored entry's genome.

        The returned individual carries only the genome (a fresh, unevaluated individual);
        performances are intentionally left unset so callers can re-evaluate if needed.
        """
        raise NotImplementedError

    @staticmethod
    def _canonical_key(payload: Any) -> str:
        """
        Serialize ``payload`` to a deterministic string usable as a dictionary key.

        Keys are sorted recursively so dict ordering never affects the result, and any value
        that is not natively JSON-serializable (e.g. a numpy scalar) is coerced via ``str``.
        Subclasses assemble a plain nested structure from a genome and pass it here.
        """
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    # ------------------------------------------------------------------ #
    # insertion
    # ------------------------------------------------------------------ #
    def add(self,
            individual: Individual,
            generation: int,
            construction: str,
            error: bool,
            ei: Optional[float] = None,
            operation: Optional[str] = None,
            parent_ids: Optional[Tuple[int, int]] = None) -> ArchiveEntry:
        """
        Record ``individual`` as a new archive entry and return it.

        Args:
            individual (Individual): The evaluated individual to record. Its genome is
                deep-copied; its ``train_performance``/``val_performance`` are read directly.
            generation (int): Generation that produced the individual.
            construction (str): ``RANDOM_CONSTRUCTION`` or ``TPE_CONSTRUCTION``.
            error (bool): True if any fold errored during evaluation. When True, missing
                train/validation performances are tolerated and stored as None.
            ei (Optional[float]): Expected improvement assigned at construction. Ignored for
                random construction (forced to ``-inf``). For TPE construction it must be
                supplied here or already set on the individual (``individual.ei``).
            operation (Optional[str]): Variation operator that produced the offspring
                (``MUTATION_OPERATION`` / ``CROSSOVER_OPERATION`` /
                ``CROSSOVER_MUTATION_OPERATION``), or None for the initial population.
            parent_ids (Optional[Tuple[int, int]]): Archive ids of the two parents. Must be
                supplied together with ``operation`` (both None for the initial population). For
                a mutation operation the two ids must be identical (single parent).

        Returns:
            ArchiveEntry: The newly created (and now stored) entry.
        """
        assert construction in VALID_CONSTRUCTIONS, \
            f"construction must be one of {sorted(VALID_CONSTRUCTIONS)}, got {construction!r}."

        # lineage provenance: operation and parent ids travel together (both set for an
        # offspring, both None for the initial population), and a mutation has a single parent.
        assert (operation is None) == (parent_ids is None), \
            "operation and parent_ids must be supplied together (both None for the initial population)."
        if operation is not None:
            assert operation in VALID_OPERATIONS, \
                f"operation must be one of {sorted(VALID_OPERATIONS)}, got {operation!r}."
            assert parent_ids is not None and len(parent_ids) == 2 \
                and all(isinstance(pid, int) for pid in parent_ids), \
                "parent_ids must be a tuple of exactly two integer archive ids."
            if operation == MUTATION_OPERATION:
                assert parent_ids[0] == parent_ids[1], \
                    "A mutation-only offspring has a single parent, so its two parent ids must be equal."

        # resolve expected improvement per provenance: random offspring carry no surrogate
        # score (-inf); TPE offspring must have one, from the arg or the individual itself.
        if construction == RANDOM_CONSTRUCTION:
            assert ei is None or ei == float("-inf"), \
                "Random offspring must carry ei == -inf (or leave ei unset)."
            ei_value = float("-inf")
        else:
            if ei is None:
                ei = individual.ei
            assert ei is not None, \
                "TPE offspring must have an expected-improvement value (pass ei= or set individual.ei)."
            ei_value = float(ei)

        # performances are present unless the evaluation errored on one or more folds
        train = individual.train_performance
        val = individual.val_performance
        if error:
            train = None if train is None else float(train)
            val = None if val is None else float(val)
        else:
            assert train is not None and val is not None, \
                "A non-errored individual must have train and validation performance set."
            train = float(train)
            val = float(val)

        key = self.compute_key(individual)
        entry = ArchiveEntry(
            id=self._next_id,
            key=key,
            generation=generation,
            construction=construction,
            operation=operation,
            parent_ids=parent_ids,
            ei=ei_value,
            genome=cp.deepcopy(individual.get_genotype()),
            train_performance=train,
            val_performance=val,
            error=error,
        )

        self._entries.append(entry)
        self._by_key.setdefault(key, []).append(entry.id)
        self._next_id += 1
        return entry

    # ------------------------------------------------------------------ #
    # lookup
    # ------------------------------------------------------------------ #
    def contains(self, individual: Individual) -> bool:
        """Return True if a genome-identical individual has already been archived."""
        return self.compute_key(individual) in self._by_key

    def entries_for(self, individual: Individual) -> List[ArchiveEntry]:
        """Return all archived entries whose genome matches ``individual`` (may be empty)."""
        return self.get_by_key(self.compute_key(individual))

    def get_by_key(self, key: str) -> List[ArchiveEntry]:
        """Return all entries stored under ``key`` (empty list if none)."""
        return [self._entries[i] for i in self._by_key.get(key, [])]

    def get_by_id(self, id: int) -> ArchiveEntry:
        """Return the entry with the given id."""
        assert 0 <= id < len(self._entries), f"No archive entry with id {id}."
        return self._entries[id]

    def best(self, rng: np.random.Generator) -> Optional[ArchiveEntry]:
        """
        Return a best-scoring entry: the one with the highest validation performance
        (errored/unscored entries excluded), or None if nothing scoreable has been archived.

        When several entries tie for the highest validation performance, one of them is chosen
        uniformly at random using ``rng`` so that no entry is systematically favored (e.g. the
        earliest inserted). Pass the EA's generator to keep the choice reproducible.
        """
        scored = [(e, e.val_performance) for e in self._entries
                  if not e.error and e.val_performance is not None]
        if not scored:
            return None
        best_val = max(val for _, val in scored)
        tied = [e for e, val in scored if val == best_val]
        return tied[int(rng.integers(len(tied)))]

    def to_records(self) -> List[Dict[str, Any]]:
        """Return every entry as a plain dict, in insertion order (for logging / saving)."""
        return [entry.to_record() for entry in self._entries]

    # ------------------------------------------------------------------ #
    # container protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> ArchiveEntry:
        return self._entries[index]

    def __iter__(self):
        return iter(self._entries)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={len(self._entries)}, unique_keys={len(self._by_key)})"
