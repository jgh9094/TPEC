from typeguard import typechecked
import copy as cp

from Source.Base.archive import Archive, ArchiveEntry
from Source.Base.individual import Individual
from Source.CASH.individual import CASHIndividual


@typechecked
class CASHArchive(Archive):
    """
    Archive of evaluated CASH individuals.

    The genome is the nested pipeline configuration
    ``{node: {"name": component, "params": {...}}, ...}``. Its canonical key is the whole
    configuration serialized deterministically, so two pipelines match iff every node
    picks the same component with the same parameter values.
    """

    def compute_key(self, individual: Individual) -> str:
        """Serialize the full nested pipeline configuration into a canonical key."""
        return self._canonical_key(individual.get_genotype())

    def build_individual(self, entry: ArchiveEntry) -> CASHIndividual:
        """Rebuild a fresh (unevaluated) CASHIndividual from a stored genome."""
        return CASHIndividual(cp.deepcopy(entry.genome))
