from typeguard import typechecked
import copy as cp

from Source.Base.archive import Archive, ArchiveEntry
from Source.Base.individual import Individual
from Source.HPO.individual import HPOIndividual


@typechecked
class HPOArchive(Archive):
    """
    Archive of evaluated HPO individuals.

    An HPO run optimizes a single model type, so every archived individual shares that
    ``model_type`` (supplied once at construction and used to rebuild individuals). The
    genome is the ``{parameter_name: value}`` hyperparameter mapping; its canonical key is
    the model type plus that mapping serialized deterministically, so two individuals match
    iff they carry identical hyperparameters.
    """

    def __init__(self, model_type: str) -> None:
        """
        Args:
            model_type (str): The model type these individuals optimize (e.g. 'rf', 'mlp'),
                used to rebuild individuals via :meth:`build_individual`.
        """
        super().__init__()
        self.model_type = model_type
        return

    def compute_key(self, individual: Individual) -> str:
        """Serialize the model type and hyperparameter mapping into a canonical key."""
        return self._canonical_key({
            "model_type": individual.model_type,
            "params": individual.get_genotype(),
        })

    def build_individual(self, entry: ArchiveEntry) -> HPOIndividual:
        """Rebuild a fresh (unevaluated) HPOIndividual from a stored genome."""
        return HPOIndividual(cp.deepcopy(entry.genome), self.model_type)
