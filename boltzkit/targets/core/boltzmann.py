# minimal working implementation of boltzmann targets (energy evaluation, forces, etc)
# no adaptation to pytorch or numpy yet
# The content of this package is meant to be standalone and for general purpose use (could be copied and used somewhere else)


def create_molecular_boltzmann(
    *,
    name: str | None = None,
    smiles: str | None = None,
    pdb_file: str | None = None,
    force_field: str | None = None,
    temperature: float = 300.0,
):
    """
    Different options to configure a molecular Boltzmann target. Use implementations from subpackge `.core.boltzmann`.
    Determine data-path for huggingface if possible and provide it.

    Returns:
        - openmm system
        - mdtraj?
        - z-matrix?
        - data-path for huggingface if existing
    """
