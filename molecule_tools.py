"""Small RDKit utilities for validating and describing molecules."""

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdFingerprintGenerator


# Morgan fingerprints are useful inputs for many molecular ML models.
FINGERPRINT_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def inspect_molecule(smiles):
    """Return basic properties for a valid SMILES, or None if invalid."""
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    fingerprint = FINGERPRINT_GENERATOR.GetFingerprint(molecule)

    return {
        "smiles": Chem.MolToSmiles(molecule),
        "qed": float(QED.qed(molecule)),
        "molecular_weight": float(Descriptors.MolWt(molecule)),
        "logp": float(Descriptors.MolLogP(molecule)),
        "fingerprint": fingerprint,
    }
