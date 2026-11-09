"""Small RDKit utilities for validating and describing molecules."""

from rdkit import Chem, DataStructs
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
        "hbd": int(Descriptors.NumHDonors(molecule)),
        "hba": int(Descriptors.NumHAcceptors(molecule)),
        "rotatable_bonds": int(Descriptors.NumRotatableBonds(molecule)),
        "tpsa": float(Descriptors.TPSA(molecule)),
        "heavy_atoms": int(Descriptors.HeavyAtomCount(molecule)),
        "fingerprint": fingerprint,
    }


def check_filters(properties):
    """Return whether a molecule meets the project's hard limits."""
    if properties["molecular_weight"] > 500:
        return False, "MW exceeds 500"
    if properties["qed"] < 0.4:
        return False, "QED below 0.4"
    if properties["logp"] > 5:
        return False, "LogP exceeds 5"
    if properties["hbd"] > 5:
        return False, "Too many H-bond donors"
    if properties["hba"] > 10:
        return False, "Too many H-bond acceptors"
    if properties["rotatable_bonds"] > 10:
        return False, "Too many rotatable bonds"
    if properties["tpsa"] > 140:
        return False, "TPSA exceeds 140"
    if properties["heavy_atoms"] > 40:
        return False, "Too many heavy atoms"
    return True, "Passed"


def tanimoto_similarity(fingerprint, other_fingerprint):
    """Return Morgan-fingerprint Tanimoto similarity for two molecules."""
    return float(DataStructs.TanimotoSimilarity(fingerprint, other_fingerprint))
