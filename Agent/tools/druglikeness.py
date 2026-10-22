from rdkit import Chem
from rdkit.Chem import Descriptors
from langchain_core.tools import tool

@tool
def calculate_druglikeness(smiles: str) -> dict:
    """
    Calculate molecular properties for a molecule from its SMILES string.
    Returns molecular weight, LogP, TPSA, H-bond donors, and H-bond acceptors.
    """

    # Convert the SMILES string into an RDKit molecule
    mol = Chem.MolFromSmiles(smiles)

    # Check whether the SMILES was valid
    if mol is None:
        return {"error": "Invalid SMILES"}

    #calculate molecular properties
    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "h_bond_donors": Descriptors.NumHDonors(mol),
        "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
    }