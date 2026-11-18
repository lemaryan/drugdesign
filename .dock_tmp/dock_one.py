"""Dock one SMILES against JAK2/LCK and print the project objective."""
import sys
sys.path.insert(0, "/global/home/users/maryanle/drug_design_clean")
from docking import dock_molecule
from molecule_tools import inspect_molecule
from scoring import calculate_objective

smiles = sys.argv[1]
props = inspect_molecule(smiles)
scores = dock_molecule(smiles)
jak2 = scores["docking_jak2"]
lck = scores["docking_lck"]
qed = props["qed"]
obj = calculate_objective(jak2, lck, qed)
print(f"canonical: {props['smiles']}")
print(f"jak2: {jak2}")
print(f"lck: {lck}")
print(f"qed: {qed:.4f}")
print(f"objective: {obj:.4f}")
print(f"objective_paper_convention: {obj + 8.1:.4f}")
