import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ds = pd.read_csv("dockstring-dataset.tsv", sep="\t")
print("columns:", list(ds.columns)[:12])
print("rows:", len(ds))

# objective = JAK2 - min(LCK+8.1,0) + 10(1-QED). Both correction terms are >= 0,
# so objective >= JAK2. Anything with JAK2 > -9.4 cannot reach -9.4.
cand = ds.dropna(subset=["JAK2", "LCK"])
cand = cand[cand["JAK2"] < -9.0].copy()
print("candidates with JAK2 < -9.0:", len(cand))

def qed(s):
    m = Chem.MolFromSmiles(s)
    return QED.qed(m) if m else np.nan

cand["QED"] = cand["smiles"].apply(qed)
cand["objective"] = (cand["JAK2"].to_numpy()
                     - np.minimum(cand["LCK"].to_numpy() + 8.1, 0)
                     + 10 * (1 - cand["QED"]))
best = cand.nsmallest(5, "objective")
print()
for r in best.itertuples():
    print(f"{r.objective:8.4f}  JAK2={r.JAK2:6.1f} LCK={r.LCK:6.1f} QED={r.QED:.4f}  {r.smiles[:60]}")
