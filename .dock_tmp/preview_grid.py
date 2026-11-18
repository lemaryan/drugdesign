import io, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")

fp = rdFingerprintGenerator.GetMorganGenerator(radius=2).GetFingerprint
def transparent_mol_image(smiles, size=(420, 240)):
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.drawOptions().clearBackground = False
    d.DrawMolecule(Chem.MolFromSmiles(smiles)); d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))

seeds = pd.read_csv("data/dockstring_seed_memory.csv")
history = pd.read_csv("data/experiment_history.csv")
history["canon"] = [Chem.MolToSmiles(m) if m else None
                    for m in (Chem.MolFromSmiles(s) for s in history["smiles"])]
memory = history.dropna(subset=["canon"]).drop_duplicates("canon")
memory_fps = [fp(Chem.MolFromSmiles(s)) for s in memory["canon"]]
position = {s: i for i, s in enumerate(memory["canon"])}

def max_similarity_to_others(canon):
    sims = DataStructs.BulkTanimotoSimilarity(memory_fps[position[canon]], memory_fps)
    sims[position[canon]] = -1
    return max(sims)

docked = history.dropna(subset=["actual_objective"]).drop_duplicates("canon").copy()
docked["max_sim"] = [max_similarity_to_others(s) for s in docked["canon"]]

bands = {
    "Least diverse (> 0.70)": docked[docked["max_sim"] > 0.70],
    "0.50 < similarity ≤ 0.70": docked[(docked["max_sim"] > 0.50) & (docked["max_sim"] <= 0.70)],
    "Most diverse (≤ 0.50)": docked[docked["max_sim"] <= 0.50],
}

# --- NEW: the best molecule in the DOCKSTRING dataset, as a reference row ---
library_best = seeds.nsmallest(1, "actual_objective").iloc[0]
library_canon = Chem.MolToSmiles(Chem.MolFromSmiles(library_best["smiles"]))
library_sims = DataStructs.BulkTanimotoSimilarity(
    fp(Chem.MolFromSmiles(library_canon)), memory_fps)
# The seed itself is in the history, so drop the self-match: this is the closest
# the agent's own molecules ever came to the library best.
if library_canon in position:
    library_sims[position[library_canon]] = -1
library_sim = max(library_sims)

plt.figure(figsize=(10, 9.8))
for band_i, (band, group) in enumerate(bands.items()):
    for col_i, row in enumerate(group.nsmallest(3, "actual_objective").itertuples()):
        plt.subplot(4, 3, band_i * 3 + col_i + 1)
        plt.imshow(transparent_mol_image(row.smiles))
        plt.xticks([]); plt.yticks([]); plt.gca().set_frame_on(False)
        plt.xlabel(f"objective {row.actual_objective:.2f}\n"
                   f"JAK2 {row.docking_jak2:.1f}   LCK {row.docking_lck:.1f}\n"
                   f"QED {row.qed:.2f}   Tanimoto {row.max_sim:.2f}",
                   fontsize=9, labelpad=2)
        if col_i == 0:
            plt.ylabel(f"{band}\n(n = {len(group)}, mean {group['actual_objective'].mean():.2f})",
                       fontweight="bold", fontsize=10)

# Empty left cell carries the row label, so it lines up with the three rows above
# (an axes holding an image shrinks to the image aspect and drags its label inwards).
plt.subplot(4, 3, 10)
plt.xticks([]); plt.yticks([]); plt.gca().set_frame_on(False)
plt.ylabel("Best in DOCKSTRING\ndataset (n = 260,149)", fontweight="bold", fontsize=10)

plt.subplot(4, 3, 11)
plt.imshow(transparent_mol_image(library_best["smiles"]))
plt.xticks([]); plt.yticks([]); plt.gca().set_frame_on(False)
plt.xlabel(f"objective {library_best['actual_objective']:.2f}\n"
           f"JAK2 {library_best['docking_jak2']:.1f}   LCK {library_best['docking_lck']:.1f}\n"
           f"QED {library_best['qed']:.2f}   Tanimoto {library_sim:.2f}",
           fontsize=9, labelpad=2)

plt.suptitle("Tanimoto Similarity to Previously Generated Molecules", fontweight="bold", size=20)
plt.tight_layout()
plt.savefig(".dock_tmp/preview_grid.png", dpi=150, transparent=True)
print("library best objective:", round(float(library_best["actual_objective"]), 4),
      "| nearest run molecule Tanimoto:", round(library_sim, 3))
