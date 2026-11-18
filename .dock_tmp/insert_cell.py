import json, uuid

NB = "analysis.ipynb"
nb = json.load(open(NB))

source = '''# The best-objective molecule in the full DOCKSTRING dataset: the bar the agent
# has to beat (config.LIBRARY_BEST_OBJECTIVE). Needs `dataset` with QED and
# objective from the two cells above. Same panel layout as the Tanimoto figures.
best = dataset.dropna(subset=["objective"]).nsmallest(1, "objective").iloc[0]

plt.figure(figsize=(7, 4.5))
plt.imshow(transparent_mol_image(best["smiles"], size=(560, 320)))
plt.xticks([]); plt.yticks([]); plt.gca().set_frame_on(False)
plt.xlabel(f"objective {best['objective']:.2f}\\n"
           f"JAK2 {best['JAK2']:.1f}   LCK {best['LCK']:.1f}\\n"
           f"QED {best['QED']:.2f}",
           fontsize=9, labelpad=2)
plt.ylabel(f"Best in DOCKSTRING dataset\\n"
           f"(n = {dataset['objective'].notna().sum():,}, "
           f"mean {dataset['objective'].mean():.2f})",
           fontweight="bold", fontsize=10)

plt.suptitle("Best Molecule in the DOCKSTRING Dataset", fontweight="bold", size=16)
plt.tight_layout()
plt.savefig("figures/dockstring_best_molecule.png", dpi=300, transparent=True)
plt.show()

print(best["smiles"])
'''

cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": uuid.uuid4().hex[:8],
    "metadata": {},
    "outputs": [],
    "source": source.splitlines(keepends=True),
}

# Insert immediately after the cell that computes dataset["objective"].
target = next(i for i, c in enumerate(nb["cells"])
              if 'dataset["objective"] = (dataset["JAK2"]' in "".join(c["source"]))
nb["cells"].insert(target + 1, cell)
json.dump(nb, open(NB, "w"), indent=1)
print(f"inserted after cell index {target} -> new cell at index {target + 1}")
print(f"notebook now has {len(nb['cells'])} cells")
