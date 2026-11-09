"""Create seed examples from the Dockstring dataset."""

import argparse
import csv
import math
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from config import PROJECT_DIR, SEED_MEMORY_FILE
from molecule_tools import check_filters
from scoring import calculate_objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "dockstring-dataset.tsv")
    parser.add_argument("--output", type=Path, default=SEED_MEMORY_FILE)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be positive")

    best = []
    rows_seen = 0
    with args.data.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, delimiter="\t")
        for row in reader:
            rows_seen += 1
            if rows_seen % 50000 == 0:
                print(f"Checked {rows_seen:,} dataset rows.", flush=True)

            try:
                jak2 = float(row["JAK2"])
                lck = float(row["LCK"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(jak2) or not math.isfinite(lck):
                continue

            # QED's penalty cannot be negative, so this is a lower bound.
            # Skip RDKit property work if this row cannot beat the current top set.
            lower_bound = calculate_objective(jak2, lck, 1.0)
            if len(best) == args.limit and lower_bound >= best[-1]["actual_objective"]:
                continue

            smiles = (row.get("smiles") or "").strip()
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue

            properties = {
                "qed": float(QED.qed(molecule)),
                "molecular_weight": float(Descriptors.MolWt(molecule)),
                "logp": float(Descriptors.MolLogP(molecule)),
                "hbd": int(Descriptors.NumHDonors(molecule)),
                "hba": int(Descriptors.NumHAcceptors(molecule)),
                "rotatable_bonds": int(Descriptors.NumRotatableBonds(molecule)),
                "tpsa": float(Descriptors.TPSA(molecule)),
                "heavy_atoms": int(Descriptors.HeavyAtomCount(molecule)),
            }
            passes, _ = check_filters(properties)
            if not passes:
                continue

            best.append({
                "source": "dockstring-dataset.tsv",
                "inchikey": row.get("inchikey", ""),
                "smiles": Chem.MolToSmiles(molecule),
                "qed": properties["qed"],
                "docking_jak2": jak2,
                "docking_lck": lck,
                "actual_objective": calculate_objective(jak2, lck, properties["qed"]),
                "passes_filters": True,
            })
            best.sort(key=lambda item: item["actual_objective"])
            best = best[:args.limit]

    if not best:
        raise RuntimeError("No Dockstring molecules passed the filters.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "rank", "source", "inchikey", "smiles", "qed", "docking_jak2",
        "docking_lck", "actual_objective", "passes_filters",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for rank, row in enumerate(best, start=1):
            writer.writerow({"rank": rank, **row})

    print(f"Saved {len(best)} Dockstring seed examples to {args.output}")
    print(f"Best actual objective: {best[0]['actual_objective']:.3f}")


if __name__ == "__main__":
    main()
