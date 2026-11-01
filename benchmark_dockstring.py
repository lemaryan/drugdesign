"""Compute the DOCKSTRING selective-JAK2 objective across the local dataset."""

import argparse
import csv
import math
import multiprocessing
import statistics
import time
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import QED


def qed_for(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    return float(QED.qed(molecule)) if molecule is not None else None


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dockstring-dataset.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "dockstring_jak2_benchmark.md"
    csv_path = args.output_dir / "dockstring_jak2_benchmark_top100.csv"

    rows, invalid, missing, processed = [], 0, 0, 0
    started = time.time()
    # Fork avoids re-importing stdin or the full script on macOS when this is
    # invoked in a shell heredoc, while each worker computes RDKit QED.
    context = multiprocessing.get_context("fork")
    with args.data.open(newline="", encoding="utf-8") as source, context.Pool(args.workers) as pool:
        reader = csv.DictReader(source, delimiter="\t")
        while True:
            batch = []
            try:
                for _ in range(4000):
                    row = next(reader)
                    try:
                        jak2, lck = float(row["JAK2"]), float(row["LCK"])
                        if not math.isfinite(jak2) or not math.isfinite(lck):
                            raise ValueError
                    except (TypeError, ValueError):
                        missing += 1
                        continue
                    batch.append((row, jak2, lck))
            except StopIteration:
                pass
            if not batch:
                break

            qeds = pool.map(qed_for, (item[0]["smiles"] for item in batch), chunksize=64)
            for (row, jak2, lck), qed in zip(batch, qeds):
                if qed is None:
                    invalid += 1
                    continue
                objective = jak2 - min(lck, -8.1) + 10 * (1 - qed)
                rows.append({
                    "inchikey": row.get("inchikey", ""),
                    "smiles": row["smiles"],
                    "JAK2": jak2,
                    "LCK": lck,
                    "QED": qed,
                    "objective": objective,
                })
            processed += len(batch)
            print(f"Processed {processed:,} rows in {time.time() - started:.0f}s", flush=True)

    if not rows:
        raise RuntimeError("No usable DOCKSTRING rows found.")
    rows.sort(key=lambda row: row["objective"])
    gated_rows = sorted((row for row in rows if row["QED"] >= 0.4), key=lambda row: row["objective"])
    objectives = [row["objective"] for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["rank", "inchikey", "smiles", "JAK2", "LCK", "QED", "objective"])
        writer.writeheader()
        for rank, row in enumerate(rows[:100], 1):
            writer.writerow({"rank": rank, **row})

    best, best_gated = rows[0], gated_rows[0] if gated_rows else None
    lines = [
        "# DOCKSTRING Selective-JAK2 Dataset Benchmark", "",
        "Computed from `dockstring-dataset.tsv` using the paper objective:", "",
        "`objective = JAK2 - min(LCK, -8.1) + 10 × (1 - QED)`", "",
        "Lower is better. QED was calculated from each SMILES with RDKit. This is an in-dataset reference, not a universal success threshold. The best value is an optimistic reference because these molecules are also the source dataset for the project’s surrogate models.", "",
        "## Dataset summary", "",
        f"- Valid molecules with finite JAK2 and LCK scores: **{len(rows):,}**",
        f"- Missing/non-finite target rows skipped: **{missing:,}**",
        f"- Invalid SMILES skipped: **{invalid:,}**",
        f"- RDKit version: **{rdBase.rdkitVersion}**",
        f"- Objective mean: **{statistics.mean(objectives):.4f}**",
        f"- Objective median: **{statistics.median(objectives):.4f}**",
        f"- Objective 5th / 25th / 75th / 95th percentiles: **{quantile(objectives, .05):.4f} / {quantile(objectives, .25):.4f} / {quantile(objectives, .75):.4f} / {quantile(objectives, .95):.4f}**",
        f"- Best objective in full dataset: **{best['objective']:.4f}**",
        f"- Best objective among molecules with QED ≥ 0.4: **{best_gated['objective']:.4f}**" if best_gated else "- No molecules pass QED ≥ 0.4.",
        "", "## Best molecule in the starting dataset", "",
        "| Objective rank | Objective | JAK2 | LCK | QED | SMILES |", "|---:|---:|---:|---:|---:|---|",
        f"| 1 | {best['objective']:.4f} | {best['JAK2']:.2f} | {best['LCK']:.2f} | {best['QED']:.4f} | `{best['smiles']}` |",
        "", "## Best starting-dataset molecule passing QED ≥ 0.4", "",
        "| Objective | JAK2 | LCK | QED | SMILES |", "|---:|---:|---:|---:|---|",
        (f"| {best_gated['objective']:.4f} | {best_gated['JAK2']:.2f} | {best_gated['LCK']:.2f} | {best_gated['QED']:.4f} | `{best_gated['smiles']}` |" if best_gated else "| — | — | — | — | — |"),
        "", "## How to use the benchmark", "",
        "Compare candidates scored with actual DOCKSTRING JAK2/LCK docking and RDKit QED against this distribution and the best starting-dataset molecule. Use the same score source on both sides: Chemprop predictions are estimates and should be compared separately from actual docking. The project’s QED ≥ 0.4 gate is not part of the paper objective; it is shown as a separate filtered comparison. The full top 100 rows are in `dockstring_jak2_benchmark_top100.csv`.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Best: {best}")
    print(f"Best with QED >= 0.4: {best_gated}")
    print(f"Wrote {report_path} and {csv_path}")


if __name__ == "__main__":
    main()
