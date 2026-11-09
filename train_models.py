"""Prepare the Dockstring table and train JAK2/LCK Chemprop regressors.

Run this from the Chemprop 2 environment. The source file is kept unchanged;
small two-column CSVs are generated under models/training_data/.
"""

import argparse
import csv
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "dockstring-dataset.tsv"
TARGETS = ("JAK2", "LCK")
CHEMPROP_CLI = shutil.which("chemprop")


def prepare_target_csvs(data_path, output_dir, max_rows=None, seed=0):
    """Validate the input table and write one clean CSV per target."""
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = []
    output_handles = []
    row_counts = {target: 0 for target in TARGETS}
    sampled_rows = {target: [] for target in TARGETS} if max_rows else None
    rngs = {target: random.Random(seed) for target in TARGETS}

    try:
        with data_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source, delimiter="\t")
            required = {"smiles", *TARGETS}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Dataset is missing required columns: {', '.join(sorted(missing))}"
                )

            for target in TARGETS:
                handle = (output_dir / f"{target.lower()}_training.csv").open(
                    "w", newline="", encoding="utf-8"
                )
                output_handles.append(handle)
                writer = csv.writer(handle)
                writer.writerow(["smiles", target])
                writers.append(writer)

            for line_number, row in enumerate(reader, start=2):
                smiles = (row.get("smiles") or "").strip()
                if not smiles:
                    continue
                for target, writer in zip(TARGETS, writers):
                    raw_value = (row.get(target) or "").strip()
                    if not raw_value:
                        continue
                    try:
                        value = float(raw_value)
                    except ValueError as error:
                        raise ValueError(
                            f"Invalid {target} score on line {line_number}: {raw_value!r}"
                        ) from error
                    if not math.isfinite(value):
                        continue
                    row_counts[target] += 1
                    if max_rows is None:
                        writer.writerow([smiles, value])
                    else:
                        # Reservoir sampling gives a reproducible uniform sample
                        # without loading the full dataset into memory.
                        sample = sampled_rows[target]
                        if len(sample) < max_rows:
                            sample.append((smiles, value))
                        else:
                            replacement = rngs[target].randrange(row_counts[target])
                            if replacement < max_rows:
                                sample[replacement] = (smiles, value)

        if max_rows is not None:
            for target, writer in zip(TARGETS, writers):
                writer.writerows(sampled_rows[target])
                row_counts[target] = len(sampled_rows[target])
    finally:
        for handle in output_handles:
            handle.close()

    for target, count in row_counts.items():
        if count < 100:
            raise ValueError(
                f"Only {count} usable {target} rows found; check the dataset before training."
            )
        print(f"Prepared {count:,} labeled {target} molecules.")
    return row_counts


def train_target(
    target, dataset_path, output_dir, ensemble_size, epochs, seed,
    message_hidden_dim, ffn_hidden_dim, depth, ffn_num_layers,
):
    """Train and evaluate one regression endpoint with a scaffold split."""
    target_dir = output_dir / target.lower()
    if CHEMPROP_CLI is None:
        raise RuntimeError(
            "Could not find the Chemprop CLI. Activate the environment where "
            "Chemprop is installed, then run this script again."
        )

    # Invoke the CLI script with this Python interpreter. This avoids using
    # the interpreter recorded in ~/.local/bin/chemprop's shebang, which can
    # accidentally load packages from a different Savio Python installation.
    command = [
        sys.executable, CHEMPROP_CLI, "train",
        "--data-path", str(dataset_path),
        "--smiles-columns", "smiles",
        "--target-columns", target,
        "--task-type", "regression",
        "--split-type", "scaffold_balanced",
        "--split-sizes", "0.8333333333", "0.0833333333", "0.0833333334",
        "--data-seed", str(seed),
        "--ensemble-size", str(ensemble_size),
        "--epochs", str(epochs),
        "--message-hidden-dim", str(message_hidden_dim),
        "--ffn-hidden-dim", str(ffn_hidden_dim),
        "--depth", str(depth),
        "--ffn-num-layers", str(ffn_num_layers),
        "--metrics", "rmse", "mae",
        "--output-dir", str(target_dir),
    ]
    print(f"\nTraining {target}; output: {target_dir}")
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "models" / "trained")
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--message-hidden-dim", type=int, default=4096)
    parser.add_argument("--ffn-hidden-dim", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--ffn-num-layers", type=int, default=4)
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Optional reproducible random sample size per target (useful for quick pilots).",
    )
    args = parser.parse_args()

    if min(
        args.ensemble_size, args.epochs, args.message_hidden_dim,
        args.ffn_hidden_dim, args.depth, args.ffn_num_layers,
    ) < 1:
        parser.error("Model sizes, ensemble size, and epochs must be positive")
    if args.max_rows is not None and args.max_rows < 100:
        parser.error("--max-rows must be at least 100")

    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        parser.error(f"Dataset not found: {data_path}")
    data_dir = PROJECT_DIR / "models" / "training_data"
    prepare_target_csvs(data_path, data_dir, args.max_rows, args.seed)

    for target in TARGETS:
        train_target(
            target,
            data_dir / f"{target.lower()}_training.csv",
            args.output_dir.resolve(),
            args.ensemble_size,
            args.epochs,
            args.seed,
            args.message_hidden_dim,
            args.ffn_hidden_dim,
            args.depth,
            args.ffn_num_layers,
        )

    print("\nTraining finished. Point the agent at these directories:")
    print(f"export JAK2_MODEL_PATH=\"{args.output_dir.resolve() / 'jak2'}\"")
    print(f"export LCK_MODEL_PATH=\"{args.output_dir.resolve() / 'lck'}\"")


if __name__ == "__main__":
    main()
