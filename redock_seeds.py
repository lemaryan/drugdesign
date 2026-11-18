"""Re-dock selected molecules with several Vina seeds to measure scoring noise.

DOCKSTRING's Target.dock uses a fixed default seed (974528263), so a single
history entry reports one draw from the docking search distribution. This
script repeats the docking with different seeds and reports the spread, which
is what any claimed objective improvement has to be larger than.

Runs in the Python 3.10 docking environment:

    $DOCKING_PYTHON -s redock_seeds.py --targets data/redock_targets.csv \
        --out data/redock_seeds.csv --seeds 974528263,1,2,3,4
"""

import argparse
import csv
import math
import multiprocessing
import os
import sys
import time


# Mirrors scoring.calculate_objective so this script does not need to import
# the agent environment's modules (it runs under the docking interpreter).
LCK_SCORE_CUTOFF = 8.1
QED_PENALTY_WEIGHT = 10.0


def calculate_objective(jak2, lck, qed):
    """Lower is better, as in scoring.calculate_objective."""
    return float(
        jak2 - min(lck + LCK_SCORE_CUTOFF, 0.0) + QED_PENALTY_WEIGHT * (1.0 - qed)
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, help="CSV of molecules to re-dock.")
    parser.add_argument("--out", required=True, help="Where to write per-seed scores.")
    parser.add_argument(
        "--seeds",
        default="974528263,20260101,20260102,20260103,20260104",
        help="Comma-separated Vina seeds. The first should be DOCKSTRING's "
        "default so the original scores can be checked for reproducibility.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Molecule/seed pairs to dock concurrently. Each worker gets an "
        "equal share of the available CPUs.",
    )
    return parser.parse_args()


# Each worker process loads the receptors once and reuses them.
_TARGETS = {}


def _dock_pair(job):
    """Dock one molecule with one seed against both targets."""
    molecule, seed, cpus = job

    from dockstring import load_target

    if not _TARGETS:
        for name in ("JAK2", "LCK"):
            _TARGETS[name] = load_target(name)

    smiles = molecule["smiles"]
    qed = float(molecule["qed"])

    row = {
        "rank": molecule["rank"],
        "iteration": molecule["iteration"],
        "smiles": smiles,
        "qed": f"{qed:.6f}",
        "seed": seed,
        "orig_jak2": molecule["orig_jak2"],
        "orig_lck": molecule["orig_lck"],
        "orig_objective": molecule["orig_objective"],
        "error": "",
    }

    started = time.time()
    scores = {}
    for name, target in _TARGETS.items():
        try:
            score, _ = target.dock(smiles, num_cpus=cpus, seed=seed)
        except Exception as error:  # a failed pose should not stop the sweep
            row["error"] = f"{name}: {type(error).__name__}: {error}"[:300]
            score = None

        if score is None or not math.isfinite(float(score)):
            if not row["error"]:
                row["error"] = f"{name}: no finite score"
            scores = {}
            break

        scores[name] = float(score)

    if scores:
        row["docking_jak2"] = scores["JAK2"]
        row["docking_lck"] = scores["LCK"]
        row["objective"] = calculate_objective(scores["JAK2"], scores["LCK"], qed)

    return row, time.time() - started


def main():
    args = parse_arguments()

    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "Run this with the docking environment's Python 3.10, not "
            f"Python {sys.version_info.major}.{sys.version_info.minor}."
        )

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    with open(args.targets, newline="") as handle:
        molecules = list(csv.DictReader(handle))

    workers = max(1, min(args.workers, len(molecules) * len(seeds)))
    cpus_per_worker = max(1, total_cpus // workers)

    jobs = [
        (molecule, seed, cpus_per_worker)
        for molecule in molecules
        for seed in seeds
    ]

    print(
        f"Re-docking {len(molecules)} molecules x {len(seeds)} seeds "
        f"x 2 targets = {len(jobs)} pairs; {workers} workers x "
        f"{cpus_per_worker} CPUs (of {total_cpus}).",
        flush=True,
    )

    fieldnames = [
        "rank",
        "iteration",
        "smiles",
        "qed",
        "seed",
        "docking_jak2",
        "docking_lck",
        "objective",
        "orig_jak2",
        "orig_lck",
        "orig_objective",
        "error",
    ]

    # Write incrementally so a timeout still leaves usable results.
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        if workers == 1:
            results = (_dock_pair(job) for job in jobs)
        else:
            pool = multiprocessing.Pool(processes=workers)
            results = pool.imap_unordered(_dock_pair, jobs)

        done = 0
        for row, elapsed in results:
            writer.writerow(row)
            handle.flush()  # keep partial results if the job hits its time limit
            done += 1
            print(
                f"[{done}/{len(jobs)}] rank {row['rank']:>8} seed {row['seed']:>10} "
                f"jak2={row.get('docking_jak2', 'NA')} "
                f"lck={row.get('docking_lck', 'NA')} "
                f"obj={row.get('objective', 'NA')} "
                f"({elapsed:.1f}s) {row['error']}",
                flush=True,
            )

        if workers > 1:
            pool.close()
            pool.join()

    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
