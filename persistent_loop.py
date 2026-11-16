"""Checkpointed optimization loop intended for long Savio allocations.

Each completed iteration is appended to the experiment CSV. Restart with
``--resume`` after a walltime limit to continue from that memory.
"""

import argparse
import signal
from pathlib import Path

import pandas as pd

from agent import log, run_agent
from config import (
    DEFAULT_CANDIDATES,
    DEFAULT_DOCK_TOP,
    HISTORY_FILE,
    LIBRARY_BEST_25TH_OBJECTIVE,
    LIBRARY_BEST_OBJECTIVE,
    PAPER_CONVENTION_OFFSET,
)
from memory import load_history


stop_requested = False

# Consecutive iteration failures tolerated before the loop gives up.
MAX_CONSECUTIVE_FAILURES = 5


def request_stop(signum, _frame):
    global stop_requested
    stop_requested = True
    log(f"Received signal {signum}; will stop after the current iteration is saved")


def next_iteration(history_path):
    history = load_history(history_path)
    if history.empty or "iteration" not in history:
        return 1
    values = pd.to_numeric(history["iteration"], errors="coerce").dropna()
    return int(values.max()) + 1 if not values.empty else 1


def best_actual(history_path):
    history = load_history(history_path)
    if history.empty:
        return None
    values = pd.to_numeric(history.get("actual_objective"), errors="coerce").dropna()
    return float(values.min()) if not values.empty else None


def benchmark_line(history_path):
    """Report top-1 and top-25 against the DOCKSTRING library reference lines.

    These are the two values the DOCKSTRING paper plots as its reference (Figure 10
    marks "the best (and 25th best) value in the dockstring dataset"), so they are
    the pair worth watching rather than top-1 alone.
    """
    history = load_history(history_path)
    if history.empty or "smiles" not in history.columns:
        return None

    rows = history.copy()
    rows["actual_objective"] = pd.to_numeric(
        rows.get("actual_objective"), errors="coerce"
    )
    rows = rows.dropna(subset=["actual_objective"])
    rows = rows.drop_duplicates("smiles")
    if rows.empty:
        return None

    scores = rows["actual_objective"].sort_values().tolist()
    best = scores[0]
    parts = [
        f"docked={len(scores)}",
        f"top-1={best:.2f} (library {LIBRARY_BEST_OBJECTIVE:.2f}"
        f"{', BEATEN' if best < LIBRARY_BEST_OBJECTIVE else ''})",
    ]
    if len(scores) >= 25:
        top25 = scores[24]
        parts.append(
            f"top-25={top25:.2f} (library {LIBRARY_BEST_25TH_OBJECTIVE:.2f}"
            f"{', BEATEN' if top25 < LIBRARY_BEST_25TH_OBJECTIVE else ''})"
        )
    else:
        parts.append(f"top-25=n/a ({len(scores)}/25 docked)")
    parts.append(f"paper convention: top-1={best + PAPER_CONVENTION_OFFSET:.2f}")
    return "; ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Checkpointed JAK2 optimization loop")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Maximum additional iterations for this allocation")
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--dock-top", type=int, default=DEFAULT_DOCK_TOP)
    parser.add_argument("--history", type=Path, default=HISTORY_FILE)
    parser.add_argument("--resume", action="store_true",
                        help="Continue numbering from the last completed history row")
    parser.add_argument("--run-docking", action="store_true")
    parser.add_argument("--stop-objective", type=float, default=None,
                        help="Stop when a docked candidate reaches this actual objective")
    args = parser.parse_args()
    if min(args.iterations, args.candidates, args.dock_top) < 1:
        parser.error("iterations, candidates, and dock-top must be positive")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    iteration = next_iteration(args.history) if args.resume else 1
    completed = 0
    log(f"Persistent loop starting at iteration {iteration}; memory: {args.history}")
    consecutive_failures = 0
    while completed < args.iterations and not stop_requested:
        # One bad iteration must not end the allocation. Run 41379608 lost an
        # eight-hour job to a single transient API error raised inside the
        # propose node, so failures are logged and the loop moves on. Repeated
        # back-to-back failures mean something is actually broken, not flaky,
        # and then it does stop.
        try:
            run_agent(1, args.candidates, args.dock_top, args.run_docking,
                      args.history, iteration)
            consecutive_failures = 0
        except Exception as error:  # noqa: BLE001 - keep the allocation alive
            consecutive_failures += 1
            log(
                f"Iteration {iteration} failed ({type(error).__name__}: "
                f"{str(error)[:200]}); continuing "
                f"[{consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}]"
            )
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log(
                    f"Stopping: {consecutive_failures} iterations failed in a row, "
                    "which is a real fault rather than a flaky endpoint"
                )
                break
        completed += 1
        iteration += 1
        best = best_actual(args.history)
        if best is not None:
            log(f"Best actual objective in memory: {best:.4f}")
            summary = benchmark_line(args.history)
            if summary:
                log(f"Benchmark: {summary}")
            if args.stop_objective is not None and best <= args.stop_objective:
                log(f"Reached stopping objective {args.stop_objective:.4f}")
                break
    log(f"Persistent loop stopped after {completed} iterations; memory retained at {args.history}")


if __name__ == "__main__":
    main()
