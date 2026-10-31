"""Run DOCKSTRING in its own Python environment."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys


RESULT_PREFIX = "DOCKING_RESULT="


def _docking_python():
    """Find the Python executable in the separate docking environment."""
    configured_path = os.environ.get("DOCKING_PYTHON")
    if configured_path:
        python = Path(configured_path).expanduser()
    else:
        # The two conda environments are expected to be siblings:
        # .../envs/drugdesign and .../envs/docking.
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            raise RuntimeError(
                "Set DOCKING_PYTHON to the Python executable in the docking "
                "environment."
            )
        python = Path(conda_prefix).parent / "docking" / "bin" / "python"

    if not python.is_file():
        raise RuntimeError(
            f"Could not find the docking Python at {python}. Set DOCKING_PYTHON "
            "to the path of the Python 3.10 executable in your docking environment."
        )
    return str(python)


def dock_molecule(smiles):
    """Return JAK2 and LCK Vina affinities for a SMILES string."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("Please provide a non-empty SMILES string.")

    command = [
        _docking_python(),
        "-s",  # Do not load packages from ~/.local into the docking environment.
        str(Path(__file__).resolve()),
        "--dock-worker",
    ]
    result = subprocess.run(
        command,
        input=json.dumps({"smiles": smiles}),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"DOCKSTRING worker failed: {details[-2000:]}")

    # DOCKSTRING may write progress messages to stdout, so look for our result
    # marker instead of assuming its output contains only JSON.
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])

    raise RuntimeError(
        "The docking worker finished without returning scores. "
        f"Worker output: {result.stdout[-2000:]}"
    )


def _dock_in_docking_environment(smiles):
    """This function runs only in the Python 3.10 docking environment."""
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "The DOCKSTRING worker must use Python 3.10; "
            f"it is running Python {sys.version_info.major}.{sys.version_info.minor}."
        )

    try:
        from dockstring import load_target
    except ImportError as error:
        raise RuntimeError(
            "DOCKSTRING is not installed in the docking environment."
        ) from error

    # Use the number of CPUs Slurm gave the job, or one CPU when run locally.
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    scores = {}

    # DOCKSTRING docks against one protein target at a time.
    for target_name in ["JAK2", "LCK"]:
        target = load_target(target_name)
        score, _ = target.dock(smiles, num_cpus=cpus)

        if score is None or not math.isfinite(float(score)):
            raise RuntimeError(
                f"DOCKSTRING did not return a valid score for {target_name}."
            )

        scores[f"docking_{target_name.lower()}"] = float(score)

    return scores


if __name__ == "__main__" and "--dock-worker" in sys.argv:
    request = json.loads(sys.stdin.read())
    scores = _dock_in_docking_environment(request["smiles"])
    print(RESULT_PREFIX + json.dumps(scores))
