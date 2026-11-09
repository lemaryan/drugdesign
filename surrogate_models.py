"""Batch predictions from the Chemprop 2 checkpoints used by this project.

The Chemprop command-line interface is the stable boundary here: training and
prediction both run with the same installed Chemprop version and checkpoint.
"""

import csv
import os
import subprocess
import tempfile
from pathlib import Path


def _predict_target(smiles_list, target, env_name):
    """Predict one endpoint for a list of SMILES using a Chemprop checkpoint."""
    model_path = os.getenv(env_name)
    if not model_path:
        raise RuntimeError(f"Set {env_name} to the trained model directory.")

    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"{env_name} does not exist: {model_path}")
    if not smiles_list:
        return []

    with tempfile.TemporaryDirectory(prefix="merck_agent_predict_") as tmp_dir:
        input_path = Path(tmp_dir) / "molecules.csv"
        output_path = Path(tmp_dir) / "predictions.csv"
        with input_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["smiles"])
            writer.writerows([[smiles] for smiles in smiles_list])

        command = [
            "chemprop", "predict",
            "--test-path", str(input_path),
            "--model-paths", str(model_path),
            "--smiles-columns", "smiles",
            "--preds-path", str(output_path),
        ]
        subprocess.run(command, check=True)

        with output_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(smiles_list):
            raise RuntimeError(
                f"Chemprop returned {len(rows)} {target} predictions for "
                f"{len(smiles_list)} molecules."
            )
        prediction_columns = [key for key in rows[0] if key != "smiles"]
        if not prediction_columns:
            raise RuntimeError(f"Chemprop output has no {target} prediction column.")

        # A directory may contain an ensemble. Chemprop writes its aggregate
        # prediction in the output table; there should be one target column.
        try:
            return [float(row[prediction_columns[0]]) for row in rows]
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Chemprop returned a non-numeric {target} score.") from error


def predict_many(smiles_list):
    """Return predictions for both targets, keeping input order intact."""
    jak2_values = _predict_target(smiles_list, "JAK2", "JAK2_MODEL_PATH")
    lck_values = _predict_target(smiles_list, "LCK", "LCK_MODEL_PATH")
    return [
        {"predicted_jak2": jak2, "predicted_lck": lck}
        for jak2, lck in zip(jak2_values, lck_values)
    ]


