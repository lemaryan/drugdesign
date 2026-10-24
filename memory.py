"""CSV-based memory for previously evaluated molecules."""

from pathlib import Path

import pandas as pd


COLUMNS = [
    "iteration",
    "smiles",
    "qed",
    "predicted_jak2",
    "predicted_lck",
    "combined_score",
    "passes_filters",
    "docking_jak2",
    "docking_lck",
]


def load_history(path):
    """Load experiment history, or return an empty table."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def known_smiles(history):
    """Return the canonical SMILES already stored in memory."""
    if history.empty or "smiles" not in history:
        return set()
    return set(history["smiles"].dropna().astype(str))


def save_results(results, path):
    """Append this iteration's results to the CSV history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_rows = pd.DataFrame(results)
    for column in COLUMNS:
        if column not in new_rows:
            new_rows[column] = None
    new_rows = new_rows[COLUMNS]

    if path.exists():
        old_rows = pd.read_csv(path)
        new_rows = pd.concat([old_rows, new_rows], ignore_index=True)

    new_rows.to_csv(path, index=False)


def summarize_history(history, limit=5):
    """Make a short feedback summary for the next LLM prompt."""
    if history.empty:
        return "No previous results yet."

    columns = ["smiles", "qed", "predicted_jak2", "predicted_lck", "combined_score"]
    available = [column for column in columns if column in history.columns]
    recent = history.tail(limit)[available]

    return recent.to_string(index=False)
