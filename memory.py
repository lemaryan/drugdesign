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
    """Load history, treating a blank CSV as an empty table."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)


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
        try:
            old_rows = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            old_rows = pd.DataFrame(columns=COLUMNS)
        new_rows = pd.concat([old_rows, new_rows], ignore_index=True)

    new_rows.to_csv(path, index=False)


def summarize_history(history, limit=5):
    """Summarize strong candidates and recent real docking results."""
    if history.empty:
        return "No previous results yet."

    history = history.copy()
    if "smiles" in history.columns:
        history = history.dropna(subset=["smiles"])
        history = history.drop_duplicates(subset=["smiles"], keep="last")

    score_columns = [
        "qed",
        "predicted_jak2",
        "predicted_lck",
        "combined_score",
        "docking_jak2",
        "docking_lck",
    ]
    for column in score_columns:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")

    summary_columns = [
        "smiles",
        "qed",
        "predicted_jak2",
        "predicted_lck",
        "combined_score",
    ]
    available = [column for column in summary_columns if column in history.columns]

    if "combined_score" in history.columns:
        top = history.dropna(subset=["combined_score"])
        top = top.sort_values("combined_score", ascending=False).head(limit)
    else:
        top = history.tail(limit)

    sections = ["Best candidates by combined model score:"]
    if top.empty:
        sections.append("No scored candidates yet.")
    else:
        sections.append(top[available].to_string(index=False))

    docking_columns = ["docking_jak2", "docking_lck"]
    if all(column in history.columns for column in docking_columns):
        docked = history.dropna(subset=docking_columns).tail(limit)
        if not docked.empty:
            observed_columns = [
                column for column in summary_columns + docking_columns
                if column in docked.columns
            ]
            sections.extend([
                "Recent actual DOCKSTRING results (kcal/mol):",
                docked[observed_columns].to_string(index=False),
            ])

    return "\n".join(sections)
