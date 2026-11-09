"""CSV-based memory for previously evaluated molecules."""

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from config import TARGET_JAK2_OBJECTIVE
from molecule_tools import inspect_molecule, tanimoto_similarity
from scoring import calculate_objective


COLUMNS = [
    "iteration",
    "smiles",
    "qed",
    "predicted_objective",
    "actual_objective",
    "passes_filters",
    "docking_jak2",
    "docking_lck",
    "proposal_status",
    "rejection_reason",
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
        if "predicted_objective" not in old_rows and "jak2_objective" in old_rows:
            old_rows["predicted_objective"] = old_rows["jak2_objective"]
        for column in COLUMNS:
            if column not in old_rows:
                old_rows[column] = None
        old_rows = old_rows[COLUMNS]
        new_rows = pd.concat([old_rows, new_rows], ignore_index=True)

    new_rows.to_csv(path, index=False)


def _passed_filters(history):
    """Return a boolean series for molecules that passed the hard filters."""
    if "passes_filters" not in history.columns:
        return pd.Series(True, index=history.index)
    return history["passes_filters"].astype(str).str.lower().eq("true")


def _scaffold(smiles):
    """Return a molecule's Bemis-Murcko scaffold."""
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() == 0:
        return Chem.MolToSmiles(molecule)
    return Chem.MolToSmiles(scaffold)


def _diverse_examples(candidates, limit):
    """Pick low-scoring examples that are structurally different."""
    candidates = candidates.drop_duplicates("smiles")
    candidates = candidates.head(100).to_dict("records")
    selected = []
    fingerprints = {}

    while candidates and len(selected) < limit:
        if not selected:
            winner = candidates[0]
            properties = inspect_molecule(winner["smiles"])
            fingerprints[winner["smiles"]] = (
                properties["fingerprint"] if properties else None
            )
        else:
            ranked = []
            for row in candidates:
                smiles = row["smiles"]
                if smiles not in fingerprints:
                    properties = inspect_molecule(smiles)
                    fingerprints[smiles] = (
                        properties["fingerprint"] if properties else None
                    )
                fingerprint = fingerprints[smiles]
                similarities = []
                if fingerprint is not None:
                    similarities = [
                        tanimoto_similarity(fingerprint, fingerprints[item["smiles"]])
                        for item in selected
                        if fingerprints.get(item["smiles"]) is not None
                    ]
                max_similarity = max(similarities, default=0.0)
                ranked.append((max_similarity, row["_selection_objective"], row))
            winner = min(ranked, key=lambda item: (item[0], item[1]))[2]

        selected.append(winner)
        candidates.remove(winner)

    return selected


def _format_examples(rows, columns):
    """Format selected memory rows as a short table."""
    if not rows:
        return "No examples available."
    table = pd.DataFrame(rows).reindex(columns=columns)
    return table.to_string(index=False)


def _iteration_trends(history, passed):
    """Summarize recent objective, filter, scaffold, and improvement trends."""
    if "iteration" not in history.columns:
        return "Iteration trend data is not available."

    history = history.copy()
    history["iteration"] = pd.to_numeric(history["iteration"], errors="coerce")
    history = history.dropna(subset=["iteration"])
    if history.empty:
        return "Iteration trend data is not available."

    iterations = sorted(history["iteration"].unique())
    latest = history[history["iteration"] == iterations[-1]]
    latest_passed = latest.loc[passed.reindex(latest.index, fill_value=False)]
    latest_scores = pd.to_numeric(
        latest_passed.get("predicted_objective", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()

    statuses = latest.get("proposal_status", pd.Series("", index=latest.index))
    counted = ~statuses.isin(["invalid", "duplicate", "dataset_duplicate"])
    pass_rate = passed.loc[latest.index[counted]].mean() if counted.any() else 0.0
    scaffolds = {
        scaffold for scaffold in latest_passed["smiles"].map(_scaffold)
        if scaffold is not None
    } if "smiles" in latest_passed else set()

    summary = [f"Latest iteration: {int(iterations[-1])}"]
    if len(latest_scores):
        summary.append(f"mean predicted objective={latest_scores.mean():.2f}")
    summary.append(f"filter pass rate={pass_rate:.0%}")
    summary.append(f"unique passing scaffolds={len(scaffolds)}")

    if len(iterations) > 1:
        previous = history[history["iteration"] == iterations[-2]]
        previous_passed = previous.loc[passed.reindex(previous.index, fill_value=False)]
        previous_scores = pd.to_numeric(
            previous_passed.get("predicted_objective", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        if len(latest_scores) and len(previous_scores):
            change = latest_scores.mean() - previous_scores.mean()
            trend = "improved" if change < 0 else "worsened" if change > 0 else "unchanged"
            summary.append(
                f"mean predicted objective {trend} vs previous iteration "
                f"({previous_scores.mean():.2f} to {latest_scores.mean():.2f})"
            )

    return "; ".join(summary)


def summarize_history(history, seed_memory=None):
    """Build LLM context from Dockstring examples and agent-generated history."""
    sections = [
        f"Goal: reach a JAK2 objective near {TARGET_JAK2_OBJECTIVE:.1f}; lower is better.",
    ]
    seed_smiles = set()
    if seed_memory is not None and not seed_memory.empty:
        seed = seed_memory.copy()
        if "passes_filters" in seed:
            seed = seed.loc[_passed_filters(seed)]
        if "actual_objective" in seed.columns:
            seed["actual_objective"] = pd.to_numeric(
                seed["actual_objective"], errors="coerce"
            )
            seed = seed.dropna(subset=["actual_objective"])
        else:
            seed = seed.iloc[0:0]
        seed = seed.sort_values("actual_objective").drop_duplicates("smiles").head(15)
        seed_smiles = set(seed["smiles"].dropna().astype(str))
        seed_columns = [
            "source", "smiles", "qed", "docking_jak2", "docking_lck",
            "actual_objective",
        ]
        sections.extend([
            "Dockstring dataset seed examples (actual scores; reference knowledge only, not agent results):",
            _format_examples(seed.to_dict("records"), seed_columns),
        ])
    if history.empty:
        sections.append("No agent-generated molecules are available yet.")
        return "\n".join(sections)
    if "smiles" not in history.columns:
        sections.append("No molecule records are available yet.")
        return "\n".join(sections)

    history = history.copy()
    history["smiles"] = history["smiles"].fillna("").astype(str)
    history = history[history["smiles"] != ""]
    if seed_smiles:
        history = history[~history["smiles"].isin(seed_smiles)]
    if history.empty:
        sections.append("No agent-generated molecules outside the seed examples yet.")
        return "\n".join(sections)
    passed = _passed_filters(history)

    numeric_columns = [
        "iteration", "qed", "predicted_objective", "actual_objective",
        "docking_jak2", "docking_lck",
    ]
    for column in numeric_columns:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")

    # Read older history files, then use the separate current column names.
    if "predicted_objective" not in history.columns:
        old_objective = history.get(
            "jak2_objective", pd.Series(float("nan"), index=history.index)
        )
        history["predicted_objective"] = pd.to_numeric(old_objective, errors="coerce")
    if "actual_objective" not in history.columns:
        history["actual_objective"] = None
    actual_columns = ["docking_jak2", "docking_lck", "qed"]
    if all(column in history.columns for column in actual_columns):
        docked_rows = history.dropna(subset=actual_columns).copy()
    else:
        docked_rows = history.iloc[0:0].copy()
    if not docked_rows.empty:
        docked_rows["actual_objective"] = [
            calculate_objective(jak2, lck, qed)
            for jak2, lck, qed in zip(
                docked_rows["docking_jak2"],
                docked_rows["docking_lck"],
                docked_rows["qed"],
            )
        ]
        history.loc[docked_rows.index, "actual_objective"] = docked_rows[
            "actual_objective"
        ]
    passing_rows = history.loc[passed].copy()
    predicted = passing_rows.dropna(subset=["predicted_objective"])
    predicted = predicted.sort_values("predicted_objective")
    predicted = predicted.drop_duplicates("smiles", keep="first")
    predicted_rows = predicted.head(15).to_dict("records")

    actual = docked_rows.loc[passed.reindex(docked_rows.index, fill_value=False)]
    actual = actual.sort_values("actual_objective")
    actual = actual.drop_duplicates("smiles", keep="first")
    actual_rows = actual.head(15).to_dict("records")

    candidates = passing_rows.copy()
    candidates["_selection_objective"] = candidates["predicted_objective"]
    missing_score = candidates["_selection_objective"].isna()
    if "actual_objective" in candidates:
        candidates.loc[missing_score, "_selection_objective"] = candidates.loc[
            missing_score, "actual_objective"
        ]
    candidates = candidates.dropna(subset=["_selection_objective"])
    candidates = candidates.sort_values("_selection_objective")
    diverse_rows = _diverse_examples(candidates, 10)

    status = history.get("proposal_status", pd.Series("", index=history.index))
    failures = history[status.isin(["filtered", "invalid"])].copy()
    if "iteration" in failures:
        failures = failures.sort_values("iteration", ascending=False)
    failures = failures.drop_duplicates("smiles", keep="first")
    failure_rows = failures.drop_duplicates("rejection_reason", keep="first").head(5)
    failure_rows = failure_rows.to_dict("records")

    best_lines = []
    if not predicted.empty:
        best_lines.append(
            f"Best predicted objective: {predicted.iloc[0]['predicted_objective']:.2f}"
        )
    if not actual.empty:
        best_lines.append(
            f"Best actual objective: {actual.iloc[0]['actual_objective']:.2f}"
        )

    # Keep one entry per molecule across all four memory groups.
    seen = set()

    def unique_rows(rows):
        result = []
        for row in rows:
            smiles = row["smiles"]
            if smiles not in seen:
                seen.add(smiles)
                result.append(row)
        return result

    predicted_rows = unique_rows(predicted_rows)
    actual_rows = unique_rows(actual_rows)
    diverse_rows = unique_rows(diverse_rows)
    failure_rows = unique_rows(failure_rows)

    sections.extend([
        "Best results: " + "; ".join(best_lines) if best_lines else "No scored results yet.",
        "Best passing molecules by Chemprop-predicted objective:",
        _format_examples(predicted_rows, ["smiles", "qed", "predicted_objective"]),
        "Best passing molecules by actual DOCKSTRING objective:",
        _format_examples(actual_rows, [
            "smiles", "qed", "docking_jak2", "docking_lck", "actual_objective",
        ]),
        "Structurally diverse passing examples:",
        _format_examples(diverse_rows, [
            "smiles", "qed", "predicted_objective", "actual_objective",
        ]),
        "Informative filter failures:",
        _format_examples(failure_rows, ["smiles", "proposal_status", "rejection_reason"]),
        "Iteration trends: " + _iteration_trends(history, passed),
    ])

    return "\n".join(sections)
