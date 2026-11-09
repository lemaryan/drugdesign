"""Score candidate molecules with the DOCKSTRING selective-JAK2 objective."""

from config import TANIMOTO_DIVERSITY_WEIGHT
from molecule_tools import check_filters, tanimoto_similarity


LCK_SCORE_CUTOFF = 8.1
QED_PENALTY_WEIGHT = 10.0


def calculate_objective(jak2, lck, qed):
    """Calculate the selective-JAK2 objective; lower values are better."""
    return float(
        jak2 - min(lck + LCK_SCORE_CUTOFF, 0.0)
        + QED_PENALTY_WEIGHT * (1.0 - qed)
    )


def evaluate_candidate(properties, predictions):
    """Score a molecule with Chemprop predictions and apply hard filters."""
    jak2 = predictions["predicted_jak2"]
    lck = predictions["predicted_lck"]
    qed = properties["qed"]

    passes_filters, rejection_reason = check_filters(properties)

    return {
        "predicted_jak2": jak2,
        "predicted_lck": lck,
        "qed": qed,
        "passes_filters": passes_filters,
        "rejection_reason": rejection_reason,
        "predicted_objective": calculate_objective(jak2, lck, qed),
    }


def select_diverse_candidates(results, fingerprints, count, diversity_weight=None):
    """Greedily select low-objective, structurally diverse docking candidates.

    The predicted objective is min-max normalized within the eligible batch.
    At each selection step, the candidate with the lowest normalized objective
    plus a Tanimoto similarity penalty to the already-selected molecules wins.
    """
    weight = (
        TANIMOTO_DIVERSITY_WEIGHT if diversity_weight is None else diversity_weight
    )
    eligible = [row for row in results if row["passes_filters"]]
    if not eligible or count <= 0:
        return []

    objectives = [float(row["predicted_objective"]) for row in eligible]
    low, high = min(objectives), max(objectives)
    scale = high - low
    normalized = {
        id(row): ((float(row["predicted_objective"]) - low) / scale if scale else 0.0)
        for row in eligible
    }

    selected = []
    remaining = list(eligible)
    while remaining and len(selected) < count:
        scored = []
        for row in remaining:
            fingerprint = fingerprints.get(row["smiles"])
            max_similarity = 0.0
            if fingerprint is not None and selected:
                similarities = [
                    tanimoto_similarity(fingerprint, fingerprints[chosen["smiles"]])
                    for chosen in selected
                    if fingerprints.get(chosen["smiles"]) is not None
                ]
                max_similarity = max(similarities, default=0.0)
            combined = normalized[id(row)] + weight * max_similarity
            scored.append((combined, row, max_similarity))

        combined, winner, max_similarity = min(
            scored, key=lambda item: (item[0], item[1]["predicted_objective"])
        )
        winner["selection_objective_normalized"] = normalized[id(winner)]
        winner["max_tanimoto_to_selected"] = max_similarity
        winner["diversity_selection_score"] = combined
        selected.append(winner)
        remaining.remove(winner)

    return selected
