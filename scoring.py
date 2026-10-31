"""Score candidate molecules with the DOCKSTRING selective-JAK2 objective."""

from config import MIN_QED


LCK_SCORE_CUTOFF = -8.1
QED_PENALTY_WEIGHT = 10.0


def evaluate_candidate(properties, predictions):
    """Calculate the selective-JAK2 objective; lower values are better.

    This uses the DOCKSTRING selective-JAK2 objective with Chemprop predictions
    as estimates of the docking scores:
    JAK2 - min(LCK, -8.1) + 10 * (1 - QED).
    """
    jak2 = predictions["predicted_jak2"]
    lck = predictions["predicted_lck"]
    qed = properties["qed"]

    # The LCK term stops rewarding weaker LCK binding once its score reaches -8.1.
    # The QED penalty discourages poor drug-likeness while optimizing docking.
    objective = (
        jak2
        - min(lck, LCK_SCORE_CUTOFF)
        + QED_PENALTY_WEIGHT * (1.0 - qed)
    )

    # Keep the project's existing minimum QED gate as an extra drug-likeness check.
    passes_filters = qed >= MIN_QED

    return {
        "predicted_jak2": jak2,
        "predicted_lck": lck,
        "qed": qed,
        "passes_filters": passes_filters,
        "jak2_objective": float(objective),
    }
