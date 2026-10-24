"""Filter and rank candidate molecules using simple, readable rules."""

from config import MAX_JAK2_SCORE, MIN_LCK_SCORE, MIN_QED


def evaluate_candidate(properties, predictions):
    """Combine RDKit properties and model predictions into one result."""
    jak2 = predictions["predicted_jak2"]
    lck = predictions["predicted_lck"]
    qed = properties["qed"]

    # Hard filters stop one very good metric from hiding a poor one.
    passes_filters = (
        jak2 <= MAX_JAK2_SCORE
        and lck >= MIN_LCK_SCORE
        and qed >= MIN_QED
    )

    # Simple score for ranking. The filters above remain separate.
    # Lower JAK2 is better, higher LCK is better, and higher QED is better.
    # These weights are starter choices, not scientifically validated weights.
    score = (-jak2 * 0.5) + (lck * 0.2) + (qed * 5.0)

    return {
        "predicted_jak2": jak2,
        "predicted_lck": lck,
        "qed": qed,
        "passes_filters": passes_filters,
        "combined_score": score,
    }
