"""Main loop for the readable molecular design agent."""

import argparse

from config import DATA_DIR, DEFAULT_CANDIDATES, DEFAULT_DOCK_TOP, HISTORY_FILE
from docking import dock_molecule
from llm_proposer import propose_smiles
from memory import known_smiles, load_history, save_results, summarize_history
from molecule_tools import inspect_molecule
from scoring import evaluate_candidate
from surrogate_models import predict_many


# Example molecules are only for testing the software workflow, not drug candidates.
DEMO_SMILES = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CCN",
    "CCOC(=O)C",
]


def run_demo(iterations):
    """Test RDKit, filtering, ranking, and CSV saving without an LLM or docking."""
    from rdkit import Chem

    print("Running DEMO mode: no LLM, Chemprop, or DOCKSTRING calls.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, iterations + 1):
        results = []
        for smiles in DEMO_SMILES:
            properties = inspect_molecule(smiles)
            if properties is None:
                continue

            # Clearly fake scores so this demo cannot be mistaken for predictions.
            predictions = {"predicted_jak2": -7.5, "predicted_lck": -6.0}
            evaluation = evaluate_candidate(properties, predictions)

            results.append({
                "iteration": iteration,
                "smiles": properties["smiles"],
                "qed": properties["qed"],
                "predicted_jak2": predictions["predicted_jak2"],
                "predicted_lck": predictions["predicted_lck"],
                "combined_score": evaluation["combined_score"],
                "passes_filters": evaluation["passes_filters"],
                "docking_jak2": None,
                "docking_lck": None,
            })

        save_results(results, HISTORY_FILE)
        print(f"Demo iteration {iteration}: saved {len(results)} molecules to {HISTORY_FILE}")

    print("Demo complete. Demo predictions are placeholders, not model results.")


def run_agent(iterations, candidates_per_iteration, dock_top, run_docking):
    """Run proposal -> validation -> prediction -> ranking -> optional docking."""
    history = load_history(HISTORY_FILE)

    for iteration in range(1, iterations + 1):
        print(f"\n[Iteration {iteration}] Proposing molecules")

        feedback = summarize_history(history)
        proposed_smiles = propose_smiles(candidates_per_iteration, feedback)
        already_seen = known_smiles(history)

        candidates = []
        for smiles in proposed_smiles:
            properties = inspect_molecule(smiles)

            # Skip invalid molecules before spending time on model predictions.
            if properties is None:
                print(f"  Skipping invalid SMILES: {smiles}")
                continue

            canonical_smiles = properties["smiles"]
            if canonical_smiles in already_seen:
                print(f"  Skipping duplicate: {canonical_smiles}")
                continue

            candidates.append(properties)

        try:
            predictions_by_candidate = predict_many(
                [candidate["smiles"] for candidate in candidates]
            )
        except (FileNotFoundError, RuntimeError) as error:
            raise RuntimeError(
                "Chemprop prediction failed. Train the models and set "
                "JAK2_MODEL_PATH and LCK_MODEL_PATH to their checkpoint directories. "
                "Also confirm the Chemprop 2 command is available in this environment."
            ) from error

        results = []
        for properties, predictions in zip(candidates, predictions_by_candidate):
            canonical_smiles = properties["smiles"]

            evaluation = evaluate_candidate(properties, predictions)

            result = {
                "iteration": iteration,
                "smiles": canonical_smiles,
                "qed": properties["qed"],
                "predicted_jak2": evaluation["predicted_jak2"],
                "predicted_lck": evaluation["predicted_lck"],
                "combined_score": evaluation["combined_score"],
                "passes_filters": evaluation["passes_filters"],
                "docking_jak2": None,
                "docking_lck": None,
            }
            results.append(result)

        # Highest combined scores first; only molecules passing all filters
        # are eligible for docking.
        results.sort(key=lambda row: row["combined_score"], reverse=True)
        selected = [row for row in results if row["passes_filters"]][:dock_top]

        if run_docking:
            for result in selected:
                try:
                    docking_scores = dock_molecule(result["smiles"])
                except NotImplementedError as error:
                    raise RuntimeError(
                        "DOCKSTRING adapter is not configured. Edit docking.py first."
                    ) from error
                result.update(docking_scores)

        save_results(results, HISTORY_FILE)
        history = load_history(HISTORY_FILE)

        print(f"  Valid, new molecules: {len(results)}")
        print(f"  Passed filters: {sum(row['passes_filters'] for row in results)}")
        print(f"  Selected for docking: {len(selected)}")
        for result in results[:5]:
            print(
                f"  {result['smiles']} | "
                f"JAK2={result['predicted_jak2']:.2f} | "
                f"LCK={result['predicted_lck']:.2f} | "
                f"QED={result['qed']:.2f} | "
                f"score={result['combined_score']:.2f}"
            )

    print(f"\nFinished. Results saved to: {HISTORY_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Readable molecular design agent")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--dock-top", type=int, default=DEFAULT_DOCK_TOP)
    parser.add_argument("--demo", action="store_true", help="Run offline demo with placeholder predictions")
    parser.add_argument("--run-docking", action="store_true", help="Enable DOCKSTRING adapter")
    args = parser.parse_args()

    if args.iterations < 1 or args.candidates < 1 or args.dock_top < 1:
        parser.error("--iterations, --candidates, and --dock-top must be positive integers")

    if args.demo:
        run_demo(args.iterations)
    else:
        run_agent(args.iterations, args.candidates, args.dock_top, args.run_docking)


if __name__ == "__main__":
    main()
