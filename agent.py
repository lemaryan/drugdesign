"""Main loop for the readable molecular design agent."""

import argparse
from datetime import datetime
from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from config import (
    DATA_DIR,
    DEFAULT_CANDIDATES,
    DEFAULT_DOCK_TOP,
    DEMO_HISTORY_FILE,
    HISTORY_FILE,
    SEED_MEMORY_FILE,
)
from docking import DockingError, dock_molecule
from llm_proposer import propose_smiles
from memory import known_smiles, load_history, save_results, summarize_history
from molecule_tools import check_filters, inspect_molecule
from scoring import calculate_objective, evaluate_candidate, select_diverse_candidates
from surrogate_models import predict_many


# Example molecules are only for testing the software workflow, not drug candidates.
DEMO_SMILES = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CCN",
    "CCOC(=O)C",
]


class AgentState(TypedDict, total=False):
    iteration: int
    iterations: int
    candidates_per_iteration: int
    dock_top: int
    run_docking: bool
    run_start: float
    iteration_start: float
    chemprop_time: float
    docking_time: float
    history: Any
    seed_memory: Any
    proposed_smiles: list[str]
    candidates: list[dict]
    proposal_records: list[dict]
    predictions: list[dict]
    results: list[dict]
    selected: list[dict]


def log(message):
    """Print a message with the current local time."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {message}", flush=True)


def run_demo(iterations):
    """Test RDKit, filtering, ranking, and CSV saving without an LLM or docking."""
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
                "predicted_objective": evaluation["predicted_objective"],
                "actual_objective": None,
                "passes_filters": evaluation["passes_filters"],
                "docking_jak2": None,
                "docking_lck": None,
            })

        save_results(results, DEMO_HISTORY_FILE)
        print(
            f"Demo iteration {iteration}: saved {len(results)} molecules "
            f"to {DEMO_HISTORY_FILE}"
        )

    print("Demo complete. Demo predictions are placeholders, not model results.")


def propose_node(state: AgentState):
    """Ask the LLM for this iteration's candidates."""
    iteration = state["iteration"]
    iteration_start = perf_counter()
    log(f"Iteration {iteration}/{state['iterations']}: starting")
    feedback = summarize_history(state["history"], state["seed_memory"])
    llm_start = perf_counter()
    log(
        f"Iteration {iteration}: asking LLM for "
        f"{state['candidates_per_iteration']} molecules"
    )
    proposed = propose_smiles(state["candidates_per_iteration"], feedback)
    log(
        f"Iteration {iteration}: LLM returned {len(proposed)} molecules in "
        f"{perf_counter() - llm_start:.2f}s"
    )
    return {"proposed_smiles": proposed, "iteration_start": iteration_start}


def validate_node(state: AgentState):
    """Remove invalid, duplicate, and filtered proposals before prediction."""
    started = perf_counter()
    iteration = state["iteration"]
    already_seen = known_smiles(state["history"])
    seed_smiles = known_smiles(state["seed_memory"])
    candidates = []
    records = []

    for smiles in state["proposed_smiles"]:
        properties = inspect_molecule(smiles)
        if properties is None:
            log(f"Skipping invalid SMILES: {smiles}")
            records.append({
                "iteration": iteration,
                "smiles": smiles,
                "passes_filters": False,
                "proposal_status": "invalid",
                "rejection_reason": "RDKit could not parse SMILES",
            })
            continue

        canonical_smiles = properties["smiles"]
        if canonical_smiles in seed_smiles:
            log(f"Skipping Dockstring dataset example: {canonical_smiles}")
            records.append({
                "iteration": iteration,
                "smiles": canonical_smiles,
                "qed": properties["qed"],
                "passes_filters": False,
                "proposal_status": "dataset_duplicate",
                "rejection_reason": "Already present in the Dockstring seed examples",
            })
            continue
        if canonical_smiles in already_seen:
            log(f"Skipping duplicate: {canonical_smiles}")
            records.append({
                "iteration": iteration,
                "smiles": canonical_smiles,
                "qed": properties["qed"],
                "passes_filters": False,
                "proposal_status": "duplicate",
                "rejection_reason": "Canonical SMILES already appears in history",
            })
            continue

        passes, reason = check_filters(properties)
        if not passes:
            log(f"Skipping filtered molecule ({reason}): {canonical_smiles}")
            records.append({
                "iteration": iteration,
                "smiles": canonical_smiles,
                "qed": properties["qed"],
                "passes_filters": False,
                "proposal_status": "filtered",
                "rejection_reason": reason,
            })
            already_seen.add(canonical_smiles)
            continue

        candidates.append(properties)
        already_seen.add(canonical_smiles)

    log(
        f"Iteration {iteration}: validation finished in {perf_counter() - started:.2f}s; "
        f"{len(candidates)} valid, new molecules"
    )
    return {"candidates": candidates, "proposal_records": records}


def predict_node(state: AgentState):
    """Run both Chemprop models for the valid candidates."""
    started = perf_counter()
    iteration = state["iteration"]
    log(
        f"Iteration {iteration}: starting Chemprop predictions for "
        f"{len(state['candidates'])} molecules"
    )
    try:
        predictions = predict_many([item["smiles"] for item in state["candidates"]])
    except (FileNotFoundError, RuntimeError) as error:
        raise RuntimeError(
            "Chemprop prediction failed. Train the models and set "
            "JAK2_MODEL_PATH and LCK_MODEL_PATH to their checkpoint directories. "
            "Also confirm the Chemprop 2 command is available in this environment."
        ) from error

    elapsed = perf_counter() - started
    log(f"Iteration {iteration}: Chemprop predictions finished in {elapsed:.2f}s")
    return {
        "predictions": predictions,
        "chemprop_time": state["chemprop_time"] + elapsed,
    }


def score_node(state: AgentState):
    """Calculate objectives and choose diverse candidates for docking."""
    started = perf_counter()
    results = []
    for properties, predictions in zip(state["candidates"], state["predictions"]):
        evaluation = evaluate_candidate(properties, predictions)
        results.append({
            "iteration": state["iteration"],
            "smiles": properties["smiles"],
            "qed": properties["qed"],
            "predicted_jak2": evaluation["predicted_jak2"],
            "predicted_lck": evaluation["predicted_lck"],
            "predicted_objective": evaluation["predicted_objective"],
            "actual_objective": None,
            "passes_filters": evaluation["passes_filters"],
            "proposal_status": "scored",
            "rejection_reason": None,
            "docking_jak2": None,
            "docking_lck": None,
        })

    fingerprints = {
        item["smiles"]: item["fingerprint"] for item in state["candidates"]
    }
    results.sort(key=lambda row: row["predicted_objective"])
    selected = select_diverse_candidates(results, fingerprints, state["dock_top"])
    log(
        f"Iteration {state['iteration']}: scoring and selection finished in "
        f"{perf_counter() - started:.2f}s; selected {len(selected)}"
    )
    return {"results": results, "selected": selected}


def dock_node(state: AgentState):
    """Dock selected candidates and calculate objectives from actual scores."""
    started = perf_counter()
    results = [dict(row) for row in state["results"]]
    by_smiles = {row["smiles"]: row for row in results}
    selected = [by_smiles[row["smiles"]] for row in state["selected"]]

    for result in selected:
        candidate_started = perf_counter()
        log(f"Iteration {state['iteration']}: docking {result['smiles']}")
        try:
            scores = dock_molecule(result["smiles"])
        except NotImplementedError as error:
            raise RuntimeError(
                "DOCKSTRING adapter is not configured. Edit docking.py first."
            ) from error
        except DockingError as error:
            elapsed = perf_counter() - candidate_started
            log(f"Docking failed after {elapsed:.2f}s: {error}")
            continue

        result.update(scores)
        result["actual_objective"] = calculate_objective(
            scores["docking_jak2"], scores["docking_lck"], result["qed"]
        )
        elapsed = perf_counter() - candidate_started
        log(f"Docking finished in {elapsed:.2f}s: {scores}")

    elapsed = perf_counter() - started
    log(f"Iteration {state['iteration']}: all docking finished in {elapsed:.2f}s")
    return {
        "results": results,
        "selected": selected,
        "docking_time": state["docking_time"] + elapsed,
    }


def save_node(state: AgentState):
    """Append this iteration to memory and prepare the next one."""
    started = perf_counter()
    save_results(state["proposal_records"] + state["results"], HISTORY_FILE)
    history = load_history(HISTORY_FILE)
    iteration = state["iteration"]
    log(f"Iteration {iteration}: results saved in {perf_counter() - started:.2f}s")
    log(f"Iteration {iteration}: valid, new molecules: {len(state['results'])}")
    log(
        f"Iteration {iteration}: passed filters: "
        f"{sum(row['passes_filters'] for row in state['results'])}"
    )
    log(f"Iteration {iteration}: selected for docking: {len(state['selected'])}")

    for result in state["selected"]:
        log(
            f"Docking selection: {result['smiles']} | "
            f"predicted objective={result['predicted_objective']:.2f} | "
            f"max Tanimoto to earlier selection="
            f"{result['max_tanimoto_to_selected']:.2f} | "
            f"combined selection score={result['diversity_selection_score']:.3f}"
        )
    for result in state["results"][:5]:
        log(
            f"{result['smiles']} | JAK2={result['predicted_jak2']:.2f} | "
            f"LCK={result['predicted_lck']:.2f} | QED={result['qed']:.2f} | "
            f"predicted objective={result['predicted_objective']:.2f}"
        )
    elapsed = perf_counter() - state["iteration_start"]
    log(f"Iteration {iteration} finished in {elapsed:.2f}s")
    return {"history": history, "iteration": iteration + 1}


def finish_node(state: AgentState):
    """Print totals after the last graph iteration."""
    log(
        f"Run finished in {perf_counter() - state['run_start']:.2f}s. "
        f"Total Chemprop time: {state['chemprop_time']:.2f}s. "
        f"Total DOCKSTRING time: {state['docking_time']:.2f}s. "
        f"Results saved to: {HISTORY_FILE}"
    )
    return {}


def _after_scoring(state: AgentState):
    return "dock" if state["run_docking"] else "save"


def _after_iteration(state: AgentState):
    return "continue" if state["iteration"] <= state["iterations"] else "finish"


def build_agent_graph():
    """Build the proposal and scoring workflow as a LangGraph state graph."""
    graph = StateGraph(AgentState)
    graph.add_node("propose", propose_node)
    graph.add_node("validate", validate_node)
    graph.add_node("predict", predict_node)
    graph.add_node("score", score_node)
    graph.add_node("dock", dock_node)
    graph.add_node("save", save_node)
    graph.add_node("finish", finish_node)

    graph.add_edge(START, "propose")
    graph.add_edge("propose", "validate")
    graph.add_edge("validate", "predict")
    graph.add_edge("predict", "score")
    graph.add_conditional_edges(
        "score", _after_scoring, {"dock": "dock", "save": "save"}
    )
    graph.add_edge("dock", "save")
    graph.add_conditional_edges(
        "save", _after_iteration, {"continue": "propose", "finish": "finish"}
    )
    graph.add_edge("finish", END)
    return graph.compile()


def run_agent(iterations, candidates_per_iteration, dock_top, run_docking):
    """Run the molecule workflow with LangGraph."""
    graph = build_agent_graph()
    initial_state = {
        "iteration": 1,
        "iterations": iterations,
        "candidates_per_iteration": candidates_per_iteration,
        "dock_top": dock_top,
        "run_docking": run_docking,
        "run_start": perf_counter(),
        "chemprop_time": 0.0,
        "docking_time": 0.0,
        "history": load_history(HISTORY_FILE),
        "seed_memory": load_history(SEED_MEMORY_FILE),
    }
    graph.invoke(initial_state, {"recursion_limit": iterations * 7 + 10})


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
