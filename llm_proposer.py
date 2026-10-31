"""Ask the configured OpenAI-compatible API to propose candidate SMILES."""

import json
import re
import time

from langchain_openai import ChatOpenAI

from config import LLM_MODEL, NRP_API_KEY, NRP_BASE_URL


def make_llm():
    """Create an LLM client using the NRP OpenAI-compatible endpoint."""
    if not NRP_API_KEY:
        raise RuntimeError("NRP_API_KEY is missing. Export it before running the agent.")

    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=NRP_API_KEY,
        base_url=NRP_BASE_URL,
        temperature=0.7,
    )


def _extract_json_list(text):
    """Extract a JSON list from the model response."""
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError("The LLM response did not contain a JSON list of SMILES.")

    result = json.loads(match.group(0))
    if not isinstance(result, list):
        raise ValueError("Expected a JSON list.")
    return [item.strip() for item in result if isinstance(item, str) and item.strip()]


def propose_smiles(number_to_propose, history_summary="No previous results yet."):
    """Generate a batch of SMILES strings. Validation happens in molecule_tools.py."""
    llm = make_llm()

    prompt = f"""
You are helping a medicinal chemistry team propose small molecules for selective JAK2
binding. Suggest {number_to_propose} novel, chemically plausible candidate molecules.

Optimization goals:
- Minimize the DOCKSTRING selective-JAK2 objective, using the following predicted
  scores as estimates of Vina scores in kcal/mol:
  JAK2 score - min(LCK score, -8.1) + 10 * (1 - QED).
- A lower, more negative Vina score means stronger predicted binding. The objective
  therefore favors stronger JAK2 binding, weaker LCK binding, and higher QED.
- Keep molecules drug-like. Favor balanced, compact structures. Avoid increasing
  molecular size or lipophilicity just to improve a docking score; large, greasy
  molecules can score well while having poor drug-like properties.
- As practical guidelines, favor molecules near common oral small-molecule ranges:
  molecular weight <= 500 Da, logP <= 5, hydrogen-bond donors <= 5, hydrogen-bond
  acceptors <= 10, and rotatable bonds <= 10. These are heuristics, not proof of
  developability. The project also filters candidates at QED >= 0.4; prefer higher QED
  when it does not undermine the JAK2-versus-LCK objective.
- Favor chemically valid, single-fragment molecules without radicals. Provide varied
  scaffolds and avoid close repeats of known SMILES.

How to use the history:
- `predicted_jak2` and `predicted_lck` are Chemprop estimates of docking scores.
- `docking_jak2` and `docking_lck`, when present, are actual DOCKSTRING/Vina scores.
- Lower `jak2_objective` values are better. Use strong JAK2 scores, weaker LCK scores,
  and good QED together; do not optimize docking scores alone.

Candidate history:
{history_summary}

Return exactly a JSON array of SMILES strings, with no markdown or explanation.
Do not include names, properties, or extra keys.
"""
    started = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - started
    print(f"  LLM proposal request completed in {elapsed:.1f}s")
    return _extract_json_list(response.content)



