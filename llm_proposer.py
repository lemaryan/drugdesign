"""Ask the configured OpenAI-compatible API to propose candidate SMILES."""

import json
import re

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
You are assisting a research team studying molecular docking selectivity.
Suggest {number_to_propose} chemically plausible candidate molecules as SMILES strings.

Research goals:
- JAK2 docking score: more negative is better.
- LCK docking score: less negative / higher is better for selectivity.
- QED: higher is better.
- Avoid repeating known SMILES.
- Do not claim that a candidate is a real drug or that its properties are experimentally proven.

Use this history as feedback:
{history_summary}

Return ONLY a JSON array of SMILES strings. No markdown and no explanations.
"""
    response = llm.invoke(prompt)
    return _extract_json_list(response.content)
