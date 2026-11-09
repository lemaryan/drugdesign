"""Ask the configured OpenAI-compatible API to propose candidate SMILES."""

import json
import re

from langchain_openai import ChatOpenAI

from config import LLM_MODEL, NRP_API_KEY, NRP_BASE_URL, TARGET_JAK2_OBJECTIVE


def make_llm():
    """Create an LLM client using the NRP OpenAI-compatible endpoint."""
    if not NRP_API_KEY:
        raise RuntimeError("NRP_API_KEY is missing. Export it before running the agent.")

    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=NRP_API_KEY,
        base_url=NRP_BASE_URL,
        temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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
You are proposing molecules for selective JAK2 binding. Return {number_to_propose}
chemically valid, single-fragment SMILES with varied scaffolds.

The program calculates the predicted objective for each molecule:
JAK2 - min(LCK + 8.1, 0) + 10 * (1 - QED)
Lower is better. Aim for a predicted objective near {TARGET_JAK2_OBJECTIVE:.1f}.
This is a stretch goal: the current best examples are closer to -8.4. With QED near
0.9, a score near -11 generally needs a JAK2 prediction near -12 and an LCK prediction
that is weaker than -8.1. Favor strong JAK2 binding, weaker LCK binding, and high QED.
Use the separate predicted and actual examples, diverse molecules, failure reasons,
and iteration trends as guidance. Try varied scaffolds and learn from failures, but
do not calculate scores or decide whether a molecule passes; the program does that.

Keep molecules within these limits where possible: MW <= 500, QED >= 0.4, LogP <= 5,
HBD <= 5, HBA <= 10, rotatable bonds <= 10, TPSA <= 140, and heavy atoms <= 40.
The program applies these filters, checks duplicates, predicts scores, and makes all
final ranking decisions. Return candidate SMILES only.

Candidate history:
{history_summary}

Return exactly a JSON array of SMILES strings, with no markdown or explanation.
Do not include names, properties, or extra keys.
"""
    response = llm.invoke(prompt)
    return _extract_json_list(response.content)
