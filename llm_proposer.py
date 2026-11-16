"""Ask the configured OpenAI-compatible API to propose candidate SMILES.

The proposer runs in two modes per iteration:

* exploit - generate close analogs of the best molecules found so far. This is the
  expected-improvement analogue: GP-BO with EI was the only method in the DOCKSTRING
  paper to beat the library best on selective JAK2, and it works by improving on the
  incumbent. Without this mode the loop takes independent draws instead of search
  steps, and the best-so-far curve stops moving.
* explore - generate genuinely new scaffolds, to keep adding chemotype families
  rather than polishing a single one.
"""

import json
import re
import time

from langchain_openai import ChatOpenAI
from rdkit import Chem, RDLogger

from config import (
    LIBRARY_BEST_OBJECTIVE,
    LLM_MODEL,
    LLM_REQUEST_ATTEMPTS,
    LLM_RETRY_SECONDS,
    MAX_PROPOSAL_RETRIES,
    MIN_VALID_FRACTION,
    NRP_API_KEY,
    NRP_BASE_URL,
    TARGET_JAK2_OBJECTIVE,
)

RDLogger.DisableLog("rdApp.*")


# Every SMILES below was checked with RDKit before being written into the prompt.
SMILES_RULES = """
SMILES syntax rules. Past batches lost up to 40% of proposals to parse errors, so
follow these exactly:
- Never write a group abbreviation as if it were an atom. CF3, NO2, SO2Me, OMe, Ph,
  Me, Et, tBu and Bn are NOT valid SMILES atoms.
  Write: CF3 as C(F)(F)F, NO2 as [N+](=O)[O-], SO2Me as S(=O)(=O)C, OMe as OC,
  Ph as c1ccccc1, Me as C, Et as CC, tBu as C(C)(C)C, Bn as Cc1ccccc1.
  Correct: FC(F)(F)c1ccccc1 and O=[N+]([O-])c1ccccc1
- Every ring-closure digit must appear exactly twice, and each aromatic ring must
  have the right number of atoms: 6 for a benzene or pyridine ring, 5 for a pyrrole,
  pyrazole, imidazole, furan or thiophene ring.
  A fused bicyclic shares exactly two atoms between its two rings.
- Count the atoms in each ring before closing it. This proposal failed because its
  five-membered ring held only four atoms:
    broken:   CC(C)Oc1ccc(Nc2ncnc3[nH]c(C)c23)cc1
    repaired: CC(C)Oc1ccc(Nc2ncnc3[nH]c(C)cc23)cc1
- Write an NH inside an aromatic ring as [nH], as in c1ccc2[nH]ncc2c1.
- Ring-closure labels above 9 need a percent sign: %10, %11. A bare "10" reads as
  ring 1 then ring 0. You will not need more than 4 open rings at once, so use the
  digits 1-4 and reuse them after a ring closes.
- Do not produce a series that differs only by an incrementing number. Each SMILES
  must be a chemically different molecule, not a relabelled copy of the last one.
- Return one connected molecule per string: no dots, no counter-ions, no salts.
- Anything under 16 heavy atoms is discarded as a fragment. Aim for roughly
  300-450 Da and 20-32 heavy atoms.
""".strip()


# Explore mode only. These say "copy this core verbatim", which is right when
# building a new scaffold but fights the "modify the parent" instruction in
# exploit mode, where it drove the model to echo parents back unchanged.
SCAFFOLD_TEMPLATES = """
Building fused ring systems:
- NEVER write a fused two-ring aromatic system from memory. This is the single
  largest source of rejected proposals: an invented fusion usually cannot be
  kekulized. All of these were rejected, and they are the pattern to avoid -
  note how each tries to build a fused core out of `nc2nc3...3n2`:
    rejected: COc1ccc(Nc2nc3[nH]c(N)nc3n2)cc1
    rejected: COc1ccc(Nc2nc3scnc3n2)cc1
    rejected: COc1ccc(Nc2nc3[nH]ccc2c3)cc1
    rejected: O=C(Nc1ccc(F)cc1)c1cnc2[nH]ccc12
- Instead, START from one of these COMPLETE verified molecules and change only its
  peripheral substituents. The fused core is already written and already numbered
  correctly, so you never have to renumber a ring closure:
    COc1ccc(Nc2ncnc3[nH]ccc23)cc1       pyrrolo[2,3-d]pyrimidin-4-amine
    COc1ccc(Nc2ncnc3[nH]ncc23)cc1       pyrazolo[3,4-d]pyrimidin-4-amine
    COc1ccc(Nc2ncnc3ccccc23)cc1         quinazolin-4-amine
    COc1ccc(Nc2ncnc3[nH]cnc23)cc1       purin-6-amine
    COc1ccc(Nc2ncnc3ccsc23)cc1          thieno[3,2-d]pyrimidin-4-amine
    COc1ccc(Nc2ncnc3ccoc23)cc1          furo[3,2-d]pyrimidin-4-amine
    O=C(Nc1ccc(F)cc1)c1ccc2[nH]ncc2c1   indazole-5-carboxamide
    O=C(Nc1ccc(F)cc1)c1ccc2[nH]ccc2c1   indole-5-carboxamide
    O=C(Nc1ccc(F)cc1)c1ccc2[nH]cnc2c1   benzimidazole-5-carboxamide
    O=C(Nc1ccc(F)cc1)c1cnc2[nH]ccc2c1   7-azaindole-5-carboxamide
    O=C(Nc1ccc(F)cc1)c1ccc2ncccc2c1     quinoline-6-carboxamide
    O=C(Nc1ccc(F)cc1)c1cnn2ccccc12      pyrazolo[1,5-a]pyrimidine-3-carboxamide
  Change the OCH3, the F, the aniline ring, or add substituents to the outer rings.
  Leave the `c2...c23` and `c2...c2c1` fused portions exactly as written.
- A single (unfused) ring you may write freely: benzene c1ccccc1, pyridine c1ccncc1,
  pyrimidine c1cncnc1, pyrazole c1cc[nH]n1, imidazole c1c[nH]cn1, thiophene c1ccsc1,
  piperazine C1CNCCN1, morpholine C1COCCN1, piperidine C1CCNCC1.
- Those templates are STARTING POINTS, not answers: returning one unchanged is a
  duplicate and scores nothing. Decorate it into a full molecule.
""".strip()


PROPERTY_LIMITS = """
Keep molecules within these limits: MW <= 500, QED >= 0.4, LogP <= 5, HBD <= 5,
HBA <= 10, rotatable bonds <= 10, TPSA <= 140, heavy atoms <= 40. Molecules outside
them are discarded before scoring, so a strong binder that breaks a limit is wasted.
""".strip()


def _objective_block():
    """Describe what the objective rewards, in mechanistic terms."""
    return f"""
The program scores each molecule with:
  objective = JAK2 - min(LCK + 8.1, 0) + 10 * (1 - QED)
Lower is better. Three levers, all of which matter:
- JAK2 docking score: more negative is better. This is the main term.
- LCK selectivity: you are penalised only while LCK binding is STRONGER than -8.1.
  Weakening LCK past -8.1 earns nothing more, so the aim is LCK just weaker than
  -8.1 while JAK2 stays strong. JAK2 and LCK scores correlate at 0.80 across the
  library, which is what makes this hard and where the remaining headroom is.
- QED: weighted 10, so each 0.1 of QED is worth a full 1.0 of objective. A QED of
  0.90 costs 1.0; a QED of 0.50 costs 5.0 and cannot be recovered by binding.
The best molecule in the DOCKSTRING library scores {LIBRARY_BEST_OBJECTIVE:.2f};
the target for this run is {TARGET_JAK2_OBJECTIVE:.1f}.
""".strip()


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


def _parseable(smiles):
    """Return whether RDKit can parse a SMILES string."""
    return Chem.MolFromSmiles(smiles) is not None


def _request(llm, prompt, log=None):
    """Send one prompt and return the SMILES list, or an empty list on failure.

    Every exception is caught, including API errors. A 500 from the endpoint used
    to propagate out of the propose node and kill the whole SLURM job: run
    41379608 died after 32 minutes and 6 iterations on a single
    `OpenAIAPIError: Error code: 500`. Losing one batch is an acceptable cost;
    losing an eight-hour allocation is not.
    """
    for attempt in range(1, LLM_REQUEST_ATTEMPTS + 1):
        try:
            return _extract_json_list(llm.invoke(prompt).content)
        except (ValueError, json.JSONDecodeError) as error:
            # The model replied, but not with a JSON list. Retrying may help.
            if log:
                log(f"  malformed reply ({type(error).__name__}), attempt {attempt}")
        except Exception as error:  # noqa: BLE001 - endpoint errors must not kill the run
            if log:
                log(
                    f"  LLM request failed ({type(error).__name__}: "
                    f"{str(error)[:120]}), attempt {attempt}"
                )
            if attempt < LLM_REQUEST_ATTEMPTS:
                time.sleep(LLM_RETRY_SECONDS * attempt)
    if log:
        log(f"  giving up on this request after {LLM_REQUEST_ATTEMPTS} attempts")
    return []


def _propose_with_retry(prompt, wanted, log=None):
    """Ask for SMILES, re-asking while too many of them fail to parse.

    Returns (smiles, stats). Invalid strings are still returned so the caller can
    record them: the rejection rows are what teach the next prompt.
    """
    llm = make_llm()
    collected = []
    attempts = 0
    invalid_seen = []

    current = prompt
    while attempts <= MAX_PROPOSAL_RETRIES:
        attempts += 1
        returned = _request(llm, current, log)
        if not returned:
            # A malformed or empty reply is usually transient. Retrying costs one
            # call; giving up costs the whole sub-batch for this iteration.
            if log:
                log(f"  proposal attempt {attempts}: no parseable JSON list, retrying")
            continue

        valid = [item for item in returned if _parseable(item)]
        invalid = [item for item in returned if not _parseable(item)]
        invalid_seen.extend(invalid)
        collected.extend(returned)

        fraction = len(valid) / len(returned) if returned else 0.0
        if log:
            log(
                f"  proposal attempt {attempts}: {len(returned)} returned, "
                f"{len(valid)} parseable ({fraction:.0%})"
            )
        if fraction >= MIN_VALID_FRACTION or attempts > MAX_PROPOSAL_RETRIES:
            break

        missing = max(wanted - len([s for s in collected if _parseable(s)]), 1)
        current = (
            f"{prompt}\n\n"
            f"Your previous reply had {len(invalid)} unparseable SMILES. RDKit "
            f"rejected these:\n" + "\n".join(invalid[:10]) + "\n\n"
            f"The usual causes are a group abbreviation used as an atom (CF3, NO2, "
            f"OMe) and a ring closed with the wrong number of atoms. Return "
            f"{missing} replacement SMILES that parse. Count the atoms in every "
            f"ring before closing it. JSON array only."
        )

    # Cap the parseable molecules at what was asked for, so batch size stays
    # reproducible, but keep every unparseable string: those rows are what teach
    # the next prompt its syntax errors.
    valid = [item for item in collected if _parseable(item)][:wanted]
    stats = {
        "attempts": attempts,
        "returned": len(collected),
        "invalid": len(invalid_seen),
    }
    return valid + invalid_seen, stats


ANALOG_EXAMPLES = """
Worked examples of the edit size wanted. Each pair is parent then analog, and every
one of these strings parses:
  Nc1ncnc2[nH]ccc12          ->  Nc1ncnc2[nH]c(Cl)cc12     add a chlorine
  Nc1ncnc2ccccc12            ->  Nc1ncnc2ccncc12           ring carbon to nitrogen
  O=C(Nc1ccccc1)c1ccccc1     ->  O=C(Nc1ccncc1)c1ccccc1    phenyl to pyridyl
  O=C(c1ccccc1)N1CCOCC1      ->  O=C(c1ccccc1)N1CCNCC1     morpholine to piperazine
  Cc1ccc(-c2ccccc2)cc1       ->  Cc1ccc(-c2ccc(F)cc2)cc1   fluorine on the far ring
Notice that the analog is always a DIFFERENT string from the parent, and always
recognisably the same chemotype.
""".strip()


EDIT_TYPES = [
    "swap or add one substituent on an aromatic ring (H to F, Cl, CH3, CN, OCH3 or OH)",
    "replace one aromatic ring carbon with nitrogen",
    "move an existing substituent to a different ring position",
    "swap one ring for a close relative (phenyl to pyridyl, morpholine to piperazine)",
    "add one small polar group to raise QED",
    "change a halogen for a different halogen",
    "replace a methyl with a cyclopropyl, or the reverse",
    "swap the amide direction, or change a ring size by one atom",
]


def _canonical(smiles):
    """Return a canonical SMILES, or None if the string does not parse."""
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule) if molecule is not None else None


def _analog_assignments(parent_count, wanted):
    """Assign each requested analog a specific parent and a specific edit.

    Naming the edit per slot makes echoing a parent structurally awkward: the model
    has to answer a concrete instruction rather than complete a list of molecules.
    """
    lines = []
    for index in range(wanted):
        parent = index % parent_count
        # Offset the edit by the pass number so a later pass over the same parent
        # asks for a different change instead of repeating the first request.
        edit = EDIT_TYPES[(index + index // parent_count) % len(EDIT_TYPES)]
        lines.append(f"  {index + 1}. from parent {parent + 1}: {edit}")
    return "\n".join(lines)


def propose_exploit(number_to_propose, hits, history_summary, log=None):
    """Generate close analogs of the molecules that already score best.

    `hits` is the DataFrame from memory.top_hits. Proposals that come back as an
    unchanged parent are dropped and re-requested: an echoed parent is always a
    duplicate downstream, so it would silently waste the batch.
    """
    if number_to_propose < 1 or hits is None or hits.empty:
        return [], {"attempts": 0, "returned": 0, "invalid": 0, "echoed": 0}

    from memory import format_top_hits  # local import avoids a circular import

    hits_block = format_top_hits(hits)
    parents = {
        canonical
        for canonical in (_canonical(item) for item in hits["smiles"].astype(str))
        if canonical
    }

    prompt = f"""
You are optimising molecules for selective JAK2 binding. Your job in this batch is
LEAD OPTIMISATION: return {number_to_propose} close analogs of the molecules listed
below. These are your own best results so far, measured by real docking.

{_objective_block()}

Your numbered parent molecules:
{hits_block}

Produce exactly these {number_to_propose} analogs, one per line of this work order.
Each one is a specific edit to a specific parent:
{_analog_assignments(len(parents), number_to_propose)}

Keep the core scaffold of the parent recognisable in each analog. You SHOULD stay
close to these structures: small, deliberate changes to a known-good molecule are
how the objective improves. But each answer must be the EDITED molecule, never the
parent as given.

{ANALOG_EXAMPLES}

{PROPERTY_LIMITS}

Context from earlier iterations:
{history_summary}

{SMILES_RULES}

Every molecule listed above has already been measured. None of them can score
again. Your answers must be structures that appear NOWHERE above.

CRITICAL: every SMILES you return must be a MODIFIED structure, not a copy. Copying
a parent from the list above, or any molecule from the history, scores nothing and
wastes the slot - those strings are already measured. Before you return each one,
check that it differs from its parent by at least one atom or bond.

Return exactly a JSON array of {number_to_propose} MODIFIED SMILES strings, with no
markdown and no explanation. No names, properties, or extra keys.
"""
    # An echoed parent is already measured, so it would be silently dropped as a
    # duplicate later. Filter echoes out and re-ask for the shortfall, because the
    # model does echo the list when asked for analogs of it.
    seen = set()
    kept = []
    invalid = []
    stats = {"attempts": 0, "returned": 0, "invalid": 0, "echoed": 0}
    current = prompt

    for _ in range(MAX_PROPOSAL_RETRIES + 1):
        wanted = number_to_propose - len(kept)
        if wanted < 1:
            break

        collected, round_stats = _propose_with_retry(current, wanted, log)
        stats["attempts"] += round_stats["attempts"]
        stats["returned"] += round_stats["returned"]
        stats["invalid"] += round_stats["invalid"]

        echoed = 0
        for item in collected:
            canonical = _canonical(item)
            if canonical is None:
                invalid.append(item)
                continue
            if canonical in parents:
                echoed += 1
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            kept.append(item)
        stats["echoed"] += echoed

        if echoed and log:
            log(f"  dropped {echoed} proposal(s) that echoed a parent unchanged")
        if len(kept) >= number_to_propose or not echoed:
            break

        missing = number_to_propose - len(kept)
        current = (
            f"{prompt}\n\n"
            f"Your previous reply returned {echoed} parent molecule(s) unchanged. "
            f"Those are already measured and score nothing. Return {missing} SMILES "
            f"that are each a MODIFIED version of a listed parent: change a "
            f"substituent, swap a ring atom, or replace a ring with a close relative. "
            f"Compare each string against the parent list before returning it - if it "
            f"matches one exactly, edit it again. JSON array only."
        )

    return kept[:number_to_propose] + invalid, stats


def propose_explore(number_to_propose, history_summary, log=None):
    """Generate new scaffolds, to add chemotype families rather than deepen one."""
    if number_to_propose < 1:
        return [], {"attempts": 0, "returned": 0, "invalid": 0}

    prompt = f"""
You are proposing molecules for selective JAK2 binding. Your job in this batch is
SCAFFOLD EXPLORATION: return {number_to_propose} molecules built on scaffolds that
do not appear in the history below.

{_objective_block()}

Aim for ATP-competitive kinase chemotypes with a hinge-binding donor-acceptor pair
and a drug-like size, around 300-450 Da. Use distinct scaffold classes across your
proposals, and do not return several variations of one core: a later batch handles
optimisation, this one is for coverage. Each proposal should be a different
chemotype from the others in this batch.

{PROPERTY_LIMITS}

Context from earlier iterations, including the structures already tried:
{history_summary}

{SMILES_RULES}

{SCAFFOLD_TEMPLATES}

Return exactly a JSON array of {number_to_propose} SMILES strings, with no markdown
and no explanation. No names, properties, or extra keys.
"""
    return _propose_with_retry(prompt, number_to_propose, log)


def propose_smiles(number_to_propose, history_summary="No previous results yet."):
    """Explore-only proposal. Kept for the earlier single-mode call signature."""
    smiles, _ = propose_explore(number_to_propose, history_summary)
    return smiles
