"""Settings for the molecular design agent."""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
HISTORY_FILE = DATA_DIR / "experiment_history.csv"
SEED_MEMORY_FILE = DATA_DIR / "dockstring_seed_memory.csv"
DEMO_HISTORY_FILE = DATA_DIR / "demo_experiment_history.csv"

LLM_MODEL = os.getenv("LLM_MODEL", "qwen3")
NRP_BASE_URL = os.getenv("NRP_BASE_URL", "https://ellm.nrp-nautilus.io/v1")
NRP_API_KEY = os.getenv("NRP_API_KEY")

DEFAULT_CANDIDATES = 20
DEFAULT_DOCK_TOP = 2

# Best objective of any molecule in the 260k DOCKSTRING library, in this project's
# convention (scoring.calculate_objective). This is the reference line the DOCKSTRING
# paper plots in Figure 10 and the value a generative method has to beat.
# NOTE ON CONVENTIONS: the paper's Eq. 3 uses `jak2 - min(lck, -8.1) + 10(1 - QED)`,
# while scoring.calculate_objective uses `jak2 - min(lck + 8.1, 0) + 10(1 - QED)`.
# These differ by exactly +8.1 for every molecule, so the ranking and the argmin are
# identical, but the printed numbers are not comparable to the paper's. Add 8.1 to
# convert a value here into the paper's convention.
LIBRARY_BEST_OBJECTIVE = -9.40
LIBRARY_BEST_25TH_OBJECTIVE = -8.85
PAPER_CONVENTION_OFFSET = 8.1

# Stretch target used in the proposer prompt: beat the library best.
TARGET_JAK2_OBJECTIVE = -10.0

# Tradeoff for greedy diversity-aware selection (0 = objective only).
TANIMOTO_DIVERSITY_WEIGHT = float(os.getenv("TANIMOTO_DIVERSITY_WEIGHT", "0.25"))

# --- Proposal strategy -------------------------------------------------------
# Share of each batch spent on analogs of the current best molecules (the
# expected-improvement analogue). The remainder goes to novel scaffolds.
# Lowered from 0.7: by iteration 700 only 2.1% of exploit proposals survived to
# be scored (79% were duplicates) and the best objective had not moved in 450
# iterations. Analog work on the leading family is finished; the batch is worth
# more spent on chemotypes that have never been docked.
EXPLOIT_FRACTION = float(os.getenv("EXPLOIT_FRACTION", "0.4"))
# Lower size bound. A 9-12 heavy-atom fragment cannot be a selective kinase
# inhibitor, and docking one wastes the most expensive step in the loop. Every
# molecule in the run's top 15 has at least 20 heavy atoms, and this floor would
# have rejected nothing scoring better than -5.18.
MIN_HEAVY_ATOMS = int(os.getenv("MIN_HEAVY_ATOMS", "16"))
# How many scored molecules to show the proposer, and across how many distinct
# Bemis-Murcko scaffold families to spread them.
# Raised from 8: by ~470 dockings the analog space around the leading
# aminopyrimidinone family was mined out and most proposals came back as
# duplicates. Showing more parents, drawn from more distinct chemotypes, points
# the analog work at neighbourhoods that have not been explored yet.
TOP_HITS_SHOWN = int(os.getenv("TOP_HITS_SHOWN", "12"))
ANALOG_FAMILIES = int(os.getenv("ANALOG_FAMILIES", "6"))
# How many already-mined chemotypes to name in the explore prompt as an explicit
# avoid-list. Families here are generic (element-agnostic) Murcko scaffolds, not
# raw ones: see memory._chemotype for why the raw scaffold cannot measure this.
CHEMOTYPES_AVOIDED = int(os.getenv("CHEMOTYPES_AVOIDED", "6"))
# Reject an explore proposal whose generic chemotype has already been docked this
# many times. 0 disables the check and lets any chemotype through.
CHEMOTYPE_SATURATION = int(os.getenv("CHEMOTYPE_SATURATION", "25"))
# Recently tried SMILES shown to the proposer so it can avoid duplicates. Raised
# with the batch size, since a 40-candidate batch repeats more without it.
RECENT_SMILES_SHOWN = int(os.getenv("RECENT_SMILES_SHOWN", "60"))
# Attempts per individual LLM request, with a linear backoff between them. The
# NRP endpoint returns transient 500s, and an unhandled one killed run 41379608
# after 32 minutes, so every request is retried before the batch is abandoned.
LLM_REQUEST_ATTEMPTS = int(os.getenv("LLM_REQUEST_ATTEMPTS", "4"))
LLM_RETRY_SECONDS = float(os.getenv("LLM_RETRY_SECONDS", "10"))
# Extra LLM calls allowed per sub-batch when too many SMILES fail to parse.
MAX_PROPOSAL_RETRIES = int(os.getenv("MAX_PROPOSAL_RETRIES", "2"))
# Retry a sub-batch if fewer than this share of returned SMILES are parseable.
MIN_VALID_FRACTION = float(os.getenv("MIN_VALID_FRACTION", "0.75"))
