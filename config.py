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
TARGET_JAK2_OBJECTIVE = -11.0
# Tradeoff for greedy diversity-aware selection (0 = objective only).
TANIMOTO_DIVERSITY_WEIGHT = float(os.getenv("TANIMOTO_DIVERSITY_WEIGHT", "0.25"))
