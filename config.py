"""Settings for the molecular design agent."""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
HISTORY_FILE = DATA_DIR / "experiment_history.csv"
DEMO_HISTORY_FILE = DATA_DIR / "demo_experiment_history.csv"

LLM_MODEL = os.getenv("LLM_MODEL", "qwen3")
NRP_BASE_URL = os.getenv("NRP_BASE_URL", "https://ellm.nrp-nautilus.io/v1")
NRP_API_KEY = os.getenv("NRP_API_KEY")

# Minimum QED eligibility gate; the paper objective also includes a continuous QED penalty.
MIN_QED = float(os.getenv("MIN_QED", "0.4"))

DEFAULT_CANDIDATES = 20
DEFAULT_DOCK_TOP = 2
