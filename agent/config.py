"""Central configuration. All secrets come from environment variables / .env (never hard-coded)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
_tb = os.getenv("GEMINI_THINKING_BUDGET", "").strip()
GEMINI_THINKING_BUDGET = int(_tb) if _tb else None

DATA_DIR = ROOT / "data"
STORAGE_DIR = ROOT / "storage"
HANDBOOK_PATH = DATA_DIR / "student_handbook.md"
MEMORY_DB = Path(os.getenv("MEMORY_DB", STORAGE_DIR / "memory.db"))
TICKETS_FILE = Path(os.getenv("TICKETS_FILE", STORAGE_DIR / "tickets.json"))
TRACE_FILE = Path(os.getenv("TRACE_FILE", STORAGE_DIR / "traces.jsonl"))

# Retrieval / context-selection knobs
RETRIEVAL_TOP_K = 3          # max policy chunks returned per search
RETRIEVAL_MIN_SCORE = 0.10   # chunks below this cosine score are treated as "not relevant"
MEMORY_TOP_K = 3             # max non-preference memories injected per turn
MAX_HISTORY_TURNS = 4        # only the last N user/assistant exchanges are sent to the model
MAX_AGENT_STEPS = 5          # hard cap on LLM calls per question (prevents tool loops)
