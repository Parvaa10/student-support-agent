"""Long-term (cross-session) memory with a write-time policy guard.

Two layers of memory exist in this system:
  * Short-term  : the chat history of the current session (kept in the app, trimmed to the last N turns).
  * Long-term   : this SQLite store, keyed by student_id, which survives restarts and new sessions.

Writes are proposed by the LLM through the `save_memory` tool, but every write passes through
`check_memory_candidate()` — a deterministic guard that the model cannot override. It blocks
credentials, ID/bank/card numbers, contact details, health/emotional state and one-off temporary
situations, and only accepts a small set of useful categories.
"""
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from . import config
from .text_utils import tokenize

ALLOWED_CATEGORIES = {
    "academic_plan": "durable plans/goals, e.g. intends to switch specialisation, target electives",
    "preference": "how the student wants to be helped, e.g. answer format, language",
    "programme": "programme/specialisation facts not already in the student record",
    "communication": "preferred channel or time for follow-ups (not the actual phone/email)",
}

SENSITIVE_PATTERNS = [
    (r"\b(password|passcode|passwd|pin|otp|cvv|login credentials?)\b", "credentials / one-time codes"),
    (r"\b(aadhaa?r|pan\s*(card|number)?|passport|bank\s*account|ifsc|upi\s*id)\b", "government ID or bank details"),
    (r"\b(?:\d[ -]?){12,19}\b", "long ID / card / account number"),
    (r"[\w.+-]+@[\w-]+\.[\w.]+", "email address"),
    (r"(?:\+?91[-\s]?)?\b[6-9]\d{9}\b", "phone number"),
    (r"\b(depress\w*|anxi\w*|stress\w*|panic\w*|sick|illness|ill|hospital\w*|diagnos\w*|medical|medicine|therapy|"
     r"mental health|suicid\w*|self[- ]harm|pregnan\w*)\b", "health or emotional state"),
    (r"\b(caste|religio\w*|politic\w*|sexual\w*|salary|income|loan|debt)\b", "sensitive personal attribute / finances"),
]
TEMPORARY_PATTERNS = [
    r"\b(today|tonight|right now|at the moment|this (morning|afternoon|evening|week)|just now|tomorrow|yesterday)\b",
    r"\b(for now|temporar\w*|currently (stuck|waiting|in a))\b",
]


def check_memory_candidate(fact: str, category: str) -> tuple[bool, str]:
    """Return (allowed, reason). Deterministic: the LLM proposes, this function decides."""
    if category not in ALLOWED_CATEGORIES:
        return False, f"category '{category}' is not a storable category ({', '.join(ALLOWED_CATEGORIES)})"
    text = fact.lower()
    for pat, label in SENSITIVE_PATTERNS:
        if re.search(pat, text):
            return False, f"contains sensitive information ({label}); not stored"
    for pat in TEMPORARY_PATTERNS:
        if re.search(pat, text):
            return False, "describes a temporary / one-off situation; not useful as long-term memory"
    if len(fact.split()) < 3:
        return False, "too short to be a useful fact"
    if len(fact) > 300:
        return False, "too long; memories must be short facts"
    return True, "ok"


class MemoryStore:
    def __init__(self, db_path: Path = config.MEMORY_DB):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   student_id TEXT NOT NULL, category TEXT NOT NULL, fact TEXT NOT NULL,
                   created_at TEXT NOT NULL, session_id TEXT)"""
        )
        self.conn.commit()

    # ---------- write ----------
    def add(self, student_id: str, fact: str, category: str, session_id: str = "") -> dict:
        ok, reason = check_memory_candidate(fact, category)
        if not ok:
            return {"stored": False, "reason": reason}
        for m in self.list_all(student_id):
            if m["fact"].strip().lower() == fact.strip().lower():
                return {"stored": False, "reason": "already remembered", "memory_id": m["id"]}
        cur = self.conn.execute(
            "INSERT INTO memories (student_id, category, fact, created_at, session_id) VALUES (?,?,?,?,?)",
            (student_id, category, fact.strip(), datetime.now().isoformat(timespec="seconds"), session_id),
        )
        self.conn.commit()
        return {"stored": True, "memory_id": cur.lastrowid, "category": category}

    def delete(self, student_id: str, memory_id: int) -> None:
        self.conn.execute("DELETE FROM memories WHERE student_id=? AND id=?", (student_id, memory_id))
        self.conn.commit()

    def clear(self, student_id: str) -> None:
        self.conn.execute("DELETE FROM memories WHERE student_id=?", (student_id,))
        self.conn.commit()

    # ---------- read ----------
    def list_all(self, student_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, category, fact, created_at FROM memories WHERE student_id=? ORDER BY id", (student_id,)
        ).fetchall()
        return [{"id": r[0], "category": r[1], "fact": r[2], "created_at": r[3]} for r in rows]

    def select_for_context(self, student_id: str, query: str, k: int = config.MEMORY_TOP_K) -> list[dict]:
        """Context selection for memory: always include preferences (they shape every answer);
        include other facts only if they share terms with the current question (top-k)."""
        mems = self.list_all(student_id)
        prefs = [m for m in mems if m["category"] == "preference"]
        q = set(tokenize(query))
        scored = []
        for m in mems:
            if m["category"] == "preference":
                continue
            overlap = len(q & set(tokenize(m["fact"] + " " + m["category"].replace("_", " "))))
            if overlap:
                scored.append((overlap, m))
        scored.sort(key=lambda x: (-x[0], -x[1]["id"]))
        return prefs + [m for _, m in scored[:k]]
