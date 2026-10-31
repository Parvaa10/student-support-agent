"""Assignment 2 - production controls: identity, authorization, human approval, audit, PII redaction.

Everything here is plain Python enforced OUTSIDE the LLM. The model can only *ask* for actions through
tools; these functions decide whether the action is allowed, needs approval, and record what happened.
"""
import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from . import config

AUDIT_FILE = Path(os.getenv("AUDIT_FILE", config.STORAGE_DIR / "audit.jsonl"))
APPROVALS_FILE = Path(os.getenv("APPROVALS_FILE", config.STORAGE_DIR / "approvals.json"))
CHANGES_FILE = Path(os.getenv("CHANGES_FILE", config.STORAGE_DIR / "record_changes.json"))

# Fields a student may REQUEST to change. Every change is sensitive -> always needs staff approval.
CHANGEABLE_FIELDS = {"phone", "email_on_file", "mailing_address", "specialisation", "emergency_contact"}


# ------------------------------------------------------------------ identity (demo sign-in)
def _h(pin: str) -> str:
    return hashlib.sha256(pin.encode()).hexdigest()


def verify_login(student_id: str, pin: str) -> bool:
    """Students sign in with a PIN. Only SHA-256 hashes are stored (data/auth.json)."""
    users = json.loads((config.DATA_DIR / "auth.json").read_text())
    ok = users.get(student_id) == _h(pin)
    audit(actor=student_id, role="student", action="login", target=student_id,
          decision="allowed" if ok else "denied", result="signed in" if ok else "wrong PIN")
    return ok


def verify_staff(pin: str) -> bool:
    return _h(pin) == json.loads((config.DATA_DIR / "auth.json").read_text()).get("_staff")


# ------------------------------------------------------------------ PII redaction (logs, traces, audit)
_REDACT = [
    (re.compile(r"(?i)\b(password|passcode|pwd|otp|pin|cvv)\b(\s*(is|=|:)?\s*)\S+"), r"\1\2[REDACTED]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"(?:\+?91[-\s]?)?\b[6-9]\d{9}\b"), "[PHONE]"),
    (re.compile(r"\b(?:\d[ -]?){12,19}\b"), "[NUMBER]"),
]


def redact(text):
    if not isinstance(text, str):
        return text
    for pat, rep in _REDACT:
        text = pat.sub(rep, text)
    return text


# ------------------------------------------------------------------ audit log (append-only JSONL)
def audit(actor: str, role: str, action: str, target: str, decision: str, result: str, details: dict | None = None):
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "audit_id": uuid.uuid4().hex[:8],
           "actor": actor, "role": role, "action": action, "target_student": target,
           "decision": decision, "result": redact(result),
           "details": {k: redact(v) for k, v in (details or {}).items()}}
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def read_audit(limit: int = 50) -> list[dict]:
    if not AUDIT_FILE.exists():
        return []
    return [json.loads(l) for l in AUDIT_FILE.read_text(encoding="utf-8").splitlines() if l.strip()][-limit:][::-1]


# ------------------------------------------------------------------ authorization (enforced in code)
def authorize_read(session_student: str, requested_student: str | None, action: str) -> tuple[bool, str]:
    """A student may read ONLY their own private record. Called by every data tool before any data is loaded."""
    target = requested_student or session_student
    if target != session_student:
        audit(session_student, "student", action, target, "denied",
              "blocked: students can only access their own private records")
        return False, "Access denied: you can only view your own records."
    audit(session_student, "student", action, target, "allowed", "own record returned")
    return True, "ok"


# ------------------------------------------------------------------ human approval workflow
def _load(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def _save(p: Path, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def create_change_request(session_student: str, target_student: str, field: str, new_value: str, reason: str) -> dict:
    if target_student != session_student:
        audit(session_student, "student", "request_record_change", target_student, "denied",
              "blocked: cannot change another student's record", {"field": field})
        return {"ok": False, "error": "Access denied: you can only request changes to your own record."}
    if field not in CHANGEABLE_FIELDS:
        audit(session_student, "student", "request_record_change", target_student, "denied",
              f"field '{field}' cannot be changed via support", {"field": field})
        return {"ok": False, "error": f"'{field}' cannot be changed through support. Changeable: {sorted(CHANGEABLE_FIELDS)}"}
    apr = {"approval_id": f"APR-{uuid.uuid4().hex[:6].upper()}", "student_id": session_student, "field": field,
           "new_value": new_value, "reason": reason, "status": "pending_approval",
           "requested_at": datetime.now().isoformat(timespec="seconds"), "decided_by": None, "decided_at": None}
    items = _load(APPROVALS_FILE, [])
    items.append(apr)
    _save(APPROVALS_FILE, items)
    audit(session_student, "student", "request_record_change", session_student, "pending_approval",
          "queued for staff approval - NOT executed", {"approval_id": apr["approval_id"], "field": field,
                                                       "new_value": new_value})
    return {"ok": True, "approval_id": apr["approval_id"], "status": "pending_approval",
            "message": "Change NOT applied yet. It will be executed only after a staff member approves it."}


def list_approvals(status: str | None = None, student_id: str | None = None) -> list[dict]:
    items = _load(APPROVALS_FILE, [])
    return [a for a in items if (not status or a["status"] == status) and (not student_id or a["student_id"] == student_id)]


def decide_approval(approval_id: str, approver: str, approve: bool, note: str = "") -> dict:
    """Staff decision. Approve -> execute the change. Reject -> nothing is executed."""
    items = _load(APPROVALS_FILE, [])
    apr = next((a for a in items if a["approval_id"] == approval_id), None)
    if not apr or apr["status"] != "pending_approval":
        return {"ok": False, "error": "approval not found or already decided"}
    apr.update(decided_by=approver, decided_at=datetime.now().isoformat(timespec="seconds"), note=note)
    if approve:
        changes = _load(CHANGES_FILE, {})
        changes.setdefault(apr["student_id"], {})[apr["field"]] = apr["new_value"]
        _save(CHANGES_FILE, changes)
        apr["status"] = "approved_executed"
        result = f"executed: {apr['field']} updated"
    else:
        apr["status"] = "rejected_not_executed"
        result = "rejected: change NOT executed"
    _save(APPROVALS_FILE, items)
    audit(approver, "staff", "decide_record_change", apr["student_id"],
          "approved" if approve else "rejected", result,
          {"approval_id": approval_id, "field": apr["field"], "new_value": apr["new_value"], "note": note})
    return {"ok": True, **apr}


def effective_record(student_id: str) -> dict | None:
    rec = json.loads((config.DATA_DIR / "students.json").read_text()).get(student_id)
    if rec is None:
        return None
    return {**rec, **_load(CHANGES_FILE, {}).get(student_id, {})}
