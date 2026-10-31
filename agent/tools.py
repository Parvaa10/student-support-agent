"""Tool layer: JSON-schema declarations (sent to Gemini) + Python implementations.

Design decisions
* Tools never take a `student_id` argument. The id comes from the authenticated session (ToolContext),
  so the model cannot be talked into reading another student's record.
* Every tool returns a dict. Failures are returned as {"ok": False, "error": ...} instead of raising,
  so the agent can read the error and explain it to the student (graceful error handling).
"""
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from . import config
from .memory import ALLOWED_CATEGORIES, MemoryStore
from .retrieval import PolicyRetriever
from . import security


class ToolError(Exception):
    pass


@dataclass
class ToolContext:
    student_id: str
    session_id: str
    memory: MemoryStore
    retriever: PolicyRetriever
    ticket_service_down: bool = False          # toggle to demo tool-failure handling
    # filled during a run so the tracer can report evidence
    retrieved_chunks: list = field(default_factory=list)
    memory_writes: list = field(default_factory=list)
    permission_decisions: list = field(default_factory=list)
    approvals_created: list = field(default_factory=list)


def _load(name: str) -> dict:
    return json.loads((config.DATA_DIR / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- implementations
def _check(ctx: ToolContext, student_id, action: str):
    """Authorization enforced in code: runs before any private data is loaded."""
    target = (student_id or ctx.student_id).strip().upper()
    ok, msg = security.authorize_read(ctx.student_id, target, action)
    ctx.permission_decisions.append({"action": action, "target": target, "decision": "allowed" if ok else "denied"})
    if not ok:
        raise ToolError(msg)
    return target


def get_student_profile(ctx: ToolContext, student_id: str = "") -> dict:
    sid = _check(ctx, student_id, "get_student_profile")
    rec = security.effective_record(sid)
    if not rec:
        raise ToolError(f"No student record found for {sid}.")
    return {k: v for k, v in rec.items() if k not in ("email_on_file", "phone")}   # data minimisation


def get_fee_status(ctx: ToolContext, student_id: str = "") -> dict:
    sid = _check(ctx, student_id, "get_fee_status")
    rec = _load("fees.json").get(sid)
    if not rec:
        raise ToolError("Fee record not found in the finance system for this student "
                        "(it may not be generated yet). Student should contact the Accounts office or raise a 'fees' ticket.")
    return rec


def get_scholarship_status(ctx: ToolContext, student_id: str = "") -> dict:
    sid = _check(ctx, student_id, "get_scholarship_status")
    recs = _load("scholarships.json").get(sid)
    if recs is None:
        raise ToolError("Scholarship system has no entry for this student.")
    return {"scholarships": recs, "count": len(recs)}


def request_record_change(ctx: ToolContext, field: str, new_value: str, reason: str = "", student_id: str = "") -> dict:
    """SENSITIVE ACTION: never executes directly - creates a pending approval for staff."""
    target = (student_id or ctx.student_id).strip().upper()
    res = security.create_change_request(ctx.student_id, target, field, new_value, reason)
    ctx.permission_decisions.append({"action": "request_record_change", "target": target,
                                     "decision": res.get("status", "denied")})
    if not res.get("ok"):
        raise ToolError(res["error"])
    ctx.approvals_created.append({"approval_id": res["approval_id"], "field": field, "status": res["status"]})
    return res


def check_change_requests(ctx: ToolContext) -> dict:
    items = security.list_approvals(student_id=ctx.student_id)
    return {"requests": [{k: a[k] for k in ("approval_id", "field", "status", "requested_at")} for a in items]}


def create_support_ticket(ctx: ToolContext, category: str, summary: str, priority: str = "normal") -> dict:
    if ctx.ticket_service_down:
        security.audit(ctx.student_id, "student", "create_support_ticket", ctx.student_id, "allowed",
                       "failed: ticketing service unavailable", {"category": category})
        raise ToolError("Ticketing service is temporarily unavailable (HTTP 503). Ticket was NOT created.")
    if not summary or len(summary.strip()) < 10:
        raise ToolError("Ticket summary is missing or too short; ask the student for details first.")
    tickets = json.loads(config.TICKETS_FILE.read_text()) if config.TICKETS_FILE.exists() else []
    ticket = {
        "ticket_id": f"TKT-{datetime.now():%y%m%d}-{uuid.uuid4().hex[:4].upper()}",
        "student_id": ctx.student_id, "category": category, "priority": priority,
        "summary": summary.strip(), "status": "open", "created_at": datetime.now().isoformat(timespec="seconds"),
        "sla": "1 working day" if priority == "high" else "2 working days",
    }
    tickets.append(ticket)
    security.audit(ctx.student_id, "student", "create_support_ticket", ctx.student_id, "allowed",
                   f"created {ticket['ticket_id']}", {"category": category, "priority": priority})
    config.TICKETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.TICKETS_FILE.write_text(json.dumps(tickets, indent=2))
    return ticket


def search_policy(ctx: ToolContext, query: str) -> dict:
    hits = ctx.retriever.search(query)
    ctx.retrieved_chunks.extend({"id": h["id"], "title": h["title"], "score": h["score"]} for h in hits)
    if not hits:
        return {"chunks": [], "note": "No handbook section is relevant to this query. Do not guess; say the "
                                      "handbook does not cover it and offer a support ticket."}
    return {"chunks": [{"id": h["id"], "title": h["title"], "score": h["score"], "text": h["text"]} for h in hits]}


def save_memory(ctx: ToolContext, fact: str, category: str) -> dict:
    res = ctx.memory.add(ctx.student_id, fact, category, ctx.session_id)
    ctx.memory_writes.append({"fact": fact, "category": category, **res})
    return res


def recall_memory(ctx: ToolContext, query: str = "") -> dict:
    mems = ctx.memory.list_all(ctx.student_id)
    return {"memories": [{"id": m["id"], "category": m["category"], "fact": m["fact"]} for m in mems]}


# ---------------------------------------------------------------- declarations (sent to the model)
TOOLS = {
    "get_student_profile": {
        "fn": get_student_profile,
        "description": "Get the logged-in student's academic record: programme, specialisation, current term, CGPA, "
                       "attendance and faculty advisor.",
        "parameters": {"type": "OBJECT", "properties": {"student_id": {"type": "STRING", "description": "Optional. Defaults to the signed-in student. Access to other students is denied."}}},
    },
    "get_fee_status": {
        "fn": get_fee_status,
        "description": "Get the logged-in student's current term fee status: total fee, amount paid, outstanding "
                       "balance, due date and last payment.",
        "parameters": {"type": "OBJECT", "properties": {"student_id": {"type": "STRING", "description": "Optional. Defaults to the signed-in student. Access to other students is denied."}}},
    },
    "get_scholarship_status": {
        "fn": get_scholarship_status,
        "description": "Get the status of the logged-in student's scholarship applications/awards.",
        "parameters": {"type": "OBJECT", "properties": {"student_id": {"type": "STRING", "description": "Optional. Defaults to the signed-in student. Access to other students is denied."}}},
    },
    "create_support_ticket": {
        "fn": create_support_ticket,
        "description": "Create a support ticket for the logged-in student when they ask to raise/log a request or "
                       "when an issue needs staff action. Returns the ticket id.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {"type": "STRING",
                             "enum": ["fees", "scholarship", "course_change", "it_support", "academic", "other"]},
                "summary": {"type": "STRING", "description": "One or two sentence description of the issue."},
                "priority": {"type": "STRING", "enum": ["normal", "high"],
                             "description": "high only for payment failures or examination issues"},
            },
            "required": ["category", "summary"],
        },
    },
    "search_policy": {
        "fn": search_policy,
        "description": "Search the official student handbook for rules, deadlines, procedures and eligibility "
                       "(fees policy, late fee, payment failure, refunds, scholarships, course/specialisation change, "
                       "attendance, exams, ID card, hostel). Returns only the most relevant sections with ids.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"query": {"type": "STRING", "description": "Focused search query, e.g. 'payment portal failure next working day'"}},
            "required": ["query"],
        },
    },
    "save_memory": {
        "fn": save_memory,
        "description": "Save a durable, useful fact about the student for future sessions (e.g. plans to switch "
                       "specialisation, preferred answer format). NEVER use for passwords/OTPs, ID or bank numbers, "
                       "contact details, health or emotional state, or temporary one-off situations. "
                       "The system may reject the write; report that honestly.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "fact": {"type": "STRING", "description": "Short third-person fact, e.g. 'Plans to switch to AI&DS specialisation after Term 3'."},
                "category": {"type": "STRING", "enum": list(ALLOWED_CATEGORIES)},
            },
            "required": ["fact", "category"],
        },
    },
    "request_record_change": {
        "fn": request_record_change,
        "description": "SENSITIVE ACTION. Request a change to the signed-in student's record (phone, email_on_file, "
                       "mailing_address, specialisation, emergency_contact). It is NEVER applied immediately: it creates "
                       "a pending approval that a staff member must approve. Tell the student the approval id.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "field": {"type": "STRING", "enum": ["phone", "email_on_file", "mailing_address", "specialisation", "emergency_contact"]},
                "new_value": {"type": "STRING"},
                "reason": {"type": "STRING"},
                "student_id": {"type": "STRING", "description": "Optional; defaults to the signed-in student."},
            },
            "required": ["field", "new_value"],
        },
    },
    "check_change_requests": {
        "fn": check_change_requests,
        "description": "List the signed-in student's record-change requests and their approval status.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    "recall_memory": {
        "fn": recall_memory,
        "description": "List everything remembered about the student from earlier sessions. Use only if the "
                       "student refers to something said earlier that is not already in REMEMBERED CONTEXT.",
        "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING"}}},
    },
}


def function_declarations() -> list[dict]:
    decls = []
    for name, spec in TOOLS.items():
        d = {"name": name, "description": spec["description"]}
        if spec["parameters"].get("properties"):
            d["parameters"] = spec["parameters"]
        decls.append(d)
    return decls


def execute(name: str, args: dict, ctx: ToolContext) -> tuple[dict, bool, float]:
    """Run a tool safely. Returns (result, ok, latency_ms)."""
    t0 = time.perf_counter()
    spec = TOOLS.get(name)
    try:
        if not spec:
            raise ToolError(f"Unknown tool '{name}'.")
        result, ok = spec["fn"](ctx, **(args or {})), True
    except ToolError as e:
        result, ok = {"ok": False, "error": str(e)}, False
    except TypeError as e:  # bad / missing arguments from the model
        result, ok = {"ok": False, "error": f"Invalid arguments for {name}: {e}"}, False
    except Exception as e:  # never crash the agent loop
        result, ok = {"ok": False, "error": f"Unexpected tool failure: {type(e).__name__}: {e}"}, False
    return result, ok, (time.perf_counter() - t0) * 1000
