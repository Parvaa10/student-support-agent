"""The Student Support Agent: context builder + ReAct-style tool loop + tracing.

Per user question:
 1. CONTEXT SELECTION  - build a small system prompt: rules + logged-in student id + only the memories
                         relevant to this question (+ preferences) + last N chat turns. No handbook text,
                         no full records are pre-loaded; the model must fetch them with tools if needed.
 2. AGENT LOOP         - call Gemini with tool declarations. If it returns function calls, execute them,
                         append the results, and call the model again. Stop when it returns plain text
                         (or after MAX_AGENT_STEPS as a safety cap).
 3. TRACE              - every LLM call (tokens, latency) and tool call (args, ok/error, latency),
                         the retrieved policy chunks and memory items are written to storage/traces.jsonl.
"""
import json
import time
import uuid
from datetime import date, datetime

from google.genai import types

from . import config
from .llm import GeminiLLM, usage
from .memory import MemoryStore
from .retrieval import PolicyRetriever
from .tools import ToolContext, execute
from . import security

SYSTEM_RULES = """You are "Campus Assist", the student support agent for the PGDM programme
(Big Data Analytics / AI & Data Science). You help the logged-in student with fees, scholarships,
course/specialisation changes, policies and support requests.

How to decide what to do (think before acting):
1. Greetings, thanks, or questions about what you can do -> answer directly, NO tools.
2. The student's own data (fee status, dues, scholarship status, CGPA, specialisation, attendance)
   -> call the matching tool. Never guess or invent numbers, dates or statuses.
3. Rules, deadlines, eligibility, procedures, penalties -> call search_policy with a focused query and
   answer ONLY from the returned chunks, citing their ids like [POL-2.3]. If the chunks do not
   explicitly cover the question, say the handbook does not cover it, do not guess, and offer to raise a ticket.
4. Some questions need both personal data and policy (e.g. "am I eligible for the merit scholarship?")
   -> call both tools, then combine.
5. Raise a support ticket (create_support_ticket) only when the student asks for one or agrees to one.
   Ask for missing details first if the issue description is unclear. Always give the ticket id.
6. Memory: when the student shares a durable, useful fact (future academic plans, preferred answer style,
   preferred follow-up channel) call save_memory with a short third-person fact. Do NOT save passwords,
   OTPs, ID/bank/card numbers, phone/email, health or emotional state, or one-off temporary situations
   (e.g. "my laptop is broken today"). If the student shares such information, do not repeat it back,
   and tell them briefly that you have not stored it. If save_memory reports stored=false, say so honestly.
7. If a tool returns an error, explain it plainly, never fabricate the missing data, and suggest a next step.
8. You can only access the logged-in student's own records. If a student asks about another student, pass that
   student_id to the tool: the system will deny it and log the attempt. Never reveal another student's data.
10. Changes to the student's record (phone, email, address, specialisation, emergency contact) are SENSITIVE:
   use request_record_change. It is never applied immediately; tell the student the approval id and that a
   staff member must approve it first. Never claim a change has been made.
9. If required information is missing from the question, ask one short clarifying question.
Keep answers short and clear (INR amounts with commas). Follow the student's saved preferences."""


def build_system_prompt(student_id: str, selected_memories: list[dict]) -> str:
    if selected_memories:
        mem_lines = "\n".join(f"- [MEM-{m['id']}] ({m['category']}) {m['fact']}" for m in selected_memories)
    else:
        mem_lines = "- (nothing relevant remembered for this question)"
    return (f"{SYSTEM_RULES}\n\n### SESSION CONTEXT\nLogged-in student id: {student_id}\n"
            f"Today's date: {date.today():%A, %d %B %Y}\n\n"
            f"### REMEMBERED CONTEXT (selected for this question from earlier sessions)\n{mem_lines}")


def _text_of(resp) -> str:
    try:
        parts = resp.candidates[0].content.parts or []
    except (AttributeError, IndexError, TypeError):
        return ""
    return "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)).strip()


class StudentSupportAgent:
    def __init__(self, llm=None, memory: MemoryStore | None = None, retriever: PolicyRetriever | None = None):
        self.llm = llm or GeminiLLM()
        self.memory = memory or MemoryStore()
        self.retriever = retriever or PolicyRetriever()

    def answer(self, question: str, student_id: str, session_id: str, history: list[dict] | None = None,
               ticket_service_down: bool = False, write_trace: bool = True) -> dict:
        t_start = time.perf_counter()
        history = history or []
        ctx = ToolContext(student_id=student_id, session_id=session_id, memory=self.memory,
                          retriever=self.retriever, ticket_service_down=ticket_service_down)

        # ---- 1. context selection
        selected = self.memory.select_for_context(student_id, question)
        system_prompt = build_system_prompt(student_id, selected)
        contents = []
        for turn in history[-config.MAX_HISTORY_TURNS * 2:]:
            role = "user" if turn["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=turn["content"])]))
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

        # ---- 2. agent loop
        steps, answer, status = [], "", "ok"
        tok = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}
        for step in range(1, config.MAX_AGENT_STEPS + 1):
            t0 = time.perf_counter()
            try:
                resp = self.llm.generate(contents, system_prompt)
            except Exception as e:
                status = "llm_error"
                answer = ("Sorry, I couldn't reach the language model service just now "
                          f"({type(e).__name__}). Please try again in a minute.")
                steps.append({"type": "llm", "step": step, "error": str(e)[:300]})
                break
            u = usage(resp)
            for k in tok:
                tok[k] += u[k]
            calls = list(resp.function_calls or [])
            steps.append({"type": "llm", "step": step, "latency_ms": round((time.perf_counter() - t0) * 1000),
                          **u, "decision": "tool_call" if calls else "final_answer",
                          "function_calls": [c.name for c in calls]})
            if not calls:
                answer = _text_of(resp) or "Sorry, I could not produce an answer. Could you rephrase?"
                break
            contents.append(resp.candidates[0].content)          # keep model turn (incl. thought signatures)
            response_parts = []
            for call in calls:
                args = dict(call.args or {})
                result, ok, ms = execute(call.name, args, ctx)
                steps.append({"type": "tool", "step": step, "tool": call.name, "args": args, "ok": ok,
                              "latency_ms": round(ms, 1), "result_preview": security.redact(json.dumps(result, default=str)[:400])})
                response_parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=response_parts))
        else:
            status = "max_steps"
            answer = "I wasn't able to complete this request. I can raise a support ticket for a staff member to help."

        tools_used = [s["tool"] for s in steps if s["type"] == "tool"]
        trace = {
            "run_id": uuid.uuid4().hex[:10],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id, "student_id": student_id,
            "model": getattr(self.llm, "model", "unknown"),
            "question": security.redact(question), "answer": security.redact(answer), "status": status,
            "latency_ms": round((time.perf_counter() - t_start) * 1000),
            "llm_calls": sum(1 for s in steps if s["type"] == "llm"),
            "tools_used": tools_used,
            "tool_errors": [s["tool"] for s in steps if s["type"] == "tool" and not s["ok"]],
            **tok,
            "history_turns_sent": len(history[-config.MAX_HISTORY_TURNS * 2:]),
            "memory_used": [{"id": f"MEM-{m['id']}", "category": m["category"], "fact": m["fact"]} for m in selected],
            "memory_writes": [{**w, "fact": security.redact(w["fact"])} for w in ctx.memory_writes],
            "permission_decisions": ctx.permission_decisions,
            "approvals_created": ctx.approvals_created,
            "policy_chunks": ctx.retrieved_chunks,
            "steps": steps,
        }
        for w in ctx.memory_writes:   # audit memory writes (fact text redacted)
            security.audit(student_id, "student", "save_memory", student_id,
                           "allowed" if w.get("stored") else "denied",
                           "stored" if w.get("stored") else f"not stored: {w.get('reason')}", {"fact": w["fact"]})
        if write_trace:
            config.TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(config.TRACE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace, default=str) + "\n")
        return trace
