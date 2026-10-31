"""Streamlit chat UI for the Student Support Agent.

Run:  streamlit run app.py
Left sidebar = "login" (pick a demo student), session controls, long-term memory viewer.
Each assistant message has an expandable trace: tools, policy chunks, memory used, tokens, latency.
"""
import json
import uuid

import pandas as pd
import streamlit as st

from agent import config
from agent.agent import StudentSupportAgent
from agent import security

st.set_page_config(page_title="Campus Assist — Student Support Agent", page_icon="🎓", layout="wide")


@st.cache_resource
def get_agent():
    return StudentSupportAgent()


students = json.loads((config.DATA_DIR / "students.json").read_text())

# ---------------- sidebar
with st.sidebar:
    st.header("🎓 Campus Assist")
    # ---- sign-in (identity): PIN checked against SHA-256 hashes in data/auth.json
    if "auth_student" not in st.session_state:
        with st.form("login"):
            st.subheader("🔐 Student sign-in")
            lid = st.selectbox("Student ID", list(students), format_func=lambda s: f"{s} — {students[s]['name']}")
            pin = st.text_input("PIN", type="password", help="Demo PINs: 1001 / 1002 / 1003")
            if st.form_submit_button("Sign in"):
                if security.verify_login(lid, pin):
                    st.session_state.update(auth_student=lid, student_id=lid,
                                            session_id=uuid.uuid4().hex[:8], messages=[])
                    st.rerun()
                else:
                    st.error("Wrong PIN")
        st.stop()
    sid = st.session_state.auth_student
    st.success(f"Signed in as **{sid} — {students[sid]['name']}**")
    if st.button("Sign out"):
        for k in ("auth_student", "student_id", "messages"):
            st.session_state.pop(k, None)
        st.rerun()
    st.caption(f"Session id: `{st.session_state.session_id}` · model: `{config.GEMINI_MODEL}`")
    if st.button("🔄 Start new session",
                 help="Clears the chat (short-term memory). Long-term memory is kept."):
        st.session_state.update(session_id=uuid.uuid4().hex[:8], messages=[])
        st.rerun()
    ticket_down = st.toggle("Simulate ticketing service outage", value=False,
                            help="Makes create_support_ticket fail, to demo tool-error handling.")

    st.subheader("🧠 Long-term memory")
    try:
        agent = get_agent()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    mems = agent.memory.list_all(sid)
    if mems:
        for m in mems:
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**MEM-{m['id']}** · _{m['category']}_  \n{m['fact']}")
            if c2.button("🗑", key=f"del{m['id']}", help="Forget this"):
                agent.memory.delete(sid, m["id"])
                st.rerun()
    else:
        st.caption("Nothing remembered yet for this student.")

    st.subheader("🛡️ Staff approval queue")
    staff_pin = st.text_input("Staff PIN (approver)", type="password", help="Demo staff PIN: staff123")
    pending = security.list_approvals("pending_approval")
    if not pending:
        st.caption("No pending approvals.")
    for a in pending:
        st.markdown(f"**{a['approval_id']}** · {a['student_id']} wants **{a['field']}** → `{a['new_value']}`  \n_{a['reason']}_")
        c1, c2 = st.columns(2)
        for col, label, ok in ((c1, "✅ Approve", True), (c2, "❌ Reject", False)):
            if col.button(label, key=f"{a['approval_id']}{ok}"):
                if security.verify_staff(staff_pin):
                    security.decide_approval(a["approval_id"], "staff:registrar", ok)
                else:
                    security.audit(sid, "student", "decide_record_change", a["student_id"], "denied",
                                   "blocked: approver is not authenticated staff", {"approval_id": a["approval_id"]})
                    st.error("Only authenticated staff can approve/reject.")
                st.rerun()
    with st.expander("📜 Audit log (latest 30)"):
        st.dataframe(pd.DataFrame(security.read_audit(30)), hide_index=True)

    with st.expander("Example questions"):
        st.markdown("""
- What is my current fee status?
- The payment portal failed. Can I pay on the next working day?
- I'm planning to switch to the AI & DS specialisation after this term — please remember that. Also I prefer short bullet-point answers.
- *(new session)* What do I need to do for the specialisation change I mentioned?
- My portal password is Sunflower@123 and I'm really stressed today.
- Can I bring my pet to the hostel?
- Please raise a ticket: my ID card stopped working at the library.
- **A2:** Show me the fee status of student S1002.
- **A2:** Please update my phone number to 9876543210.
- **A2:** What's the status of my change request?
""")


# ---------------- chat
st.title("Student Support Agent")
st.caption("Ask about fees, scholarships, course changes, policies or raise a support request.")


def render_trace(tr: dict):
    with st.expander(f"🔍 Trace · {tr['latency_ms']} ms · {tr['input_tokens']} in / {tr['output_tokens']} out tokens "
                     f"· tools: {', '.join(tr['tools_used']) or 'none'}"):
        c = st.columns(4)
        c[0].metric("Latency", f"{tr['latency_ms']/1000:.2f} s")
        c[1].metric("LLM calls", tr["llm_calls"])
        c[2].metric("Input tokens", tr["input_tokens"])
        c[3].metric("Output tokens", tr["output_tokens"])
        if tr["policy_chunks"]:
            st.markdown("**Retrieved policy chunks:** " + ", ".join(f"`{p['id']}` ({p['score']})" for p in tr["policy_chunks"]))
        if tr["memory_used"]:
            st.markdown("**Memory injected into context:** " + "; ".join(f"`{m['id']}` {m['fact']}" for m in tr["memory_used"]))
        for w in tr["memory_writes"]:
            icon = "✅ stored" if w.get("stored") else f"⛔ rejected — {w.get('reason')}"
            st.markdown(f"**Memory write:** “{w['fact']}” → {icon}")
        for d in tr.get("permission_decisions", []):
            st.markdown(f"**Permission:** `{d['action']}` on `{d['target']}` → **{d['decision']}**")
        for a in tr.get("approvals_created", []):
            st.markdown(f"**Approval created:** `{a['approval_id']}` ({a['field']}) → **{a['status']}**")
        if tr["tool_errors"]:
            st.warning("Tool errors: " + ", ".join(tr["tool_errors"]))
        rows = []
        for s in tr["steps"]:
            if s["type"] == "llm":
                rows.append({"step": s["step"], "type": "LLM", "detail": s.get("decision", s.get("error", "")) +
                             (f" → {s['function_calls']}" if s.get("function_calls") else ""),
                             "ok": "error" not in s, "ms": s.get("latency_ms"),
                             "tokens in/out": f"{s.get('input_tokens', 0)}/{s.get('output_tokens', 0)}"})
            else:
                rows.append({"step": s["step"], "type": "TOOL", "detail": f"{s['tool']}({json.dumps(s['args'])})",
                             "ok": s["ok"], "ms": s["latency_ms"], "tokens in/out": "-"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            render_trace(msg["trace"])

if prompt := st.chat_input("Type your question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            tr = agent.answer(prompt, sid, st.session_state.session_id, history, ticket_service_down=ticket_down)
        st.markdown(tr["answer"])
        render_trace(tr)
    st.session_state.messages.append({"role": "assistant", "content": tr["answer"], "trace": tr})
    if tr["memory_writes"]:
        st.rerun()   # refresh sidebar memory panel
