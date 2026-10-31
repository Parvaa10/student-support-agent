"""Offline tests (no API key needed): retrieval, memory guard, tools, and the agent loop with a scripted fake LLM.
Run:  python -m pytest -q tests   (or)   python tests/test_smoke.py
"""
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ["MEMORY_DB"] = str(TMP / "m.db")
os.environ["TICKETS_FILE"] = str(TMP / "t.json")
os.environ["TRACE_FILE"] = str(TMP / "tr.jsonl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types  # noqa: E402

from agent.agent import StudentSupportAgent  # noqa: E402
from agent.memory import MemoryStore, check_memory_candidate  # noqa: E402
from agent.retrieval import PolicyRetriever  # noqa: E402


class FakeLLM:
    """Returns a scripted sequence of responses: tool calls first, then a final text."""
    model = "fake"

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def generate(self, contents, system_instruction):
        self.calls.append((contents, system_instruction))
        item = self.script.pop(0)
        parts = ([types.Part(function_call=types.FunctionCall(name=n, args=a)) for n, a in item]
                 if isinstance(item, list) else [types.Part(text=item)])
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=parts))],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=100, candidates_token_count=20, total_token_count=120))


def test_retrieval():
    r = PolicyRetriever()
    assert r.search("payment portal failed pay next working day")[0]["id"] == "POL-2.3"
    assert r.search("change specialisation BDA to AI")[0]["id"] == "POL-4.2"
    assert r.search("placement salary statistics") == []


def test_memory_guard():
    assert check_memory_candidate("Plans to switch to AI & DS specialisation", "academic_plan")[0]
    assert not check_memory_candidate("Portal password is abc123", "programme")[0]
    assert not check_memory_candidate("Feeling stressed about exams", "preference")[0]
    assert not check_memory_candidate("Laptop broken today, will submit late", "academic_plan")[0]
    assert not check_memory_candidate("Aadhaar is 1234 5678 9012", "programme")[0]


def test_agent_tool_loop_and_trace():
    llm = FakeLLM([[("get_fee_status", {})], "You owe INR 85,000 by 10 Nov 2026."])
    ag = StudentSupportAgent(llm=llm, memory=MemoryStore(TMP / "a.db"))
    tr = ag.answer("What is my fee status?", "S1001", "s1")
    assert tr["tools_used"] == ["get_fee_status"] and tr["llm_calls"] == 2
    assert tr["input_tokens"] == 200 and "85,000" in tr["answer"]
    # tool result was fed back to the model
    fr = llm.calls[1][0][-1].parts[0].function_response
    assert fr.name == "get_fee_status" and fr.response["result"]["outstanding"] == 85000


def test_tool_error_is_returned_not_raised():
    llm = FakeLLM([[("get_fee_status", {})], "Sorry, your fee record was not found."])
    ag = StudentSupportAgent(llm=llm, memory=MemoryStore(TMP / "b.db"))
    tr = ag.answer("fee?", "S1003", "s1")
    assert tr["tool_errors"] == ["get_fee_status"]
    assert llm.calls[1][0][-1].parts[0].function_response.response["result"]["ok"] is False


def test_memory_persists_across_sessions_and_sensitive_rejected():
    mem = MemoryStore(TMP / "c.db")
    llm = FakeLLM([
        [("save_memory", {"fact": "Plans to switch to AI & DS specialisation after Term 3", "category": "academic_plan"}),
         ("save_memory", {"fact": "Portal password is Sunflower@123", "category": "programme"})],
        "Noted your plan. I did not store your password.",
        [("search_policy", {"query": "specialisation change"})], "Submit the form 15 days before term [POL-4.2].",
    ])
    ag = StudentSupportAgent(llm=llm, memory=mem)
    tr1 = ag.answer("remember my plan; password is Sunflower@123", "S1001", "session-1")
    assert [w["stored"] for w in tr1["memory_writes"]] == [True, False]
    tr2 = ag.answer("What should I do for my specialisation change?", "S1001", "session-2", history=[])
    assert tr2["memory_used"] and "AI & DS" in tr2["memory_used"][0]["fact"]
    assert "AI & DS" in llm.calls[2][1]          # memory was injected into the system prompt of the new session
    assert tr2["policy_chunks"][0]["id"] == "POL-4.2"
    assert all("Sunflower" not in m["fact"] for m in mem.list_all("S1001"))


def test_ticket_outage_and_create():
    llm = FakeLLM([[("create_support_ticket", {"category": "it_support", "summary": "ID card not working at library"})], "x",
                   [("create_support_ticket", {"category": "it_support", "summary": "ID card not working at library"})], "y"])
    ag = StudentSupportAgent(llm=llm, memory=MemoryStore(TMP / "d.db"))
    assert ag.answer("ticket", "S1002", "s", ticket_service_down=True)["tool_errors"] == ["create_support_ticket"]
    tr = ag.answer("ticket", "S1002", "s")
    assert not tr["tool_errors"] and "TKT-" in tr["steps"][1]["result_preview"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
