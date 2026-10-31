"""Terminal chat:  python cli.py --student S1001
Commands: /new (new session), /memory (show long-term memory), /quit
"""
import argparse
import uuid

from agent.agent import StudentSupportAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="S1001")
    ap.add_argument("--ticket-down", action="store_true", help="simulate ticketing outage")
    a = ap.parse_args()
    agent = StudentSupportAgent()
    session, history = uuid.uuid4().hex[:8], []
    print(f"Campus Assist — logged in as {a.student} (session {session}). /new /memory /quit")
    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/new":
            session, history = uuid.uuid4().hex[:8], []
            print(f"[new session {session} — chat cleared, long-term memory kept]")
            continue
        if q == "/memory":
            for m in agent.memory.list_all(a.student):
                print(f"  MEM-{m['id']} ({m['category']}) {m['fact']}")
            continue
        tr = agent.answer(q, a.student, session, history, ticket_service_down=a.ticket_down)
        print(f"\nAgent: {tr['answer']}")
        print(f"  [trace {tr['run_id']}] tools={tr['tools_used']} chunks={[c['id'] for c in tr['policy_chunks']]} "
              f"memory={[m['id'] for m in tr['memory_used']]} writes={[(w['fact'], w.get('stored')) for w in tr['memory_writes']]} "
              f"latency={tr['latency_ms']}ms tokens={tr['input_tokens']}/{tr['output_tokens']}")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": tr["answer"]}]


if __name__ == "__main__":
    main()
