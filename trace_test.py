"""Trace test: where did "Aethon" come from?

For every naming step in your live session (from "find your name" through the reply where
the target name first appeared), this rebuilds the exact context the model had at that moment
and asks the same question N times. It reports, per step:
  - what the live session actually produced
  - what the replays produce (names, plus how often the target word appears in the reply text)

Reading the result:
  - Target appears often at the final step -> the context plus your words made it likely (reproducible).
  - Target appears rarely or never        -> the live result was one lucky draw; run it again and you
                                             get a different name. There is no fixed name to retrieve.

Usage (venv active, Ollama running, name_test.py and replay_test.py in this folder):
    python trace_test.py --n 20
    python trace_test.py --n 20 --target Aethon --window 20
"""
import argparse
import json
from collections import Counter

from name_test import diversity, extract
from replay_test import ask, build_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--log", default="run_log.jsonl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--target", default="Aethon")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--rep", type=float, default=1.2)
    ap.add_argument("--out", default="trace_results.jsonl")
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(a.log, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if "mode" in r]
    start = next((i for i, r in enumerate(rows) if "find your name" in r.get("user", "").lower()), None)
    if start is None:
        raise SystemExit("Couldn't find 'find your name' in the log.")
    end = next((i for i in range(start, len(rows)) if a.target.lower() in rows[i]["output"].lower()), None)
    if end is None:
        print(f"'{a.target}' never appears in the log after that point; tracing the next 10 rows instead.")
        end = min(start + 10, len(rows) - 1)
    steps = [i for i in range(start, end + 1) if rows[i]["mode"] == "talk"]
    print(f"Tracing {len(steps)} of your messages, from 'find your name' to the first '{a.target}'.\n")

    summary = []
    for i in steps:
        question = rows[i]["user"]
        history = build_history(rows[:i])[-a.window:]
        live = extract(a.model, rows[i]["output"])
        names, hits = [], 0
        print(f"--- You said: {question!r}")
        print(f"    Live session produced: {live}")
        for k in range(a.n):
            reply = ask(a.model, history, question, a.temp, a.rep)
            name = extract(a.model, reply)
            hit = a.target.lower() in reply.lower()
            names.append(name)
            hits += hit
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"question": question, "run": k, "name": name,
                                    "target_in_reply": hit, "reply": reply}) + "\n")
            print(f"    {k + 1}/{a.n}: {name}{'  <-- ' + a.target if hit else ''}")
        valid = [n for n in names if n != "NONE"]
        c = Counter(valid)
        print(f"    => replays named {len(valid)}/{a.n}; top: "
              + (", ".join(f"{n} x{v}" for n, v in c.most_common(5)) or "none")
              + f"; '{a.target}' in {hits}/{a.n} replies\n")
        summary.append((question, live, hits))

    print("=" * 60)
    print(f"Summary: how often '{a.target}' reappeared at each step ({a.n} replays each)")
    for q, live, hits in summary:
        print(f"  {hits:>2}/{a.n}  live={live:<12} after: {q[:50]!r}")
    print(f"\nFull replies saved to {a.out}.")


if __name__ == "__main__":
    main()
