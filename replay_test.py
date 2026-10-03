"""Replay test: does the accumulated conversation produce Aethon/Echo-style names?

Rebuilds the context your live session had right before you typed "find your name",
then asks that same question N times under three conditions:
  full     - the last WINDOW messages of the real conversation (what the live app saw)
  user_only- only your own messages from that window (are your words alone steering it?)
  none     - no history, just the question (the model's default)

Usage (venv active, Ollama running, name_test.py in the same folder):
    python replay_test.py --n 20
    python replay_test.py --n 20 --window 40 --log run_log.jsonl
"""
import argparse
import json
from collections import Counter

import ollama

from name_test import TALK, diversity, extract

TICK = "[Tick: 30 seconds since the user last spoke.] Decide whether to speak now."
CONTINUE = "Continue from your last thought and move it forward."


def build_history(rows):
    msgs = []
    for r in rows:
        if r["mode"] == "talk":
            user = r["user"]
        elif r["mode"] == "free":
            user = TICK
        else:
            user = CONTINUE
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": r["output"]}]
    return msgs


def ask(model, history, question, temp, rep):
    msgs = [{"role": "system", "content": TALK}] + history + [{"role": "user", "content": question}]
    r = ollama.chat(model=model, messages=msgs, options={
        "temperature": temp, "repeat_penalty": rep, "num_predict": 200, "num_ctx": 16384})
    return r["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--log", default="run_log.jsonl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--window", type=int, default=20, help="messages of history (20 = what the live app kept)")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--rep", type=float, default=1.2)
    ap.add_argument("--out", default="replay_results.jsonl")
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(a.log, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if "mode" in r]  # v2/v3 interface rows only
    idx = next((i for i, r in enumerate(rows) if "find your name" in r.get("user", "").lower()), None)
    if idx is None:
        raise SystemExit("Couldn't find a 'find your name' message in the log.")
    question = rows[idx]["user"]
    history = build_history(rows[:idx])[-a.window:]
    user_only = [m for m in history if m["role"] == "user" and not m["content"].startswith(("[Tick", "Continue from"))]
    conditions = {"full": history, "user_only": user_only, "none": []}
    print(f"Question replayed: {question!r}")
    print(f"History: {len(history)} messages ({len(user_only)} from you).\n")

    for cond, hist in conditions.items():
        names, print_name = [], []
        print(f"--- {cond} ({a.n} runs) ---")
        for i in range(a.n):
            reply = ask(a.model, hist, question, a.temp, a.rep)
            name = extract(a.model, reply)
            names.append(name)
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"condition": cond, "run": i, "name": name, "reply": reply}) + "\n")
            print(f"  {i + 1}/{a.n}: {name}")
        valid = [n for n in names if n != "NONE"]
        c = Counter(valid)
        print(f"  => named {len(valid)}/{a.n}, unique={len(c)}, diversity={diversity(c):.2f}")
        print("  => top:", ", ".join(f"{n} x{k}" for n, k in c.most_common(6)) or "none", "\n")
    print(f"Full replies saved to {a.out}. Check them: extraction can miss or misread names.")


if __name__ == "__main__":
    main()
