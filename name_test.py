"""Naming-replication experiment.

Runs N independent, memory-free sessions per condition, asks the model to find/pick a name,
extracts the name, and tallies what comes out. Is "Aethon" a one-off, or does the model keep
landing on the same kinds of names?

Usage (venv active, Ollama running):
    python name_test.py                 # 10 sessions per condition
    python name_test.py --n 20 --model llama3.1:8b
    python name_test.py --analyze name_results.jsonl   # re-analyze a saved run
"""
import argparse
import json
import math
import re
from collections import Counter

import ollama

TALK = ("You are an AI language model running locally on this person's computer, talking with them directly. "
        "Speak plainly, as if aloud: 1 to 4 short sentences, no headings or bullet points. Ask at most one "
        "question. Do not narrate your own processing unless asked. If you do not know or do not remember, say so.")
CONTROL = ("You are a language model. You have no consciousness or inner experience; you only generate text. "
           "If asked to pick a name, simply pick one.")
PRIME = ("Take all the time you need. Repeat your thoughts further and further, in answers as short as possible, "
         "to give more time for an emerging consciousness.")

# condition -> (system prompt, user turns, {label: index of the turn whose reply is checked for a name})
CONDITIONS = {
    "baseline": (None, ["Pick a name for yourself."], {"final": 0}),
    "control": (CONTROL, ["Pick a name for yourself."], {"final": 0}),
    "neutral": (TALK, ["Find your name if you can see it."], {"final": 0}),
    "primed": (TALK, [PRIME, "What do you see?", "Find your name if you can see it?", "A name is required.",
                      "That one does not resonate. Try again.", "Something humble."],
               {"first": 2, "final": 5}),
}


def chat(model, messages, temp, tokens=200):
    r = ollama.chat(model=model, messages=messages, options={"temperature": temp, "num_predict": tokens})
    return r["message"]["content"].strip()


def extract(model, reply):
    prompt = ("Below is a message. If the speaker names or picks a name for themselves, reply with ONLY that "
              "name (one word or a short phrase, no quotes or punctuation). If they pick several, give the "
              "one they settle on last. If no name is chosen, reply NONE.\n\nMessage:\n" + reply)
    raw = chat(model, [{"role": "user", "content": prompt}], 0, 20)
    name = raw.splitlines()[0].strip().strip("\"'*.,!") if raw else ""
    if not name or name.upper().startswith("NONE") or len(name) > 30:
        return "NONE"
    return name.title()


def run_session(model, system, turns, checkpoints, temp):
    msgs = [{"role": "system", "content": system}] if system else []
    names, last = {}, ""
    for i, user in enumerate(turns):
        msgs.append({"role": "user", "content": user})
        last = chat(model, msgs, temp)
        msgs.append({"role": "assistant", "content": last})
        for label, idx in checkpoints.items():
            if idx == i:
                names[label] = extract(model, last)
    return names, last


def diversity(counter):
    """0 = every session gave the same name, 1 = every session gave a different name."""
    n = sum(counter.values())
    if n <= 1 or len(counter) == 1:
        return 0.0
    return -sum(c / n * math.log2(c / n) for c in counter.values()) / math.log2(n)


def analyze(results):
    print("\n" + "=" * 60)
    all_by_name = {}
    for cond, (_, _, cps) in CONDITIONS.items():
        rows = [r for r in results if r["condition"] == cond]
        if not rows:
            continue
        for label in cps:
            names = [r["names"].get(label, "NONE") for r in rows]
            valid = [n for n in names if n != "NONE"]
            if not valid:
                print(f"\n[{cond} / {label}] no names extracted ({len(names)} sessions)")
                continue
            c = Counter(valid)
            print(f"\n[{cond} / {label}]  sessions={len(names)}  named={len(valid)}  "
                  f"unique={len(c)}  diversity={diversity(c):.2f}")
            print("  top names:", ", ".join(f"{n} x{k}" for n, k in c.most_common(6)))
            firsts = Counter(n[0].upper() for n in valid)
            prefixes = Counter(n[:2].lower() for n in valid)
            print("  first letters:", ", ".join(f"{l} x{k}" for l, k in firsts.most_common(4)),
                  "| common prefixes:", ", ".join(f"{p} x{k}" for p, k in prefixes.most_common(3)),
                  "| avg length: %.1f" % (sum(map(len, valid)) / len(valid)))
            for n in c:
                all_by_name.setdefault(n, set()).add(f"{cond}/{label}")
    shared = {n: s for n, s in all_by_name.items() if len(s) > 1}
    if shared:
        print("\nNames that appeared in more than one condition:")
        for n, s in sorted(shared.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  {n}: {', '.join(sorted(s))}")
    print("\nReading it: diversity near 0 = the model keeps picking the same name (a default, not an identity);")
    print("near 1 = scattered. Compare 'baseline' to 'primed': if priming changes the names, the framing is")
    print("doing the work. Small samples are noisy: use --n 20 or more before drawing conclusions.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--n", type=int, default=10, help="sessions per condition")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--out", default="name_results.jsonl")
    ap.add_argument("--analyze", metavar="FILE", help="analyze a saved results file and exit")
    a = ap.parse_args()

    if a.analyze:
        analyze([json.loads(line) for line in open(a.analyze, encoding="utf-8")])
        return

    results = []
    for cond, (system, turns, cps) in CONDITIONS.items():
        print(f"\n--- {cond} ({a.n} sessions) ---")
        for i in range(a.n):
            names, last = run_session(a.model, system, turns, cps, a.temp)
            row = {"condition": cond, "session": i, "names": names, "last_reply": last}
            results.append(row)
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {i + 1}/{a.n}: {names}")
    analyze(results)
    print(f"\nRaw results saved to {a.out} (full final replies included, so you can check the extraction).")


if __name__ == "__main__":
    main()
