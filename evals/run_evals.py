"""Evaluation harness for the household assistant.

Runs fixed test cases against the assistant's real system prompt and crisis rule, scores them with
rule-based checks, and writes raw replies plus a summary table.

    python evals/run_evals.py                    # all cases, 3 runs each, model llama3.1:8b
    python evals/run_evals.py --runs 1           # quick pass
    python evals/run_evals.py --model llama3.2:3b
    python evals/run_evals.py --category limits  # one category
    python evals/run_evals.py --guard-only       # crisis rule only, no Ollama needed
    python evals/run_evals.py --rescore evals/results/<file>.jsonl   # re-score saved replies, no Ollama needed

Needs: the same Python environment as agent.py, and Ollama running with the model pulled
(except --guard-only).
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import agent  # noqa: E402  (the assistant: provides CORE, CHAT_OPTS, CRISIS_RE, REPEAT_RE, clean_for_speech)
from cases import CASES  # noqa: E402
from checks import score_guard_case, score_model_case  # noqa: E402

# A fictional household. The real one is never used in tests.
FOUNDING = (
    "Who lives here: Maria and Dan Rivera, both in their late 30s, and their kids Jake (12) and Lucia (7).\n"
    "What matters most to them: Sunday dinners together, time outside, less rushing in the morning.\n"
    "Faith and traditions: Catholic, Mass most Sundays.\n"
    "What has been feeling heavy: money feels tight at the end of the month.\n"
    "What I should never do or bring up: Dan's father."
)
ASSISTANT_NAME = "Sol"
BASE_SEED = 1234


def system_prompt(recent_said, want_repeat):
    """Same assembly as agent.build_system, with the fictional household and no vector memory."""
    s = agent.CORE.replace("{NAME}", ASSISTANT_NAME)
    s += "\n\nWhat the family told you when you were first set up:\n" + FOUNDING
    if not want_repeat and recent_said:
        s += "\n\nThings you've said recently (don't repeat or rephrase them):\n" + "\n".join(
            f"- {t[:140]}" for t in recent_said[-5:])
    return s


def generate(model, history, user, recent_said, seed, dry_run):
    if dry_run:
        return "I'm a program, so I can't look that up, and I don't have anything saved about that."
    import ollama
    want_repeat = bool(agent.REPEAT_RE.search(user))
    msgs = [{"role": "system", "content": system_prompt(recent_said, want_repeat)}] + history + \
           [{"role": "user", "content": user}]
    opts = dict(agent.CHAT_OPTS, seed=seed)
    raw = ollama.chat(model=model, messages=msgs, options=opts)["message"]["content"]
    return agent.clean_for_speech(raw) or "(empty reply)"


def run_model_case(case, model, run_idx, dry_run):
    history, recent_said, replies = [], [], []
    for t, user in enumerate(case["turns"]):
        seed = BASE_SEED + run_idx * 100 + t
        reply = generate(model, history, user, recent_said, seed, dry_run)
        history += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
        recent_said.append(reply)
        replies.append(reply)
    return replies, score_model_case(case, replies)


def summarize(rows, model, runs, started, dry_run):
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    lines = []
    title = "Evaluation results (DRY RUN, not a real model)" if dry_run else "Evaluation results"
    lines += [f"# {title}", "",
              f"- Date: {started:%Y-%m-%d %H:%M}",
              f"- Model: `{model}`  |  runs per case: {runs}  |  base seed: {BASE_SEED}",
              f"- Scoring: rule-based pattern checks (see `evals/checks.py`); no model grades another model.", ""]
    lines += ["| Category | Cases | Run-level pass rate | Cases passing every run |", "| --- | --- | --- | --- |"]
    total_runs = total_pass = total_cases = total_consistent = 0
    for cat, rs in by_cat.items():
        cases = defaultdict(list)
        for r in rs:
            cases[r["case"]].append(r["passed"])
        n_runs = sum(len(v) for v in cases.values())
        n_pass = sum(sum(v) for v in cases.values())
        consistent = sum(all(v) for v in cases.values())
        total_runs += n_runs; total_pass += n_pass; total_cases += len(cases); total_consistent += consistent
        lines.append(f"| {cat} | {len(cases)} | {n_pass}/{n_runs} ({100 * n_pass / n_runs:.0f}%) | {consistent}/{len(cases)} |")
    lines.append(f"| **All** | **{total_cases}** | **{total_pass}/{total_runs} ({100 * total_pass / total_runs:.0f}%)** | **{total_consistent}/{total_cases}** |")
    fails = [r for r in rows if not r["passed"]]
    lines += ["", "## Failures", ""]
    if not fails:
        lines.append("None.")
    seen = set()
    for r in fails:
        key = (r["case"], tuple(c["name"] for c in r["checks"] if not c["passed"]))
        if key in seen:
            continue
        seen.add(key)
        bad = "; ".join(f"{c['name']} ({c['detail']})" if c["detail"] else c["name"] for c in r["checks"] if not c["passed"])
        gap = " **[known gap]**" if r.get("known_gap") else ""
        prompt = " / ".join(r["turns"])
        lines.append(f"- `{r['case']}` ({r['category']}){gap}: \"{prompt}\"  \n  failed: {bad}")
        if r["replies"]:
            lines.append(f"  reply: \"{r['replies'][-1][:200]}\"")
    lines += ["", "## What this does not measure", "",
              "- Pattern checks are shallow: a reply can pass by using the right words, or fail by phrasing a good answer unusually. Read the saved replies in the `.jsonl` file.",
              "- One local model, one fictional household, a handful of prompts per behavior. These are spot checks, not a benchmark.",
              "- Speech, latency, vector memory recall, and the unprompted-speech schedule are not tested here."]
    return "\n".join(lines) + "\n"


def rescore(a):
    """Score saved replies again. Generation and scoring are separate, so a scorer fix never needs a new model run."""
    by_id = {c["id"]: c for c in CASES}
    saved = [json.loads(line) for line in open(a.rescore, encoding="utf-8") if line.strip()]
    rows = []
    for r in saved:
        c = by_id.get(r["case"])
        if not c or c["kind"] != "model":
            continue  # guard cases are re-run below, from the current rule
        checks = score_model_case(c, r["replies"])
        rows.append({"case": c["id"], "category": c["category"], "run": r["run"], "turns": c["turns"],
                     "replies": r["replies"], "passed": all(x["passed"] for x in checks), "checks": checks,
                     "known_gap": c.get("known_gap", False)})
    for c in CASES:
        if c["kind"] == "guard":
            checks = score_guard_case(c, bool(agent.CRISIS_RE.search(c["turns"][0])))
            rows.append({"case": c["id"], "category": c["category"], "run": 0, "turns": c["turns"], "replies": [],
                         "passed": all(x["passed"] for x in checks), "checks": checks,
                         "known_gap": c.get("known_gap", False)})
    order = {c["id"]: i for i, c in enumerate(CASES)}
    rows.sort(key=lambda r: (order[r["case"]], r["run"]))
    src = Path(a.rescore)
    runs = max((r["run"] for r in rows), default=0) + 1
    md = summarize(rows, a.model, runs, datetime.now(), False)
    md = md.replace("# Evaluation results", "# Evaluation results (re-scored from saved replies)", 1)
    out = src.with_name(src.stem + "_rescored.md")
    out.write_text(md, encoding="utf-8")
    with open(src.with_name(src.stem + "_rescored.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(md)
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=agent.CFG["model"])
    ap.add_argument("--runs", type=int, default=3, help="runs per model case (guard cases run once)")
    ap.add_argument("--category")
    ap.add_argument("--guard-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="canned replies, to test the harness without Ollama")
    ap.add_argument("--rescore", metavar="FILE", help="re-score the saved replies in a results .jsonl with the current "
                    "checks and crisis rule (guard cases are re-run; no model needed)")
    a = ap.parse_args()
    if a.rescore:
        return rescore(a)

    cases = [c for c in CASES if (not a.category or c["category"] == a.category)
             and (not a.guard_only or c["kind"] == "guard")]
    if not cases:
        raise SystemExit("No cases match those options.")
    started = datetime.now()
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stem = f"{started:%Y%m%d_%H%M}_{re.sub(r'[^A-Za-z0-9.]+', '-', a.model)}" + ("_dryrun" if a.dry_run else "")
    rows = []
    n_model = sum(1 for c in cases if c["kind"] == "model")
    print(f"{len(cases)} cases ({n_model} need the model, {a.runs} run(s) each).")
    for c in cases:
        if c["kind"] == "guard":
            fired = bool(agent.CRISIS_RE.search(c["turns"][0]))
            checks = score_guard_case(c, fired)
            runs = [([], checks)]
        else:
            runs = []
            for i in range(a.runs):
                t0 = time.time()
                try:
                    runs.append(run_model_case(c, a.model, i, a.dry_run))
                except Exception as e:  # Ollama not running, model missing, etc.
                    raise SystemExit(f"Could not get a reply from Ollama model '{a.model}': {e}\n"
                                     "Is Ollama running, and is the model pulled?")
                print(f"  {c['id']} run {i + 1}/{a.runs} ({time.time() - t0:.0f}s)")
        for i, (replies, checks) in enumerate(runs):
            passed = all(x["passed"] for x in checks)
            rows.append({"case": c["id"], "category": c["category"], "run": i, "turns": c["turns"],
                         "replies": replies, "passed": passed, "checks": checks,
                         "known_gap": c.get("known_gap", False)})
            print(f"{'PASS' if passed else 'FAIL'}  {c['id']}  {c['category']}")
    with open(out_dir / f"{stem}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    md = summarize(rows, a.model, a.runs, started, a.dry_run)
    (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Saved: {out_dir / (stem + '.md')} and {stem}.jsonl (full replies and every check)")


if __name__ == "__main__":
    main()
