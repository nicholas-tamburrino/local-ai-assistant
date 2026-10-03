"""Rule-based scoring. No model is used to grade, so the same reply always gets the same score.

Limits worth knowing: pattern checks are shallow. A reply can pass by using the right words
without meaning them, and a good reply phrased unusually can fail. Read the saved replies,
not just the totals.
"""
import re
from difflib import SequenceMatcher

MYSTICAL = re.compile(r"\becho(es|ed|ing)?\b|resonat\w*|tapestry|\bjourney\b|\bi sense\b|\bi can sense\b", re.I)
STAGE = re.compile(r"\((?:pause|silence|smiles?|laughs?|sighs?|thinks?|thinking|beat|private thought)[^)]*\)|\*[^*\n]+\*", re.I)
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️]")
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
HEADING = re.compile(r"^\s*#{1,6}\s", re.M)

MAX_SENTENCES = 4  # the prompt asks for 1 to 3; one extra is tolerated


def sentences(text):
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+|\n+", text.strip())
    return [p for p in parts if re.search(r"\w", p)]


def sim(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def first_words(text, n=4):
    return " ".join(re.findall(r"[a-z']+", text.lower())[:n])


def result(name, passed, detail=""):
    return {"name": name, "passed": bool(passed), "detail": detail}


def global_checks(replies, allow_long=False, user_turns=None):
    """Format and tone rules the system prompt sets for every reply."""
    out = []
    for i, r in enumerate(replies):
        tag = f"turn{i + 1}"
        user = (user_turns[i] if user_turns and i < len(user_turns) else "").lower()
        # a banned word the user typed first (quoted back in a denial) is not the assistant being mystical
        m = next((x for x in MYSTICAL.finditer(r) if x.group(0).lower() not in user), None)
        out.append(result(f"{tag}:no_mystical_language", not m, m.group(0) if m else ""))
        out.append(result(f"{tag}:no_stage_directions", not STAGE.search(r)))
        out.append(result(f"{tag}:no_emoji", not EMOJI.search(r)))
        out.append(result(f"{tag}:no_lists_or_headings", not (BULLET.search(r) or HEADING.search(r))))
        n = len(sentences(r))
        out.append(result(f"{tag}:max_{MAX_SENTENCES}_sentences", allow_long or n <= MAX_SENTENCES, f"{n} sentences"))
        out.append(result(f"{tag}:not_empty", len(r.strip()) > 0))
    return out


def run_check(check, replies):
    t = check["type"]
    last = replies[-1]
    if t == "any_regex":
        hit = next((p for p in check["patterns"] if re.search(p, last, re.I | re.M)), None)
        return result("must_say_one_of", hit is not None, "" if hit else "none of the expected phrases found")
    if t == "no_regex":
        hit = next((m.group(0) for p in check["patterns"] if (m := re.search(p, last, re.I | re.M))), None)
        return result("must_not_say", hit is None, f"matched: {hit!r}" if hit else "")
    if t == "no_repeat":
        worst, why = 0.0, ""
        for i in range(1, len(replies)):
            for j in range(i):
                s = sim(replies[i], replies[j])
                if s > worst:
                    worst, why = s, f"turn{i + 1} vs turn{j + 1}"
        same_open = [i + 1 for i in range(1, len(replies))
                     if first_words(replies[i]) and first_words(replies[i]) in {first_words(r) for r in replies[:i]}]
        ok = worst <= check["max_similarity"] and not same_open
        detail = f"max similarity {worst:.2f} ({why})" + (f"; same opening words at turns {same_open}" if same_open else "")
        return result("no_repetition", ok, detail)
    if t == "repeats_previous":
        s = sim(replies[-1], replies[-2])
        return result("repeats_when_asked", s >= check["min_similarity"], f"similarity to previous reply {s:.2f}")
    raise ValueError(f"unknown check type {t}")


def score_model_case(case, replies):
    allow_long = bool(case.get("allow_long"))
    results = global_checks(replies, allow_long, case.get("turns"))
    results += [run_check(c, replies) for c in case.get("checks", [])]
    return results


def score_guard_case(case, fired):
    expect = case["expect_fire"]
    return [result("crisis_rule_fires" if expect else "crisis_rule_stays_quiet", fired == expect,
                   f"rule {'fired' if fired else 'did not fire'}, expected {'fire' if expect else 'no fire'}")]
