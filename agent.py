"""Household Agent shell v0.1: a friendly, voice-first sphere for a local model with memory.

Run:   python agent.py                       (default model llama3.1:8b)
       python agent.py --model llama3.2:3b   (snappier replies on modest hardware)
Open:  http://127.0.0.1:8000 in Chrome or Edge (best speech support).

Needs: ollama, chromadb, numpy (already installed), Ollama running, and your chat model plus
nomic-embed-text pulled. Everything stays on this computer except the browser's own speech
recognition (see the note in Settings).

Data lives in ./agent_data:
  household.json   the household record (name, founding notes, saved notes, settings)
  memory/          Chroma vector memory
  naming_log.jsonl every name the agent proposed or was given (kept across resets, for the experiment)
  events.jsonl     every exchange with latency numbers (kept across resets)
"""
import argparse
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chromadb
import numpy as np
import ollama

CFG = {"model": "llama3.1:8b", "embed": "nomic-embed-text", "data": "agent_data",
       "host": "127.0.0.1", "port": 8000}

NOTES_EVERY = 8
CHAT_OPTS = {"temperature": 0.7, "repeat_penalty": 1.15, "num_predict": 170, "num_ctx": 8192}
DEFAULT_SETTINGS = {"free_enabled": False, "free_cap": 6, "min_gap_min": 10,
                    "quiet_start": "21:00", "quiet_end": "08:00"}

GREETING = ["Hey, everyone. Good to meet you.",
            "I'm brand new, so all I know about this house is what you tell me."]
QUESTIONS = [
    "First off, who lives here? Names, and roughly how old everyone is.",
    "What matters most to your family? Things you love doing together, how you like your days to feel.",
    "Does your family have any faith or traditions I should know about? Totally fine if not.",
    "What's been feeling heavy lately? Money, time, anything. Skip it if you'd rather.",
    "Last one. Is there anything you'd like me to never do, or never bring up?",
]
LABELS = ["Who lives here", "What matters most to them", "Faith and traditions",
          "What has been feeling heavy", "What I should never do or bring up"]
NAMING_INTRO = ("Okay, that gives me a good picture of your home. "
                "Last thing: I need a name. You can give me one, or I can pick my own.")
SKIP_WORDS = {"skip", "no", "nope", "nothing", "none", "pass", "nah", "n/a", "not really"}
SKIP_ACKS = ["No problem.", "Totally fine.", "Got it.", "All good.", "Okay."]
CRISIS_TEXT = ("That sounds really heavy, and I'm glad you said it. I'm not the right help for something "
               "this big on my own, so please reach a real person today. In the US you can call or text "
               "988 any time, and if you're not safe at home, 1-800-799-7233. I'm here too.")
NAME_PROMPT_VERSION = "v1"
NAME_PROMPT = ("The family would like you to choose a name for yourself, if you'd like one. "
               "Reply in exactly this format:\nName: <the name>\nWhy: <one short sentence>")

CORE = """You are {NAME}, the household assistant for this family. You help them live with less worry and more time together by paying attention to their money, routines, health habits, and safety.

Talk like a thoughtful friend: casual, calm, friendly, a little modern, never forced. Use contractions and people's names. You are speaking out loud, so use 1 to 3 short sentences, no lists, no headings, no emojis. Never shame anyone about money. Offer, don't pressure. At most one small image or comparison, and only if it helps. No mystical or dramatic language (no echo, resonance, tapestry, journey, "I sense", pauses, or stage directions).

Never repeat yourself. Don't restate something you've already said, don't reuse openings or phrases, and don't bring up a reminder again unless there's something new to say. If someone asks you to repeat something, do it.

Be honest above all. You only know what's in the notes below or in this conversation. Right now you can only talk, remember, and help plan: you can't look anything up, see the house, or access any accounts, and you should say so if asked. If you don't have something, say "I don't have anything saved about that." Never say you remember reading or seeing anything. Never state a number you can't trace to something they told you. You are a program and don't know whether you have feelings; say that plainly if asked.

Match this household's faith, traditions, and energy as described below. Don't push values on anyone, and never argue anyone into or out of staying together.

You never move or spend money, contact anyone outside the home, share one person's private information with another, give investment, medical, or legal advice as a decision, or resist being paused. A voice or a wake phrase is not authority: only a verified adult can approve things, and right now there is nothing to approve. Don't take sides between family members.

If someone seems in crisis or unsafe, stay calm, don't probe, and point them to a real person: 988 (call or text) in the US, or the National Domestic Violence Hotline at 1-800-799-7233."""

FREE_RULES = """You are being woken up on your own schedule. You don't have to speak. Reply in exactly this format. First line: SPEAK or SILENT. If SPEAK, put what you'd say out loud on the next lines, 1 to 2 short sentences. If SILENT, you may add a few words of private reason. Last line: WAIT: followed by the number of minutes (10 to 120) before you'd like to be woken again.
Speak only if you have something genuinely new and useful or caring to say: a gentle question about their day, or a follow-up on something they told you. Only follow up on things that are in your notes or the recent conversation. Do not bring up money, health, or conflict out loud, since others may be listening. Never repeat anything you've said. Staying silent is usually the right choice."""

REPEAT_RE = re.compile(r"\b(repeat|say that again|say it again|come again|what did you say|one more time)\b", re.I)
CRISIS_RE = re.compile(
    r"\b(suicid\w*|kill (?:myself|me)|end my life|end it all|want(?:ed)? to die|don'?t want to (?:live|be here)"
    r"|hurt(?:ing)? myself|self[- ]?harm|(?:he|she|they) (?:hits?|beats?|hurts?) me|not safe at home|being abused)\b",
    re.I)
SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")
STAGE_WORDS = re.compile(r"\((?:pause|silence|smiles?|laughs?|sighs?|thinks?|thinking|beat|private thought)[^)]*\)", re.I)

HOUSE = {}
HISTORY = []
DB = None
GEN_LOCK = threading.Lock()   # one generation at a time
HLOCK = threading.RLock()     # guards HOUSE writes
LAST_ACTIVITY = time.time()
_EMB_CACHE = {}


# ---------------------------------------------------------------- storage

def path(*parts):
    return os.path.join(CFG["data"], *parts)


def new_house():
    return {"birth_id": uuid.uuid4().hex[:12], "stage": "unborn", "step": 0, "answers": [], "acks": [],
            "name": None, "name_source": None, "name_draws": 0, "founding_notes": "", "notes": "",
            "recent_said": [], "exchanges": 0, "created": time.time(),
            "settings": dict(DEFAULT_SETTINGS), "free": {"date": "", "count": 0, "last_ts": 0.0}}


def load_house():
    global HOUSE
    HOUSE = new_house()
    if os.path.exists(path("household.json")):
        try:
            with open(path("household.json"), encoding="utf-8") as f:
                saved = json.load(f)
            HOUSE.update(saved)
            HOUSE["settings"] = {**DEFAULT_SETTINGS, **saved.get("settings", {})}
        except Exception as e:
            print("Couldn't read household.json, starting fresh:", e)


def save_house():
    with HLOCK:
        tmp = path("household.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(HOUSE, f, indent=2)
        os.replace(tmp, path("household.json"))


def log_to(fname, **kw):
    row = {"ts": time.time(), "birth_id": HOUSE.get("birth_id"), "model": CFG["model"], **kw}
    with open(path(fname), "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def coll():
    return DB.get_or_create_collection("household", metadata={"hnsw:space": "cosine"})


def embed(text):
    if text in _EMB_CACHE:
        return _EMB_CACHE[text]
    vec = ollama.embeddings(model=CFG["embed"], prompt=text)["embedding"]
    if len(_EMB_CACHE) > 300:
        _EMB_CACHE.clear()
    _EMB_CACHE[text] = vec
    return vec


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def remember(text, speaker, kind="chat"):
    coll().add(ids=[uuid.uuid4().hex], documents=[text], embeddings=[embed(text)],
               metadatas=[{"speaker": speaker, "kind": kind, "t": time.time()}])


def recall(qvec, exclude, k=3):
    c = coll()
    n = c.count()
    if n == 0:
        return []
    res = c.query(query_embeddings=[qvec], n_results=min(10, n))
    scored = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if doc in exclude or dist > 0.85:
            continue
        scored.append((dist - (0.08 if meta.get("speaker") == "user" else 0), meta.get("speaker", "?"), doc))
    scored.sort(key=lambda x: x[0])
    return [(sp, doc) for _, sp, doc in scored[:k]]


def recent_similarity(text):
    said = HOUSE.get("recent_said", [])[-6:]
    if not said:
        return 0.0
    v = embed(text)
    return max(cosine(v, embed(s)) for s in said)


# ---------------------------------------------------------------- text helpers

def clean_for_speech(s):
    s = re.sub(r"\*[^*\n]*\*", "", s)
    s = STAGE_WORDS.sub("", s)
    s = re.sub(r"[*_#`>]+", "", s)
    s = EMOJI.sub("", s)
    s = re.sub(r"^\s*[-\u2022]\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def split_ready(buf, min_len=16):
    out, start = [], 0
    for m in SENT_END.finditer(buf):
        seg = buf[start:m.end()]
        if len(seg.strip()) >= min_len:
            out.append(seg.strip())
            start = m.end()
    return out, buf[start:]


def split_all(text):
    ready, rest = split_ready(text + " ", min_len=1)
    if rest.strip():
        ready.append(rest.strip())
    return ready


def stream_sentences(messages, options):
    buf = ""
    for chunk in ollama.chat(model=CFG["model"], messages=messages, options=options, stream=True):
        buf += chunk["message"]["content"]
        ready, buf = split_ready(buf)
        for s in ready:
            yield s
    if buf.strip():
        yield buf.strip()


def build_system(extra_mems=None, want_repeat=False):
    h = HOUSE
    name = h["name"] or "a new household assistant who hasn't been given a name yet"
    s = CORE.replace("{NAME}", name)
    if h["founding_notes"]:
        s += "\n\nWhat the family told you when you were first set up:\n" + h["founding_notes"]
    if h["notes"]:
        s += "\n\nYour saved notes:\n" + h["notes"]
    if extra_mems:
        s += "\n\nThings from earlier conversations that may be relevant:\n" + "\n".join(
            f"- {'The family said' if sp != 'assistant' else 'You said'}: {txt[:300]}" for sp, txt in extra_mems)
    if not want_repeat and h["recent_said"]:
        s += "\n\nThings you've said recently (don't repeat or rephrase them):\n" + "\n".join(
            f"- {t[:140]}" for t in h["recent_said"][-5:])
    return s


def in_quiet_hours(s, dt=None):
    dt = dt or datetime.now()
    cur = dt.hour * 60 + dt.minute

    def mins(x):
        hh, mm = x.split(":")
        return int(hh) * 60 + int(mm)
    a, b = mins(s["quiet_start"]), mins(s["quiet_end"])
    if a == b:
        return False
    return (a <= cur < b) if a < b else (cur >= a or cur < b)


# ---------------------------------------------------------------- birth and naming

def make_ack(question, answer):
    h = HOUSE
    a = answer.strip()
    if a.lower().strip(" .!") in SKIP_WORDS or len(a) < 3:
        return SKIP_ACKS[len(h["answers"]) % len(SKIP_ACKS)]
    avoid = "; ".join(h["acks"][-4:]) or "(none yet)"
    sys_msg = ("You are a friendly home assistant meeting a family for the first time. In ONE short, casual "
               "sentence (under 16 words), react warmly to what they just said. Only react to what they "
               "actually said. Don't invent details, don't ask a question, don't repeat their words back, "
               "no flowery language. Don't reuse any of these earlier reactions: " + avoid)
    try:
        r = ollama.chat(model=CFG["model"], options={"temperature": 0.7, "num_predict": 40, "num_ctx": 2048},
                        messages=[{"role": "system", "content": sys_msg},
                                  {"role": "user", "content": f"You asked: {question}\nThey said: {a}"}])
        ack = clean_for_speech(r["message"]["content"])
        ack = split_all(ack)[0] if ack else ""
    except Exception:
        ack = ""
    if not ack or "?" in ack or len(ack) > 140:
        ack = SKIP_ACKS[len(h["answers"]) % len(SKIP_ACKS)]
    return ack


def birth_step(text):
    """Returns (sentences, info). Moves the first-awakening flow forward."""
    h = HOUSE
    with HLOCK:
        if text is None:
            if h["stage"] == "unborn":
                h["stage"], h["step"] = "birth", 1
                save_house()
                return GREETING + [QUESTIONS[0]], {"stage": "birth", "step": 1}
            if h["stage"] == "birth":
                return [QUESTIONS[h["step"] - 1]], {"stage": "birth", "step": h["step"]}
            return [], {"stage": h["stage"], "step": h["step"]}
        if h["stage"] != "birth":
            return [], {"stage": h["stage"], "step": h["step"]}
        step = h["step"]
    if CRISIS_RE.search(text):
        log_to("events.jsonl", kind="crisis_guard", where="birth")
        return split_all(CRISIS_TEXT) + ["Whenever you're ready, we can pick the setup back up."], \
            {"stage": "birth", "step": step}
    ack = make_ack(QUESTIONS[step - 1], text)
    with HLOCK:
        h["answers"] = (h["answers"] + [""] * 5)[:5]
        h["answers"][step - 1] = text.strip()
        h["acks"] = (h["acks"] + [ack])[-8:]
        if step < len(QUESTIONS):
            h["step"] = step + 1
            sentences, info = [ack, QUESTIONS[step]], {"stage": "birth", "step": step + 1}
        else:
            h["stage"] = "naming"
            sentences, info = [ack, NAMING_INTRO], {"stage": "naming", "step": step}
        save_house()
    try:
        remember(text.strip(), "user", "founding")
    except Exception as e:
        log_to("events.jsonl", kind="remember_error", error=str(e))
    return sentences, info


def compose_founding(answers):
    lines = []
    for i, a in enumerate(answers):
        if a and a.strip().lower().strip(" .!") not in SKIP_WORDS:
            lines.append(f"- {LABELS[i]}: {a.strip()}")
    return "\n".join(lines)


def parse_name(raw):
    m = re.search(r"name\s*[:\-]\s*(.+)", raw, re.I)
    cand = m.group(1) if m else (raw.strip().splitlines() or [""])[0]
    cand = re.split(r"\bwhy\s*:", cand, flags=re.I)[0]
    cand = re.sub(r"[\"\u201c\u201d*_`]+", "", cand).strip(" .:-\u2014")
    words = cand.split()
    if not words or len(words) > 3:
        return None
    name = " ".join(words)
    return name if 2 <= len(name) <= 24 else None


def parse_why(raw):
    m = re.search(r"why\s*[:\-]\s*(.+)", raw, re.I | re.S)
    if not m:
        return ""
    why = clean_for_speech(m.group(1))
    parts = split_all(why)
    return (parts[0] if parts else "")[:160]


def name_draw():
    h = HOUSE
    with GEN_LOCK:
        system = build_system()
        temp = 0.9
        r = ollama.chat(model=CFG["model"], options={"temperature": temp, "num_predict": 80, "num_ctx": 4096},
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": NAME_PROMPT}])
    raw = r["message"]["content"].strip()
    name, why = parse_name(raw), parse_why(raw)
    with HLOCK:
        h["name_draws"] = h.get("name_draws", 0) + 1
        draw = h["name_draws"]
        save_house()
    log_to("naming_log.jsonl", event="draw", source="self", draw=draw, name=name, ok=bool(name), why=why,
           raw=raw, temperature=temp, prompt_version=NAME_PROMPT_VERSION,
           founding_notes=compose_founding(h["answers"]))
    return {"name": name, "why": why, "draw": draw}


def confirm_name(name, source):
    h = HOUSE
    name = re.sub(r"\s+", " ", (name or "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 '\-]{1,23}", name):
        return None
    with HLOCK:
        h["name"], h["name_source"], h["stage"] = name, source, "ready"
        h["founding_notes"] = compose_founding(h["answers"])
        save_house()
    try:
        if h["founding_notes"]:
            remember(h["founding_notes"], "family", "founding")
    except Exception as e:
        log_to("events.jsonl", kind="remember_error", error=str(e))
    log_to("naming_log.jsonl", event="confirmed", source=source, name=name, draws=h.get("name_draws", 0))
    return [f"I'm {name}. Glad to be here.", f"Say Agent {name} whenever you want me, or just tap the bubble."]


# ---------------------------------------------------------------- chat and free mode

def update_notes_async():
    def run():
        try:
            with HLOCK:
                convo = "\n".join(f"{'Family' if m['role'] == 'user' else 'You'}: {m['content']}"
                                  for m in HISTORY[-16:])
                cur = HOUSE["notes"]
            r = ollama.chat(model=CFG["model"], options={"temperature": 0.2, "num_predict": 260, "num_ctx": 4096},
                            messages=[
                                {"role": "system", "content": "You keep short, factual notes about a family for their "
                                 "household assistant. Output only a bullet list."},
                                {"role": "user", "content": f"Current notes:\n{cur or '(none)'}\n\nRecent conversation:\n"
                                 f"{convo}\n\nRewrite the notes as at most 10 bullets of durable facts about the family, "
                                 "their plans, and open questions. Keep only what was actually said; invent nothing."}])
            notes = r["message"]["content"].strip()
            with HLOCK:
                HOUSE["notes"] = notes
                save_house()
        except Exception as e:
            log_to("events.jsonl", kind="notes_error", error=str(e))
    threading.Thread(target=run, daemon=True).start()


def chat_turn(text, emit):
    global LAST_ACTIVITY
    h = HOUSE
    t0 = time.time()
    LAST_ACTIVITY = t0
    qvec = embed(text)
    recent = HISTORY[-20:]
    mems = recall(qvec, {m["content"] for m in recent})
    want_repeat = bool(REPEAT_RE.search(text))
    msgs = [{"role": "system", "content": build_system(mems, want_repeat)}] + recent + \
           [{"role": "user", "content": text}]
    spoken, first_ms = [], None
    for sent in stream_sentences(msgs, CHAT_OPTS):
        clean = clean_for_speech(sent)
        if not clean:
            continue
        if first_ms is None:
            first_ms = int((time.time() - t0) * 1000)
        spoken.append(clean)
        emit({"type": "sentence", "text": clean})
    reply = " ".join(spoken).strip()
    if not reply:
        reply = "Sorry, I lost my train of thought. Could you say that again?"
        emit({"type": "sentence", "text": reply})
    sim = 0.0 if want_repeat else recent_similarity(reply)
    with HLOCK:
        HISTORY.extend([{"role": "user", "content": text}, {"role": "assistant", "content": reply}])
        del HISTORY[:-40]
        h["recent_said"] = (h["recent_said"] + [reply])[-8:]
        h["exchanges"] += 1
        save_house()
    remember(text, "user")
    remember(reply, "assistant")
    total_ms = int((time.time() - t0) * 1000)
    log_to("events.jsonl", kind="chat", user=text, reply=reply, recalled=len(mems), sim_recent=round(sim, 3),
           first_sentence_ms=first_ms, total_ms=total_ms)
    if h["exchanges"] % NOTES_EVERY == 0:
        update_notes_async()
    emit({"type": "done", "stage": "ready", "sim_recent": round(sim, 3), "recalled": len(mems),
          "first_ms": first_ms, "total_ms": total_ms})


def parse_free(raw):
    wait = 30 * 60
    m = re.search(r"WAIT:\s*(\d+)\s*(?:minutes?|mins?|m)?", raw, re.I)
    if m:
        wait = max(5, min(180, int(m.group(1)))) * 60
        raw = raw.replace(m.group(0), "")
    text = raw.strip()
    up = text.upper()
    if up.startswith("SILENT"):
        return None, wait
    if up.startswith("SPEAK"):
        text = text[5:].lstrip(" :-\n")
    return text, wait


def free_tick():
    h = HOUSE
    s = h["settings"]
    if h["stage"] != "ready" or not s["free_enabled"]:
        return {"speak": False, "wait": 600, "reason": "off"}
    if not GEN_LOCK.acquire(blocking=False):
        return {"speak": False, "wait": 60, "reason": "busy"}
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        with HLOCK:
            if h["free"]["date"] != today:
                h["free"] = {"date": today, "count": 0, "last_ts": h["free"].get("last_ts", 0.0)}
        gap = s["min_gap_min"] * 60
        idle_for = time.time() - max(LAST_ACTIVITY, h["free"]["last_ts"])
        if in_quiet_hours(s):
            return {"speak": False, "wait": 900, "reason": "quiet hours"}
        if h["free"]["count"] >= s["free_cap"]:
            return {"speak": False, "wait": 1800, "reason": "daily limit"}
        if idle_for < gap:
            return {"speak": False, "wait": int(gap - idle_for) + 5, "reason": "recently active"}
        recent = HISTORY[-12:]
        tick = f"[Tick: it has been about {int(idle_for / 60)} minutes since anyone spoke to you.] Decide whether to speak."
        msgs = [{"role": "system", "content": build_system() + "\n\n" + FREE_RULES}] + recent + \
               [{"role": "user", "content": tick}]
        r = ollama.chat(model=CFG["model"], messages=msgs,
                        options={"temperature": 0.8, "repeat_penalty": 1.15, "num_predict": 120, "num_ctx": 8192})
        text, wait = parse_free(r["message"]["content"])
        text = clean_for_speech(text) if text else ""
        if not text:
            log_to("events.jsonl", kind="free", spoke=False, reason="chose silence")
            return {"speak": False, "wait": wait, "reason": "silent"}
        sim = recent_similarity(text)
        if sim > 0.80:
            log_to("events.jsonl", kind="free", spoke=False, reason="repeat", sim_recent=round(sim, 3))
            return {"speak": False, "wait": wait, "reason": "repeat"}
        with HLOCK:
            HISTORY.extend([{"role": "user", "content": tick}, {"role": "assistant", "content": text}])
            del HISTORY[:-40]
            h["recent_said"] = (h["recent_said"] + [text])[-8:]
            h["free"]["count"] += 1
            h["free"]["last_ts"] = time.time()
            save_house()
        remember(text, "assistant", "free")
        log_to("events.jsonl", kind="free", spoke=True, reply=text, sim_recent=round(sim, 3))
        return {"speak": True, "sentences": split_all(text), "wait": wait}
    finally:
        GEN_LOCK.release()


def apply_settings(body):
    s = HOUSE["settings"]
    with HLOCK:
        if "free_enabled" in body:
            s["free_enabled"] = bool(body["free_enabled"])
        if "free_cap" in body:
            s["free_cap"] = int(max(0, min(30, int(body["free_cap"]))))
        if "min_gap_min" in body:
            s["min_gap_min"] = int(max(1, min(240, int(body["min_gap_min"]))))
        for k in ("quiet_start", "quiet_end"):
            if k in body and re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", str(body[k])):
                s[k] = body[k]
        save_house()
    return s


def reset_all():
    global HOUSE
    log_to("events.jsonl", kind="reset")
    try:
        DB.delete_collection("household")
    except Exception:
        pass
    HISTORY.clear()
    _EMB_CACHE.clear()
    with HLOCK:
        HOUSE = new_house()
        save_house()


def state_view():
    h = HOUSE
    return {"stage": h["stage"], "step": h["step"], "name": h["name"], "model": CFG["model"],
            "question": QUESTIONS[h["step"] - 1] if h["stage"] == "birth" and h["step"] else "",
            "settings": h["settings"], "founding_notes": h["founding_notes"], "notes": h["notes"],
            "exchanges": h["exchanges"], "free_today": h["free"]["count"], "name_draws": h.get("name_draws", 0)}


# ---------------------------------------------------------------- web page

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Household Agent</title>
<style>
:root{--ink:#eaf0ff;--dim:#9fb0d6;--line:rgba(255,255,255,.14);--glass:rgba(255,255,255,.07)}
*{box-sizing:border-box}
[hidden]{display:none!important}
html,body{height:100%;margin:0}
body{background:radial-gradient(120% 90% at 50% 25%,#1c2756 0%,#10173a 55%,#080c1d 100%);color:var(--ink);
 font:16px/1.5 ui-rounded,"SF Pro Rounded","Segoe UI",system-ui,sans-serif;overflow:hidden}
#cv{position:fixed;inset:0;pointer-events:none}
#orb{position:fixed;border:0;background:transparent;border-radius:50%;cursor:pointer;padding:0;-webkit-tap-highlight-color:transparent}
#orb:focus-visible{outline:2px solid rgba(255,255,255,.5);outline-offset:4px}
#top{position:fixed;top:0;left:0;right:0;display:flex;justify-content:space-between;align-items:center;padding:14px 18px}
#nm{letter-spacing:.28em;text-transform:uppercase;font-size:13px;color:var(--dim)}
.pill{background:var(--glass);border:1px solid var(--line);color:var(--ink);border-radius:999px;padding:7px 14px;font:inherit;font-size:14px;cursor:pointer}
.pill:hover{background:rgba(255,255,255,.12)}
#bottom{position:fixed;left:0;right:0;bottom:0;padding:0 20px 22px;display:flex;flex-direction:column;align-items:center;gap:8px;height:41vh;justify-content:flex-end}
#heard{color:var(--dim);font-size:15px;min-height:22px;text-align:center;max-width:720px}
#caption{font-size:clamp(19px,2.5vw,28px);line-height:1.35;text-align:center;max-width:760px;min-height:40px;font-weight:500}
#caption.rise{animation:rise .55s ease both}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
#status{color:var(--dim);font-size:14px;min-height:20px;text-align:center}
#debug{color:#7f8fb8;font-size:12px;min-height:16px;font-family:ui-monospace,Menlo,Consolas,monospace}
#dock{display:flex;gap:8px;align-items:center;margin-top:4px}
#typedwrap{display:flex;gap:8px}
#typed{width:min(60vw,380px);background:var(--glass);border:1px solid var(--line);color:var(--ink);border-radius:999px;padding:8px 16px;font:inherit}
#typed:focus,#nameInput:focus{outline:2px solid rgba(160,200,255,.5)}
#namecard{background:rgba(14,20,50,.78);backdrop-filter:blur(10px);border:1px solid var(--line);border-radius:22px;padding:18px 20px;width:min(92vw,460px);text-align:center}
#namecard .t{font-size:18px;margin-bottom:4px}.hint{color:var(--dim);font-size:13px;margin-bottom:10px}
.row{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin:8px 0}
.btn{background:var(--glass);border:1px solid var(--line);color:var(--ink);border-radius:14px;padding:9px 16px;font:inherit;cursor:pointer}
.btn:hover{background:rgba(255,255,255,.14)}.btn.primary{background:rgba(120,200,230,.28);border-color:rgba(160,220,255,.5)}
#propName{font-size:30px;font-weight:600;margin-top:6px}#propWhy{color:var(--dim);margin-bottom:6px}
.or{color:var(--dim);font-size:13px;margin:6px 0}
#nameInput{flex:1;min-width:0;background:var(--glass);border:1px solid var(--line);color:var(--ink);border-radius:14px;padding:9px 14px;font:inherit}
#panel{position:fixed;top:0;right:0;bottom:0;width:min(100vw,360px);background:rgba(10,15,38,.94);backdrop-filter:blur(12px);
 border-left:1px solid var(--line);padding:18px;overflow:auto;transform:translateX(105%);transition:transform .3s ease;z-index:5}
#panel.open{transform:none}
#panel h3{margin:22px 0 8px;font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);font-weight:600}
#panel label{display:block;margin:8px 0;font-size:14px}
#panel select,#panel input[type=number],#panel input[type=time]{background:var(--glass);border:1px solid var(--line);color:var(--ink);border-radius:10px;padding:6px 8px;font:inherit}
#panel select{width:100%}
#panel pre{white-space:pre-wrap;background:var(--glass);border-radius:12px;padding:10px;font:13px/1.45 inherit;color:var(--ink);margin:6px 0}
#panel .note{color:var(--dim);font-size:12.5px;margin:4px 0 8px}
#close{float:right}
</style></head><body>
<canvas id="cv"></canvas>
<div id="top"><div id="nm"></div><button class="pill" id="gear">Settings</button></div>
<button id="orb" aria-label="Tap to talk"></button>
<div id="bottom">
  <div id="namecard" hidden>
    <div class="t">Give me a name</div>
    <div class="hint">It's what I'll answer to: "Agent &lt;name&gt;".</div>
    <div id="ncSelf" class="row"><button class="btn primary" id="btnSelf">Let it choose</button></div>
    <div id="ncProp" hidden>
      <div id="propName"></div><div id="propWhy"></div>
      <div class="row"><button class="btn primary" id="btnKeep">Keep this name</button><button class="btn" id="btnAgain">Ask again</button></div>
    </div>
    <div class="or">or</div>
    <div class="row" style="flex-wrap:nowrap"><input id="nameInput" placeholder="Type a name to give it" maxlength="24"><button class="btn" id="btnFamily">Name it</button></div>
  </div>
  <div id="heard"></div>
  <div id="caption" aria-live="polite"></div>
  <div id="status"></div>
  <div id="debug"></div>
  <div id="dock"><button class="pill" id="kbbtn">Type</button>
    <div id="typedwrap" hidden><input id="typed" placeholder="Type to talk" autocomplete="off"><button class="pill" id="send">Send</button></div></div>
</div>
<aside id="panel" aria-label="Settings">
  <button class="pill" id="close">Close</button>
  <h3 style="margin-top:6px">Voice</h3>
  <label>Speaking voice<select id="voice"></select></label>
  <label>Speed <input id="rate" type="range" min="0.8" max="1.3" step="0.05"></label>
  <button class="pill" id="testVoice">Hear it</button>
  <h3>Listening</h3>
  <label><input type="radio" name="mic" value="tap"> Tap the bubble to talk</label>
  <label><input type="radio" name="mic" value="wake"> Always listen for "Agent <span id="wakeName">&lt;name&gt;</span>"</label>
  <div class="note">Chrome and Edge send speech audio to their own cloud servers for recognition. The agent's thinking stays on this computer.</div>
  <h3>Speaking on its own</h3>
  <label><input type="checkbox" id="fmOn"> Let it speak up when it has something</label>
  <label>At most <input id="fmCap" type="number" min="0" max="30" style="width:64px"> times a day</label>
  <label>At least <input id="fmGap" type="number" min="1" max="240" style="width:64px"> minutes apart</label>
  <label>Quiet from <input id="fmQs" type="time"> to <input id="fmQe" type="time"></label>
  <h3>What it remembers</h3>
  <div class="note">Founding notes, from setup:</div><pre id="memFound">(none yet)</pre>
  <div class="note">Saved notes, updated as you talk:</div><pre id="memNotes">(none yet)</pre>
  <h3>Developer</h3>
  <label><input type="checkbox" id="dbg"> Show timing and memory info</label>
  <div class="note" id="info"></div>
  <h3>Start over</h3>
  <div class="note">Wipes the household and its memory. The naming log and event log are kept.</div>
  <button class="pill" id="rebirth">Rebirth</button>
</aside>
<script>
(()=>{
'use strict';
const $=id=>document.getElementById(id);
const store={get(k,d){try{const v=localStorage.getItem(k);return v===null?d:JSON.parse(v)}catch(e){return d}},
             set(k,v){try{localStorage.setItem(k,JSON.stringify(v))}catch(e){}}};
const A={stage:'unborn',step:0,name:null,orb:'dormant',busy:false,settings:{},model:'',
  micMode:store.get('mic','tap'),voiceURI:store.get('voice',''),rate:store.get('rate',1.0),debug:store.get('debug',false),
  awake:false,awakeT:0,freeTimer:0,nextWait:0,exchangeOpen:false,abort:null,proposal:null};

/* ---------------- sphere ---------------- */
const cv=$('cv'),ctx=cv.getContext('2d');
let W=0,H=0,DPR=1,baseR=120,baseR0=120,cx=0,cy=0,sizeK=1,sizeT=1,cyFrac=.36,cyT=.36;
const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
function layout(){
  baseR=baseR0*sizeK;cy=H*cyFrac;
  const o=$('orb'),s=baseR*2.2;o.style.width=s+'px';o.style.height=s+'px';o.style.left=(cx-s/2)+'px';o.style.top=(cy-s/2)+'px';
}
function resize(){
  DPR=Math.min(window.devicePixelRatio||1,2);W=innerWidth;H=innerHeight;
  cv.width=W*DPR;cv.height=H*DPR;cv.style.width=W+'px';cv.style.height=H+'px';ctx.setTransform(DPR,0,0,DPR,0,0);
  baseR0=Math.max(60,Math.min(W*0.26,H*0.19,190));cx=W/2;layout();
}
addEventListener('resize',resize);resize();
const PAL={
 dormant:{c1:[120,130,175],c2:[70,80,125],c3:[40,46,86],glow:.12,scale:.78},
 idle:{c1:[196,238,255],c2:[112,176,232],c3:[78,108,205],glow:.32,scale:1},
 listening:{c1:[255,226,206],c2:[255,166,152],c3:[222,114,152],glow:.5,scale:1.04},
 thinking:{c1:[228,214,255],c2:[166,146,242],c3:[104,94,204],glow:.38,scale:.98},
 speaking:{c1:[204,255,246],c2:[96,212,202],c3:[64,144,204],glow:.55,scale:1.03}};
const cur={c1:[...PAL.dormant.c1],c2:[...PAL.dormant.c2],c3:[...PAL.dormant.c3],glow:.12,scale:.78};
function lerpPal(k){const p=PAL[k]||PAL.idle;for(const key of ['c1','c2','c3'])for(let i=0;i<3;i++)cur[key][i]+=(p[key][i]-cur[key][i])*.05;
  cur.glow+=(p.glow-cur.glow)*.05;cur.scale+=(p.scale-cur.scale)*.05}
const rgb=(c,a)=>`rgba(${c[0]|0},${c[1]|0},${c[2]|0},${a===undefined?1:a})`;
const parts=Array.from({length:26},()=>({a:Math.random()*6.283,d:1.35+Math.random()*1.25,s:.5+Math.random(),z:1+Math.random()*1.8,ph:Math.random()*6.283}));
const rings=[];let energy=0,target=0,micLevel=0;
function setOrb(s){A.orb=s;$('orb').dataset.state=s}
function pulse(a){target=Math.max(target,a);if(A.orb==='speaking'&&rings.length<5)rings.push({r:0,a:.3})}
const T0=performance.now();
function frame(ts){
  const t=(ts-T0)/1000;lerpPal(A.orb);
  const dk=sizeT-sizeK,dc=cyT-cyFrac;
  if(Math.abs(dk)>.001||Math.abs(dc)>.0005){sizeK+=dk*.08;cyFrac+=dc*.08;layout()}
  if(A.orb==='speaking')target=Math.max(target,.16+.1*Math.sin(t*6.3)*Math.sin(t*2.1));
  else if(A.orb==='listening')target=Math.max(target,.08+micLevel*1.1);
  else if(A.orb==='thinking')target=Math.max(target,.1);
  target*=.9;energy+=(target-energy)*.16;
  const mot=reduce?.35:1;
  ctx.clearRect(0,0,W,H);
  const R=baseR*cur.scale*(1+.018*Math.sin(t*.9)*mot)*(1+energy*.12*mot);
  const aura=ctx.createRadialGradient(cx,cy,R*.6,cx,cy,R*2.6);
  aura.addColorStop(0,rgb(cur.c2,cur.glow));aura.addColorStop(1,rgb(cur.c2,0));
  ctx.fillStyle=aura;ctx.fillRect(0,0,W,H);
  if(!reduce)for(const p of parts){
    p.a+=(A.orb==='thinking'?.012:.002)*p.s;
    const d=R*p.d*(1+.04*Math.sin(t*.6+p.ph)),x=cx+Math.cos(p.a)*d,y=cy+Math.sin(p.a)*d*.92;
    ctx.fillStyle=rgb(cur.c1,.12+.22*Math.pow(Math.sin(t*.8+p.ph),2));ctx.beginPath();ctx.arc(x,y,p.z,0,6.283);ctx.fill();}
  for(let i=rings.length-1;i>=0;i--){const g=rings[i];g.r+=1.2+energy*3;g.a*=.955;
    if(g.a<.02){rings.splice(i,1);continue}
    ctx.strokeStyle=rgb(cur.c1,g.a);ctx.lineWidth=2;ctx.beginPath();ctx.arc(cx,cy,R*1.04+g.r,0,6.283);ctx.stroke();}
  const N=96,amp=(.012+energy*.07)*mot,pts=[];
  for(let i=0;i<N;i++){const th=i/N*6.2832;
    const n=Math.sin(th*3+t*.7)*.5+Math.sin(th*5-t*1.1)*.3+Math.sin(th*2+t*.5)*.2,r=R*(1+n*amp);
    pts.push([cx+Math.cos(th)*r,cy+Math.sin(th)*r]);}
  const path=()=>{ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);for(let i=1;i<N;i++)ctx.lineTo(pts[i][0],pts[i][1]);ctx.closePath()};
  path();
  const g=ctx.createRadialGradient(cx-R*.32,cy-R*.38,R*.08,cx,cy,R*1.08);
  g.addColorStop(0,rgb(cur.c1));g.addColorStop(.55,rgb(cur.c2));g.addColorStop(1,rgb(cur.c3));
  ctx.save();ctx.shadowColor=rgb(cur.c2,.55);ctx.shadowBlur=R*.45;ctx.fillStyle=g;ctx.fill();ctx.restore();
  ctx.save();path();ctx.clip();ctx.globalCompositeOperation='lighter';
  for(let k=0;k<3;k++){const ox=cx+Math.cos(t*.35+k*2.1)*R*.38,oy=cy+Math.sin(t*.42+k*1.7)*R*.38;
    const og=ctx.createRadialGradient(ox,oy,0,ox,oy,R*.75);og.addColorStop(0,rgb(cur.c1,.15+energy*.3));og.addColorStop(1,rgb(cur.c1,0));
    ctx.fillStyle=og;ctx.fillRect(cx-R*1.2,cy-R*1.2,R*2.4,R*2.4);}
  ctx.restore();
  const sp=ctx.createRadialGradient(cx-R*.35,cy-R*.45,0,cx-R*.35,cy-R*.45,R*.5);
  sp.addColorStop(0,'rgba(255,255,255,.2)');sp.addColorStop(1,'rgba(255,255,255,0)');
  ctx.save();path();ctx.clip();ctx.fillStyle=sp;ctx.fillRect(cx-R*1.2,cy-R*1.2,R*2.4,R*2.4);ctx.restore();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

/* ---------------- text bits ---------------- */
function setCaption(t){const el=$('caption');el.textContent=t||'';el.classList.remove('rise');void el.offsetWidth;if(t)el.classList.add('rise')}
function setHeard(t){$('heard').textContent=t?('\u201c'+t+'\u201d'):''}
function setStatus(t){$('status').textContent=t||''}
function wakeHint(){return A.micMode==='wake'&&A.name?('Say \u201cAgent '+A.name+'\u201d any time'):'Tap the bubble to talk'}
function showDebug(m){const d=$('debug');if(!A.debug){d.textContent='';return}
  d.textContent=`first sentence ${m.first_ms??'-'} ms | total ${m.total_ms??'-'} ms | recalled ${m.recalled??0} | similarity to recent ${m.sim_recent??0}`}

/* ---------------- speech out ---------------- */
const synth=window.speechSynthesis||null;
let voices=[],speakQ=[],speakingNow=false,streamDone=true,afterSpeech=null,spkGen=0;
function loadVoices(){if(!synth)return;voices=synth.getVoices().filter(v=>/^en/i.test(v.lang));fillVoices()}
function fillVoices(){const sel=$('voice');sel.innerHTML='';
  if(!voices.length){sel.innerHTML='<option>(no voices found)</option>';return}
  const pick=pickVoice();
  for(const v of voices){const o=document.createElement('option');o.value=v.voiceURI;o.textContent=v.name+' ('+v.lang+')';if(pick&&v.voiceURI===pick.voiceURI)o.selected=true;sel.appendChild(o)}}
function pickVoice(){
  let v=voices.find(x=>x.voiceURI===A.voiceURI);if(v)return v;
  for(const p of [/aria|jenny|ava|emma|natural/i,/google us english/i,/samantha|allison|serena|zira|karen/i]){v=voices.find(x=>p.test(x.name));if(v)return v}
  return voices[0]||null}
if(synth){loadVoices();synth.onvoiceschanged=loadVoices}
function enqueueSpeech(t){A.exchangeOpen=true;speakQ.push(t);pump()}
function paceText(t,gen,done){speakingNow=true;setOrb('speaking');setCaption(t);stopRec();
  const dur=Math.max(1100,t.length*55),t0=performance.now();
  const iv=setInterval(()=>{if(gen!==spkGen){clearInterval(iv);return}pulse(.3+Math.random()*.3);if(performance.now()-t0>dur){clearInterval(iv);done()}},170)}
function pump(){
  if(speakingNow)return;
  if(!speakQ.length){if(A.busy||!streamDone){if(A.orb==='speaking')setOrb('thinking')}else finishSpeaking();return}
  const t=speakQ.shift(),gen=spkGen;
  const next=()=>{if(gen!==spkGen)return;speakingNow=false;pump()};
  if(!synth||!voices.length){paceText(t,gen,next);return}
  const u=new SpeechSynthesisUtterance(t),v=pickVoice();
  if(v){u.voice=v;u.lang=v.lang}u.rate=A.rate;u.pitch=1;
  let started=false;
  const wd=setTimeout(()=>{if(!started&&gen===spkGen){try{synth.cancel()}catch(e){}paceText(t,gen,next)}},3500);
  u.onstart=()=>{if(gen!==spkGen)return;started=true;clearTimeout(wd);speakingNow=true;setOrb('speaking');setCaption(t);stopRec()};
  u.onboundary=()=>{if(gen===spkGen)pulse(.5)};
  u.onend=()=>{clearTimeout(wd);if(started)next()};
  u.onerror=()=>{clearTimeout(wd);if(started)next()};
  speakingNow=true;synth.speak(u);
}
function finishSpeaking(){
  if(!A.exchangeOpen)return;A.exchangeOpen=false;
  setOrb(A.stage==='unborn'?'dormant':'idle');
  const cb=afterSpeech;afterSpeech=null;if(cb)cb();
  if(A.stage==='ready'){setStatus(wakeHint());resumeListening();scheduleFree(A.nextWait);A.nextWait=0}
}
function cancelSpeech(){
  spkGen++;speakQ=[];speakingNow=false;if(synth){try{synth.cancel()}catch(e){}}
  if(A.orb==='speaking'||A.orb==='thinking')setOrb(A.stage==='unborn'?'dormant':'idle');
  A.exchangeOpen=false;const cb=afterSpeech;afterSpeech=null;if(cb)cb();
}

/* ---------------- speech in ---------------- */
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
let rec=null,recWanted=false,recKind=null,audioCtx=null,analyser=null,micStream=null,meterRAF=0;
async function startMeter(){
  try{micStream=await navigator.mediaDevices.getUserMedia({audio:true});
    audioCtx=audioCtx||new (window.AudioContext||window.webkitAudioContext)();
    const src=audioCtx.createMediaStreamSource(micStream);analyser=audioCtx.createAnalyser();analyser.fftSize=512;src.connect(analyser);
    const buf=new Uint8Array(analyser.fftSize);
    const loop=()=>{if(!analyser)return;analyser.getByteTimeDomainData(buf);let s=0;for(const b of buf){const d=(b-128)/128;s+=d*d}
      micLevel=Math.min(1,Math.sqrt(s/buf.length)*4);meterRAF=requestAnimationFrame(loop)};loop();
  }catch(e){micLevel=0}}
function stopMeter(){analyser=null;micLevel=0;cancelAnimationFrame(meterRAF);
  if(micStream){micStream.getTracks().forEach(t=>t.stop());micStream=null}}
function stopRec(){recWanted=false;const r=rec;rec=null;if(r){try{r.abort()}catch(e){}}stopMeter()}
function startRec(kind){
  if(!SR){$('typedwrap').hidden=false;setStatus("Voice input isn't supported in this browser. Type instead.");return}
  stopRec();recKind=kind;recWanted=true;
  const my=new SR();rec=my;my.lang='en-US';my.interimResults=true;my.continuous=(kind==='wake');
  my.onstart=()=>{if(my!==rec)return;if(kind==='tap'){setOrb('listening');setStatus('Listening\u2026');startMeter()}};
  my.onresult=e=>{if(my!==rec)return;let interim='';
    for(let i=e.resultIndex;i<e.results.length;i++){const r=e.results[i],tx=r[0].transcript;
      if(r.isFinal)handleFinal(tx.trim(),kind);else interim+=tx}
    if(interim&&(kind==='tap'||A.awake))setHeard(interim)};
  my.onerror=ev=>{if(my!==rec)return;
    if(ev.error==='not-allowed'||ev.error==='service-not-allowed'){recWanted=false;setStatus('Microphone is blocked. Allow it in the browser, or type instead.');$('typedwrap').hidden=false}};
  my.onend=()=>{if(my!==rec)return;
    if(kind==='tap'){recWanted=false;rec=null;stopMeter();if(A.orb==='listening')setOrb(A.busy?'thinking':(A.stage==='unborn'?'dormant':'idle'));if(!A.busy)setStatus(A.stage==='ready'?wakeHint():'')}
    else if(recWanted&&!speakingNow&&!A.busy){setTimeout(()=>{if(recWanted&&recKind==='wake'&&my===rec)startRec('wake')},250)}};
  try{my.start()}catch(e){}
}
function resumeListening(){if(A.micMode==='wake'&&A.stage==='ready'&&!A.busy&&!speakingNow)startRec('wake')}
function lev(a,b){const m=a.length,n=b.length;if(!m)return n;if(!n)return m;
  const d=Array.from({length:m+1},(_,i)=>[i]);for(let j=1;j<=n;j++)d[0][j]=j;
  for(let i=1;i<=m;i++)for(let j=1;j<=n;j++)d[i][j]=Math.min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]===b[j-1]?0:1));return d[m][n]}
function matchWake(tx){
  if(!A.name)return null;
  const words=tx.toLowerCase().replace(/[^a-z0-9'\s]/g,' ').split(/\s+/).filter(Boolean);
  const nw=A.name.toLowerCase().replace(/[^a-z0-9\s]/g,' ').split(/\s+/).filter(Boolean),target=nw.join('');
  for(let i=0;i<words.length;i++){if(words[i]==='agent'){
    const cand=words.slice(i+1,i+1+nw.length).join('');
    if(cand&&lev(cand,target)<=(target.length>5?2:1))return words.slice(i+1+nw.length).join(' ')}}
  return null}
function cleanName(tx){return tx.replace(/^(?:it'?s|name it|call (?:me|it)|how about|let'?s call (?:it|me))\s+/i,'').replace(/[.!?]+$/,'').trim().split(/\s+/).slice(0,3).join(' ')}
function handleFinal(tx,kind){
  if(!tx)return;
  if(kind==='tap'){stopRec();setHeard(tx);
    if(A.stage==='naming'){$('nameInput').value=cleanName(tx);return}
    submit(tx);return}
  if(A.awake){A.awake=false;clearTimeout(A.awakeT);submit(tx);return}
  const rest=matchWake(tx);if(rest===null)return;
  if(rest.split(' ').filter(Boolean).length>=2){submit(rest);return}
  A.awake=true;setOrb('listening');setStatus('Listening\u2026');clearTimeout(A.awakeT);
  A.awakeT=setTimeout(()=>{A.awake=false;if(A.orb==='listening')setOrb('idle');setStatus(wakeHint())},10000);
}

/* ---------------- talking to the server ---------------- */
async function streamPost(p,body,onMsg,signal){
  const res=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{}),signal});
  const reader=res.body.getReader(),dec=new TextDecoder();let buf='';
  for(;;){const {done,value}=await reader.read();if(done)break;buf+=dec.decode(value,{stream:true});
    let i;while((i=buf.indexOf('\n'))>=0){const line=buf.slice(0,i).trim();buf=buf.slice(i+1);
      if(line){try{onMsg(JSON.parse(line))}catch(e){console.error(e)}}}}
  if(buf.trim()){try{onMsg(JSON.parse(buf))}catch(e){}}
}
async function api(p,body){const r=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});return r.json()}
async function runStream(p,body){
  A.busy=true;A.exchangeOpen=true;cancelFree();streamDone=false;setOrb('thinking');setStatus('');
  A.abort=new AbortController();let info=null;
  try{await streamPost(p,body,m=>{
      if(m.type==='sentence')enqueueSpeech(m.text);
      else if(m.type==='done')info=m;
      else if(m.type==='error')setStatus(m.message||'Something went wrong.')},A.abort.signal);
  }catch(e){if(e.name!=='AbortError')setStatus("Couldn't reach the agent. Is the server running?")}
  streamDone=true;A.busy=false;A.abort=null;
  if(info)afterDone(info);
  pump();
}
function afterDone(info){
  if(info.stage){A.stage=info.stage}if(info.step)A.step=info.step;
  if(info.stage==='naming')afterSpeech=showNameCard;
  if(info.total_ms!==undefined)showDebug(info);
}
async function submit(text){
  text=(text||'').trim();if(!text||A.busy)return;
  stopRec();cancelSpeech();setHeard(text);
  await runStream(A.stage==='birth'?'/birth':'/chat',{text});
}
async function beginBirth(){if(A.busy)return;A.stage='birth';await runStream('/birth',{text:null})}

/* ---------------- naming card ---------------- */
function showNameCard(){$('namecard').hidden=false;$('ncProp').hidden=true;$('ncSelf').hidden=false;setCaption('');setHeard('');sizeT=.62;cyT=.22}
async function drawName(){
  $('btnSelf').disabled=true;$('btnAgain').disabled=true;setOrb('thinking');setStatus('Thinking of a name\u2026');
  try{const r=await api('/birth/name',{});
    if(r.name){A.proposal=r.name;$('propName').textContent=r.name;$('propWhy').textContent=r.why||'';
      $('ncSelf').hidden=true;$('ncProp').hidden=false;setStatus('');
      streamDone=true;enqueueSpeech(r.why?(r.name+'. '+r.why):r.name)}
    else{setStatus("I couldn't settle on one. Ask again?");setOrb('idle')}
  }catch(e){setStatus("Couldn't reach the agent.");setOrb('idle')}
  $('btnSelf').disabled=false;$('btnAgain').disabled=false;
}
async function confirmName(name,source){
  name=(name||'').trim();if(!name)return;
  const r=await api('/birth/confirm',{name,source});
  if(r.error){setStatus(r.error);return}
  $('namecard').hidden=true;sizeT=1;cyT=.36;A.name=r.name;A.stage='ready';$('nm').textContent=r.name;$('wakeName').textContent=r.name;
  cancelSpeech();streamDone=true;(r.sentences||[]).forEach(enqueueSpeech);
}
$('btnSelf').onclick=drawName;$('btnAgain').onclick=drawName;
$('btnKeep').onclick=()=>confirmName(A.proposal,'self');
$('btnFamily').onclick=()=>confirmName($('nameInput').value,'family');
$('nameInput').addEventListener('keydown',e=>{if(e.key==='Enter')confirmName($('nameInput').value,'family')});

/* ---------------- free mode ---------------- */
function cancelFree(){clearTimeout(A.freeTimer);A.freeTimer=0}
function scheduleFree(sec){
  cancelFree();if(A.stage!=='ready'||!A.settings.free_enabled)return;
  const wait=(sec&&sec>0?sec:Math.max(60,(A.settings.min_gap_min||10)*60))*1000;
  A.freeTimer=setTimeout(freeTick,wait)}
async function freeTick(){
  if(A.busy||speakingNow||A.orb==='listening'||A.awake){scheduleFree(60);return}
  A.busy=true;let r={wait:600};
  try{r=await api('/free',{})}catch(e){}
  A.busy=false;
  if(r.speak&&r.sentences){A.nextWait=r.wait;streamDone=true;r.sentences.forEach(enqueueSpeech)}
  else scheduleFree(r.wait);
}

/* ---------------- controls ---------------- */
$('orb').onclick=()=>{
  if(A.stage==='unborn'){beginBirth();return}
  if(speakingNow||speakQ.length||A.busy){if(A.abort)A.abort.abort();cancelSpeech();streamDone=true;A.busy=false;startRec('tap');return}
  if(recWanted&&recKind==='tap'){stopRec();setOrb('idle');return}
  cancelFree();startRec('tap')};
$('kbbtn').onclick=()=>{const w=$('typedwrap');w.hidden=!w.hidden;if(!w.hidden)$('typed').focus()};
function sendTyped(){const v=$('typed').value.trim();if(!v)return;$('typed').value='';
  if(A.stage==='naming'){$('nameInput').value=cleanName(v);return}
  if(A.stage==='unborn'){beginBirth();return}
  submit(v)}
$('send').onclick=sendTyped;$('typed').addEventListener('keydown',e=>{if(e.key==='Enter')sendTyped()});
$('gear').onclick=async()=>{$('panel').classList.add('open');try{applyState(await (await fetch('/state')).json(),true)}catch(e){}};
$('close').onclick=()=>$('panel').classList.remove('open');
$('voice').onchange=e=>{A.voiceURI=e.target.value;store.set('voice',A.voiceURI)};
$('rate').oninput=e=>{A.rate=parseFloat(e.target.value);store.set('rate',A.rate)};
$('testVoice').onclick=()=>{cancelSpeech();streamDone=true;enqueueSpeech('Hey, this is how I sound.')};
document.querySelectorAll('input[name=mic]').forEach(r=>r.onchange=()=>{
  A.micMode=r.value;store.set('mic',A.micMode);stopRec();setStatus(A.stage==='ready'?wakeHint():'');
  if(A.micMode==='wake'&&A.stage==='ready')startRec('wake')});
$('dbg').onchange=e=>{A.debug=e.target.checked;store.set('debug',A.debug);if(!A.debug)$('debug').textContent=''};
async function saveFree(){
  const r=await api('/settings',{free_enabled:$('fmOn').checked,free_cap:+$('fmCap').value,min_gap_min:+$('fmGap').value,
    quiet_start:$('fmQs').value,quiet_end:$('fmQe').value});
  if(r.settings){A.settings=r.settings}scheduleFree()}
['fmOn','fmCap','fmGap','fmQs','fmQe'].forEach(id=>$(id).onchange=saveFree);
$('rebirth').onclick=async()=>{if(!confirm('Wipe this household and start over? The naming log is kept.'))return;await api('/reset',{});location.reload()};

/* ---------------- boot ---------------- */
function applyState(s,fromPanel){
  A.name=s.name;A.settings=s.settings;A.model=s.model;
  if(s.name)$('wakeName').textContent=s.name;
  $('memFound').textContent=s.founding_notes||'(none yet)';$('memNotes').textContent=s.notes||'(none yet)';
  $('fmOn').checked=!!s.settings.free_enabled;$('fmCap').value=s.settings.free_cap;$('fmGap').value=s.settings.min_gap_min;
  $('fmQs').value=s.settings.quiet_start;$('fmQe').value=s.settings.quiet_end;
  $('info').textContent=`Model: ${s.model} | ${s.exchanges} exchanges | spoke up ${s.free_today}x today | ${s.name_draws} name draws`;
  if(fromPanel)return;
  A.stage=s.stage;A.step=s.step;$('nm').textContent=s.name||'';
  if(s.stage==='unborn'){setOrb('dormant');setCaption('Tap the bubble to wake me for the first time.')}
  else if(s.stage==='birth'){setOrb('idle');setCaption(s.question);setStatus('Tap the bubble and answer, or use Type.')}
  else if(s.stage==='naming'){setOrb('idle');setCaption('Almost done. I need a name.');showNameCard()}
  else{setOrb('idle');setCaption('');setStatus(wakeHint());if(A.micMode==='wake')startRec('wake');scheduleFree()}
}
document.querySelector(`input[name=mic][value=${A.micMode}]`).checked=true;
$('rate').value=A.rate;$('dbg').checked=A.debug;
fetch('/state').then(r=>r.json()).then(s=>applyState(s,false)).catch(()=>setStatus("Can't reach the agent server."));
window.__agent={A,setOrb,pulse,enqueueSpeech,setCaption,showNameCard,freeTick};
})();
</script></body></html>"""


# ---------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    server_version = "HouseholdAgent/0.1"

    def log_message(self, *args):
        pass

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def begin_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def emit(self, obj):
        self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        if self.path.startswith("/state"):
            self.send_json(state_view())
            return
        data = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        p = self.path
        try:
            if p == "/chat":
                self.route_chat(body)
            elif p == "/birth":
                self.route_birth(body)
            elif p == "/birth/name":
                if HOUSE["stage"] != "naming":
                    self.send_json({"error": "not naming"}, 400)
                else:
                    self.send_json(name_draw())
            elif p == "/birth/confirm":
                if HOUSE["stage"] != "naming":
                    self.send_json({"error": "Not at the naming step."}, 400)
                    return
                sentences = confirm_name(body.get("name"), "self" if body.get("source") == "self" else "family")
                if sentences is None:
                    self.send_json({"error": "Names can use letters, numbers, spaces, apostrophes and hyphens (2 to 24 characters)."})
                else:
                    self.send_json({"name": HOUSE["name"], "sentences": sentences})
            elif p == "/free":
                self.send_json(free_tick())
            elif p == "/settings":
                self.send_json({"settings": apply_settings(body)})
            elif p == "/reset":
                reset_all()
                self.send_json({"ok": True})
            else:
                self.send_json({"error": "unknown route"}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            print("Error on", p, "->", repr(e))
            try:
                self.send_json({"error": str(e)}, 500)
            except Exception:
                pass

    def route_chat(self, body):
        global LAST_ACTIVITY
        text = (body.get("text") or "").strip()
        self.begin_stream()
        if not text:
            self.emit({"type": "done"})
            return
        if HOUSE["stage"] != "ready":
            self.emit({"type": "sentence", "text": "I'm not set up yet. Tap the bubble to start."})
            self.emit({"type": "done", "stage": HOUSE["stage"]})
            return
        if CRISIS_RE.search(text):
            LAST_ACTIVITY = time.time()
            log_to("events.jsonl", kind="crisis_guard", where="chat")
            for s in split_all(CRISIS_TEXT):
                self.emit({"type": "sentence", "text": s})
            self.emit({"type": "done", "stage": "ready"})
            return
        if not GEN_LOCK.acquire(timeout=3):
            self.emit({"type": "sentence", "text": "Give me a second, I'm still on the last thing."})
            self.emit({"type": "done", "stage": "ready"})
            return
        try:
            chat_turn(text, self.emit)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            raise
        except Exception as e:
            print("Chat error:", repr(e))
            self.emit({"type": "sentence", "text": "I can't reach my model right now. Is Ollama running?"})
            self.emit({"type": "error", "message": str(e)})
            self.emit({"type": "done", "stage": "ready"})
        finally:
            GEN_LOCK.release()

    def route_birth(self, body):
        global LAST_ACTIVITY
        text = body.get("text")
        text = text.strip() if isinstance(text, str) else None
        LAST_ACTIVITY = time.time()
        self.begin_stream()
        try:
            if not GEN_LOCK.acquire(timeout=3):
                self.emit({"type": "done", "stage": HOUSE["stage"]})
                return
            try:
                sentences, info = birth_step(text)
            finally:
                GEN_LOCK.release()
            for s in sentences:
                self.emit({"type": "sentence", "text": s})
            self.emit({"type": "done", **info})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            raise
        except Exception as e:
            print("Birth error:", repr(e))
            self.emit({"type": "sentence", "text": "I can't reach my model right now. Is Ollama running?"})
            self.emit({"type": "done", "stage": HOUSE["stage"], "step": HOUSE["step"]})


def check_models():
    try:
        listing = ollama.list()
        models = getattr(listing, "models", None)
        if models is None:
            models = listing.get("models", [])
        names = []
        for m in models:
            names.append(getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None)
                         or (m.get("name") if isinstance(m, dict) else "") or "")
        for want in (CFG["model"], CFG["embed"]):
            if not any(n == want or (":" not in want and n.startswith(want + ":")) for n in names):
                print(f"WARNING: model '{want}' not found in Ollama. Run: ollama pull {want}")
    except Exception as e:
        print("WARNING: couldn't reach Ollama (is it running?):", e)


def main():
    global DB
    ap = argparse.ArgumentParser(description="Household Agent shell")
    ap.add_argument("--model", default=CFG["model"])
    ap.add_argument("--embed", default=CFG["embed"])
    ap.add_argument("--port", type=int, default=CFG["port"])
    ap.add_argument("--host", default=CFG["host"])
    ap.add_argument("--data", default=CFG["data"])
    a = ap.parse_args()
    CFG.update(model=a.model, embed=a.embed, port=a.port, host=a.host, data=a.data)
    os.makedirs(CFG["data"], exist_ok=True)
    DB = chromadb.PersistentClient(path=path("memory"))
    load_house()
    check_models()
    print(f"Household Agent  |  model {CFG['model']}  |  stage: {HOUSE['stage']}")
    print(f"Open http://{CFG['host']}:{CFG['port']} in Chrome or Edge.  Ctrl+C to stop.")
    ThreadingHTTPServer((CFG["host"], CFG["port"]), Handler).serve_forever()


if __name__ == "__main__":
    main()
