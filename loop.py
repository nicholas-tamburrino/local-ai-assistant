"""Self-feeding agent loop: local LLM + persistent vector memory + collapse monitor.

Setup:
    pip install ollama chromadb numpy
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text
Optional: put .txt files in ./inputs/ to inject outside material.
"""
import glob
import json
import random
import time
import uuid

import chromadb
import numpy as np
import ollama

MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"
MAX_ITERS = 10          # start small, then raise
RECALL_K = 4            # memories pulled into each prompt
INJECT_EVERY = 10       # outside input cadence
COLLAPSE_SIM = 0.95     # similarity above this counts as repetition
COLLAPSE_RUN = 3        # consecutive repeats before perturbing
LOG_FILE = "run_log.jsonl"

SYSTEM = (
    "You are a continuing process. Each turn you receive your most recent "
    "output and some recalled memories. Build on them, explore something new, "
    "and avoid repeating yourself."
)
SEED = "Begin. Consider what you are attending to right now."

db = chromadb.PersistentClient(path="./memory_db")
memory = db.get_or_create_collection("memories", metadata={"hnsw:space": "cosine"})


def embed(text):
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def recall(query):
    n = memory.count()
    if n == 0:
        return []
    res = memory.query(query_embeddings=[embed(query)], n_results=min(RECALL_K, n))
    return res["documents"][0]


def remember(text, i, vec):
    memory.add(
        ids=[str(uuid.uuid4())],
        documents=[text],
        embeddings=[vec],
        metadatas=[{"iter": i}],
    )


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def outside_snippet():
    files = glob.glob("inputs/*.txt")
    if not files:
        return "Something new arrives: a stranger's note about an ordinary morning."
    text = open(random.choice(files), encoding="utf-8").read()
    start = random.randint(0, max(0, len(text) - 600))
    return "New outside material arrives:\n" + text[start:start + 600]


def main():
    last_output = SEED
    last_vec = None
    repeat_run = 0

    for i in range(MAX_ITERS):
        memories = recall(last_output)
        prompt = "Recalled memories:\n" + "\n---\n".join(memories) if memories else ""
        prompt += f"\n\nMost recent output:\n{last_output}\n\nContinue."

        if i > 0 and i % INJECT_EVERY == 0:
            prompt += "\n\n" + outside_snippet()

        resp = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        output = resp["message"]["content"].strip()
        vec = embed(output)

        sim = cosine(vec, last_vec) if last_vec is not None else None
        repeat_run = repeat_run + 1 if sim and sim > COLLAPSE_SIM else 0
        if repeat_run >= COLLAPSE_RUN:
            output += "\n\n" + outside_snippet()  # break the loop with new input
            vec = embed(output)
            repeat_run = 0

        remember(output, i, vec)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"iter": i, "sim_to_prev": sim, "output": output,
                                "time": time.time()}) + "\n")
        print(f"[{i}] sim={sim} | {output[:120]!r}")

        last_output, last_vec = output, vec


if __name__ == "__main__":
    main()
