# Local AI Assistant with Memory

A voice assistant that runs its language model on your own computer, with persistent memory, an animated sphere that reacts to speech, and written limits on what it will do. It started as an experiment in feeding a model's output back into its input and grew into a prototype for a household assistant.

**Status:** prototype. The author has run the interface and confirmed that it listens, speaks aloud, shows an animated sphere that moves with the voice, and opens a settings menu. Other features are implemented in code but not yet independently tested (see [Limitations](#limitations)).

## What is in this repo

| File | What it does |
| --- | --- |
| `agent.py` | The assistant: local web server and browser interface in one file |
| `loop.py` | The original experiment: a model's output fed back into its input, with a repetition monitor and a JSONL log |
| `name_test.py` | Naming experiment across four conditions, with name extraction and a results summary |
| `replay_test.py`, `trace_test.py` | Replay and trace versions of the naming test, using a conversation log (the author's log is private and not included) |
| `name_results.jsonl` | Raw output of one run of `name_test.py` |

## How the assistant works

- **Model:** runs locally through Ollama (default `llama3.1:8b`; `llama3.2:3b` is faster on CPU). Nothing is sent to a cloud language model.
- **Memory, three layers:** recent turns kept word for word; a Chroma vector store of past exchanges, recalled by similarity and weighted toward what the user said; and saved notes rewritten every 8 exchanges in the background.
- **Voice:** replies stream sentence by sentence and are spoken as they arrive, using the browser's speech synthesis. Speech input uses the browser's speech recognition, with tap-to-talk, interruption, and an optional wake phrase.
- **Interface:** a canvas sphere with five states (dormant, idle, listening, thinking, speaking), captions, a typing fallback, and a settings drawer for voice, speed, microphone mode, and limits.
- **First start:** a short birth conversation in which the household's founding notes are recorded and the family names the assistant. Each name suggestion the model makes is logged.
- **Limits in code:** a fixed crisis reply that points to 988 and the National Domestic Violence Hotline; a daily cap, minimum gap, and quiet hours for unprompted speech, enforced on the server; and a similarity check that stops the assistant repeating itself.

## Run it

You need Python 3.12 and [Ollama](https://ollama.com).

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python agent.py                # options: --model llama3.2:3b  --port 8000  --data agent_data
```

Open http://localhost:8000 in Chrome or Edge.

## What the experiments found

`name_test.py` asks a model to pick a name for itself under four conditions (plain prompt, a prompt that denies inner experience, a neutral prompt, and a long priming prompt) and tallies the results. Findings so far:

- The names the model chose depended on the conversation it was in, not on a fixed identity.
- Long loops tended to collapse into repetition or mystical language.
- The model sometimes invented perceptions and memories ("I remember reading...").
- A first version of the name extractor reported names the model had never written. The fix was to accept a name only if it appears in the reply text.

These are observations from small runs on one local model, not controlled results.

## Limitations

- Speech recognition in Chrome and Edge is usually done by the browser vendor's cloud service, so what you say may leave your computer even though the language model stays local.
- The wake phrase, voice quality, and response speed on a CPU-only machine have not been measured. The 8B model can be slow without a GPU.
- Naming, free-speech mode, memory recall, and the limits are implemented but have not been independently tested.
- The assistant does not give financial, medical, or legal decisions. It is a prototype, not a safety product, and should not be relied on in a crisis.
- The replay and trace scripts need a conversation log that is not included.

## Built by

Nicholas Tamburrino, with ChatGPT and Claude as development aids. Design decisions and review of the code are my responsibility.

## License

No license has been chosen yet. All rights reserved until one is added.
