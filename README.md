# Juno

**Decides whether you're talking to it. No wake word.**

> **Status: early (v0.1).** It runs end to end, but it's young: expect rough
> edges, and interfaces may change between releases. The accuracy figures in
> this README come from the original project's internal evaluation on a small
> set, and that evaluation isn't included here yet, so treat them as
> indicative. Issues and reports from real rooms are very welcome.

Most voice assistants need "Hey Siri" or "OK Google" because they can't tell
the difference between speech aimed at them and speech aimed at anybody else
in the room. This is the part that can tell the difference — continuous
listening, and on every utterance a decision: was that addressed to *me*, to
the person across the table, or to nobody. Nothing is sent anywhere until the
answer is yes.

This repo is *only* that decision-making technology. It ships with:

- no speech-to-text engine baked in (bring your own — three are wired up and ready)
- no text-to-speech baked in (same — plug one in, or just read the answer)
- no specific language model — BYOK, and OpenAI / Anthropic / Gemini / a
  local Ollama model all work out of the box
- no UI, no dashboard, no telemetry that leaves your machine

What it does ship: continuous voice-activity detection, a gate that decides
whether an utterance is even worth transcribing, speaker verification
("was that *you*"), and the actual addressee-detection engine that reads a
transcript for evidence of who it's for. About 6,000 lines of Python, pulled
out of a larger personal-assistant project (Juno) where everything past this
point — the tool loop, the specific model, the speech synthesis — is the
*product*, replaceable and not the interesting part. This is the part that
was hard to get right, and it's Apache-2.0-licensed so you don't have to redo it.

Every number in this README was measured — see [REPORT.txt](REPORT.txt) for
the methodology and the honest limits (section 6 of it, specifically — read
that before you quote any of this elsewhere).

---

## Quickstart

You'll have something you can talk to in about five minutes, using the
free/local defaults (local Whisper for hearing you, via `mlx-whisper` on an
Apple Silicon Mac and `faster-whisper` everywhere else, and no language
model until you add a key). Requires Python 3.10+.

```bash
git clone <this-repo-url> juno-core
cd juno-core
pip install -e ".[mlx]"               # Apple Silicon Mac
# pip install -e ".[faster-whisper]"  # Linux, Windows, Intel Mac

cp config.example.yaml config.yaml
cp .env.example .env

python run.py
```

The first run downloads the voice-activity model (2 MB) and your
speech-to-text engine's weights, then works offline. To fetch the models up
front instead, run `python -m juno_core.assets`.

Talk to it. By default it'll hear you and decide whether you were talking
to it — but until you add a language-model key below, it just tells you what
it heard instead of actually answering, and the genuinely unclear sentences
(about 1 in 5) get no second opinion, so it stays quiet on those. That's deliberate: you can
confirm the interesting half (the listening) works before deciding whose API
you want doing the answering.

To get real answers, open `.env` and add **one** key, and set the matching
provider in `config.yaml`'s `llm:` section:

| Provider | `llm.provider` | Key goes in | Get a key |
|---|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| Anthropic (Claude) | `anthropic` | `ANTHROPIC_API_KEY` | https://console.anthropic.com/settings/keys |
| Google (Gemini) | `gemini` | `GOOGLE_API_KEY` | https://aistudio.google.com/apikey |
| Ollama (local, no key, no cloud) | `ollama` | — | https://ollama.com, then `ollama pull llama3.2` |

Run `python run.py` again. That's the whole setup. Everything past this point
in the README is what each piece does and how to go further — enrolling your
voice, choosing a different speech-to-text engine, hooking in your own agent
instead of a plain chat reply, and what the numbers above actually mean.

---

## How it fits together

```
  microphone
      |
      v
  VOICE ACTIVITY DETECTION        Silero VAD (ONNX) or a zero-dependency
  juno_core/audio/vad.py          energy-based fallback. Finds where an
                                   utterance starts and stops. ~0.1 ms per
                                   32 ms frame — effectively free.
      |
      v
  WHOSE VOICE (optional)          Speaker verification against a voice you
  juno_core/audio/voiceprint.py   enrol once (see below). If it's confidently
                                   not you, the segment is dropped HERE —
                                   before transcription, which is the
                                   expensive part.
      |
      v
  WORTH TRANSCRIBING?             A gate that scores conversational timing
  juno_core/intelligence/gate.py  and acoustic tails to decide whether to
                                   even bother running speech-to-text. Ships
                                   conservative (see "The numbers" below).
      |
      v
  SPEECH-TO-TEXT                  Yours. Three ready-made options included
  juno_core/stt/                  (mlx_whisper.py, faster_whisper.py,
  (bring your own)                openai_whisper.py); implement the ~15-line
                                   interface for anything else.
      |
      v
  WAS THAT MEANT FOR ME?          The core of this repo. A ~30-signal
  juno_core/intelligence/         heuristic scorer (questions, imperatives,
  intent.py                       your name, discourse markers aimed at
                                   humans, whether it just asked something),
                                   with a language-model second opinion
                                   consulted only on genuinely ambiguous
                                   cases (~1 in 5 utterances).
      |
      v (only if yes)
  YOUR AGENT                      juno_core/pipeline.py hands off exactly
  (bring your own)                here: the accepted text plus the
                                   conversation so far. What happens next is
                                   entirely up to you — see "Connecting your
                                   own agent" below.
```

Everything above the "your agent" line has no dependency on a specific
language model, tool framework, or UI — it's ~30k lines' worth of "does the
person mean me" reduced to the roughly 6,000 that matter, with the specific
model and the tool loop it used to run against removed. That boundary is
enforced in the original project by a test that fails if the technology side
ever imports a provider — see REPORT.txt section 1.

---

## Step by step

### 1. Install

```bash
pip install -e ".[mlx]"                # recommended on Apple Silicon: local, on the GPU
pip install -e ".[faster-whisper]"     # recommended elsewhere: local, on the CPU
# or, for the cloud option instead:
pip install -e ".[http]"
```

`numpy`, `onnxruntime`, `sounddevice` and `pyyaml` are the only hard
dependencies — the half of this that decides whether it's being spoken to
needs `numpy` and one ONNX runtime, full stop. Everything else (a specific
speech-to-text engine, `requests` for any cloud API) is an extra you opt
into.

### 2. Configure

```bash
cp config.example.yaml config.yaml
cp .env.example .env
```

`config.yaml` is commented inline — every value is the default this project
was measured with. `.env` holds API keys and is gitignored; never commit it.

### 3. Choose a speech-to-text engine

This repo doesn't ship one — that's the "no STT baked in" part — but three
are ready to select by name, and adding your own is a small class (see
below). The default, `stt.provider: auto`, picks `mlx_whisper` on an Apple
Silicon Mac when it's installed and `faster_whisper` otherwise.

| | `stt.provider` | Where it lives | Needs | When to pick it |
|---|---|---|---|---|
| **mlx-whisper** (recommended on Apple Silicon) | `mlx_whisper` | `juno_core/stt/mlx_whisper.py` | `pip install mlx-whisper`, an M-series Mac | Local, offline, free. Runs Whisper on the Mac's GPU through Apple's MLX framework. First run downloads the converted weights from Hugging Face (`mlx-community`) and caches them. Doesn't run on Linux, Windows or Intel Macs. |
| **faster-whisper** (recommended elsewhere) | `faster_whisper` | `juno_core/stt/faster_whisper.py` | `pip install faster-whisper` | Local, offline, free, and runs anywhere. First run downloads model weights (a few hundred MB) and caches them. Its CTranslate2 backend has no Metal support, so on a Mac it runs on the CPU. Fine, but slower than MLX there. |
| OpenAI Whisper API | `openai_whisper` | `juno_core/stt/openai_whisper.py` | `OPENAI_API_KEY`, `pip install requests` | Simplest possible setup, no local model — audio leaves the machine. |

Set `stt.provider` (and, for the two local engines, `stt.model` — `tiny.en`
through `medium.en`, or `large-v3-turbo`; bigger is slower and more accurate;
both engines take the same names) in `config.yaml`.

**Using something else** (a different local model, a different cloud API):
put a new file in `juno_core/stt/`, implement the interface in
`juno_core/stt/__init__.py` —

```python
class MySTT(STTEngine):
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        text = ...  # however you get text out of `audio`
        return Transcript(text=text)
```

— and either add a branch to `build_stt()` in that same file, or skip the
factory entirely and pass `MySTT()` straight to `JunoPipeline(..., stt=MySTT())`
in your own copy of `run.py`.

### 4. Bring your own model (BYOK)

Set `llm.provider` in `config.yaml` to one of `openai`, `anthropic`,
`gemini`, or `ollama`, and put the matching key in `.env` (table above, under
Quickstart). `llm.model` can stay `null` to use a sensible default per
provider, or name a specific one.

This one language model is used in exactly two places: as a second opinion
for utterances the fast heuristic finds genuinely ambiguous (`intent.
llm_adjudicator` in config.yaml — turn it off to save the network round trip
and lose a little accuracy in that ambiguous ~20%), and, unless you've
connected your own agent (next section), to actually answer once an
utterance is accepted.

Adding a fifth provider is a ~50-line file in `juno_core/llm/` implementing
`generate()` — the four bundled ones are a template; none of them use an SDK,
just a plain HTTPS call, so there's no dependency to add for a fifth.

### 5. Teach it your voice (optional)

```bash
python enroll.py
```

Eight short prompted sentences, under a minute (the first time, it also
downloads the 25 MB speaker model). This is what lets the gate
tell *you* apart from someone else talking near an open microphone — see
"The numbers" below for how well (97.9% on the bundled eval). Nothing but a
192-number voice template is kept; no audio. Skip this step and everything
still works, just without that signal — the text-based addressee detection
(the part measured at 94.3% on its own) carries the whole decision instead.

### 6. Hearing it talk back (optional)

By default, `run.py` prints the answer. To hear it instead, set `tts.provider`
in `config.yaml`:

| | `tts.provider` | Where it lives | Needs |
|---|---|---|---|
| macOS `say` | `say` | `juno_core/tts/system_say.py` | Nothing — built into macOS |
| OpenAI TTS | `openai` | `juno_core/tts/openai_tts.py` | `OPENAI_API_KEY`, `requests`, and `afplay` or `ffplay` to play it |

Same pattern as speech-to-text for adding your own: implement `speak(text)`
in a new file under `juno_core/tts/`.

### 7. Run it

```bash
python run.py
```

Just talk. No wake word. It prints (and, if configured, speaks) what it
heard and what it answered. `Ctrl+C` to stop. `python run.py --quiet` if you
only want the conversation, not the pipeline's status lines.

---

## Connecting your own agent

`run.py`'s default behaviour — accept an utterance, send it and the recent
conversation to your configured language model, print/speak the reply — is
one line of integration, and it's probably not what you actually want to
build. The real seam is one function:

```python
def on_accept(text: str, decision: IntentDecision, context: ConversationContext) -> str | None:
    ...
```

`juno_core/pipeline.py`'s `JunoPipeline` calls this exactly once per
accepted utterance — never for anything it decided wasn't addressed to you.
`text` is the transcript; `context.messages(turns=6)` renders the recent
conversation in the `[{"role": ..., "content": ...}]` shape every chat API
already expects; `decision` carries the confidence and the signals that
produced it, if you want to branch on how sure it was. Return the reply (it
gets printed/spoken), or `None` if you're handling the response yourself —
handing off to a background job that'll speak later, for instance.

Pass it in directly:

```python
pipeline = JunoPipeline(config, stt=stt, model=model, on_accept=my_agent)
```

See [`examples/connect_an_agent.py`](examples/connect_an_agent.py) for a
runnable one — two tiny on-device actions (the time, opening a site) with a
language-model fallback for everything else. Copy it as a starting point for
wiring in LangChain, AutoGen, the Claude Agent SDK, a shell script, or
whatever you already have that does something when asked.

If what you're building is a **long-running task that shouldn't block the
next thing said to it** — a search, a slow tool call — there's also a
narrower interface for exactly that in
[`juno_core/intelligence/executor.py`](juno_core/intelligence/executor.py):
an `Executor` takes a `Task` (the request plus conversation history) on a
worker thread and eventually produces an `Outcome` (a sentence, spoken
whenever it's ready). It's the same shape used internally to hand a long job
to a stronger model without going deaf while it runs — read its docstring,
which explains why the interface is kept deliberately narrow.

Nothing downstream of accepted-and-handed-off is this repo's business. It
decided you were being spoken to; what happens next — what a request is
allowed to do, what needs your confirmation before it happens, how far an
agent is allowed to go on its own — is a decision for whatever you connect
here, matched to what you're actually building.

---

## The numbers

All measured on an Apple M1, 8.6 GB unified memory — see REPORT.txt for the
exact commands (`scripts/bench_juno.py`, `eval/run_intent_eval.py`) and, in
its section 6, what these numbers don't say (synthesised speech, one
machine, one enrolled voice — read it before citing any of this).

**What each stage costs**, median:

| stage | cost |
|---|---|
| voice activity, per 32 ms frame | 0.1 ms |
| whose voice was that | 24.4 ms |
| is it being spoken to (heuristic) | 0.04 ms |
| speech-to-text | ~970 ms — this dominates everything else by 40x |

The speaker check that decides whether transcription is worth paying for
costs 2.5% of what it saves.

**Addressee detection** — 87 hand-labelled utterances, heuristic scoring
alone (no language-model adjudicator):

- 94.3% accuracy
- **0/44 (0.0%) false activations** — answering when it wasn't spoken to
- 5/43 (11.6%) missed activations — staying quiet when it was
- 18/87 (20.7%) land in the ambiguity band and consult the language model

That asymmetry is deliberate, not an accident of tuning: a false activation
— speaking into somebody else's conversation — is treated as three times
worse than staying quiet once, and the thresholds (`intent.accept_threshold`,
`intent.reject_threshold` in config.yaml) were swept against that cost, not
against raw accuracy.

**Speaker verification** — 47/48 (97.9%) correct across owner-vs-three-other-
voices, at range, and down to one-word answers; worst-case margin +0.213.
The one miss was a worst-case combination — very far away, heavily
reverberant — that nobody stands at in practice. Full breakdown, including
why this replaced an earlier *distance*-based approach (the orderings simply
didn't separate — a nearby stranger and a distant owner overlap, so no
threshold could tell them apart), is in REPORT.txt section 4.2.

**What conversational timing alone can safely do** — the finding behind
`intelligence/gate.py`: over six days of real logs, 63% of everything
transcribed turned out to be the wearer talking to somebody else, not to the
assistant, and *when* it happened separated the two classes almost
perfectly (median 2 seconds since the assistant last spoke, for utterances
meant for it; 65 seconds, for the ones that weren't). But timing alone,
applied naively, also skips real conversation-openers — "what's the capital
of Mongolia" is cold by definition, the same as anything ignored before it.
Gated so a skip also needs something about the *sound* to agree (a
confidently-not-you voice, a monologue past ten seconds, or a doubtful
voice-activity reading), it catches a real but honest 3% of otherwise-wasted
transcriptions at zero measured false negatives on the current engine — see
REPORT.txt 2.7 for the table and for why this ships in shadow mode
(`intent.gate.mode: shadow`) by default: it scores and logs every segment
without ever acting on it, so you can watch it agree with the intent engine
on your own conversations before switching it to `skip`.

**Scale**: this repository is roughly 6,000 lines pulled from a ~35,000-line
project; what's here is the part that decides whether it's being spoken to,
not what it does about it. The original ships 1,128 tests; the testing
approach worth stealing, if you extend this, is in REPORT.txt section 5 —
in short, tests that drive real audio through the real pipeline rather than
stub components in isolation, because every serious bug in this project's
history passed its parts' own tests and broke where two parts met.

## Training the pre-STT gate

The gate in `intelligence/gate.py` decides whether to *look* (run
speech-to-text), never whether to *answer*. It ships rule-based and in shadow
mode. It can also load a small logistic model that scores
p(assistant-directed) from the 20 numbers `features.py` computes. Everything
needed to train and check that model is in `intelligence/gate_training.py`.

**Privacy first.** At runtime a segment's audio stays in memory: it's scored,
handed to STT, then released. Nothing in `juno_core` writes microphone audio
to disk. The gate's log records scores, named signals and (with
`intent.gate.log_features: true`) the feature numbers. It never records
samples. Don't add a recorder to get training data. Use one of the sources
below.

**Training data has to be open-source compatible.** Juno is open source
under a permissive licence (see [LICENSE](LICENSE)), and a model trained
here may ship with it, so the tooling enforces where rows come from:

| source | licence | notes |
| --- | --- | --- |
| `ami` — AMI Meeting Corpus | CC BY 4.0 | human-to-human room conversation; attribution required |
| `common_voice` — Mozilla Common Voice | CC0 | speaker/accent/mic diversity; don't try to identify speakers |
| `speech_commands` — Google Speech Commands v0.02 | CC BY 4.0 | short commands; keep it a minority of the data |
| `custom` | must be one of CC0, CC BY, PDDL, ODC-By, CDLA-Permissive, MIT, Apache-2.0 | declared per row |
| `consented` | `consent=release` or `consent=internal` | your own recordings; see below |

The tooling refuses non-commercial (NC) and share-alike (SA) licences,
unlicensed audio (podcasts, YouTube, etc.) and audio from ordinary Juno use.
None of the public sets really covers speech *to an assistant*, so you'll
need a small **consented** set: people alternating between requests to Juno,
talking to each other, and background chatter, in real rooms. The `collect`
command runs that session for you (below). Only mark it `consent=release` if
participants agreed in writing that derived models and feature tables may be
published. Otherwise mark it `internal`: you can
evaluate on it, but a model trained on it (`--allow-internal`) is marked
non-distributable. Use pseudonymous speaker IDs. The trained model file
lists every source, its licence and its attribution. Keep that list if you
redistribute the model. Licences above were checked when this was written.
Confirm them on each dataset's page when you download it.

```bash
python -m juno_core.intelligence.gate_training sources            # registry
python -m juno_core.intelligence.gate_training collect \
    --out data/gate/s1.csv --speakers p1,p2 --room kitchen --consent release
python -m juno_core.intelligence.gate_training features \
    --manifest manifest.csv --out rows.csv [--delete-source]        # clips -> numbers
python -m juno_core.intelligence.gate_training train \
    --rows rows.csv --out juno_core/data/models/gate_model.json --max-false-skip 0.01
python -m juno_core.intelligence.gate_training evaluate --rows unseen.csv --model gate_model.json
python -m juno_core.intelligence.gate_training shadow --log logs/juno.jsonl
```

- `collect` runs a guided live session of about 10 minutes for two people.
  It prompts everyone in turn to ask Juno something, give it commands, talk
  to each other (including asking each other questions, the hardest cases),
  and stay quiet while media plays, near the microphone and from across the
  room. Each segment goes through your configured microphone, VAD,
  voiceprint and feature extraction, exactly as at runtime. The label comes
  from the prompt, only the feature row is written, and it asks everyone to
  confirm consent before starting. Context columns are recorded as a cold
  start, because a staged session can't produce honest conversational
  timing. So the model learns from the *sound*, and context stays with the
  rules. Aim for a few sessions: different people, rooms and microphones.
  `train` keeps each session within one split.
- `features` reads 16 kHz integer-PCM WAVs listed in a manifest (`path`,
  `label`, `source`, plus optional `speaker`, `session`, `room`, `mic`,
  `start`/`end` and conversational context such as `since_ai` and
  `awaiting_answer`). It runs the same `extract_gate_features` production
  uses and writes feature rows. It never writes audio.
  `--delete-source` deletes each *consented* clip once its row is written.
  For features that match runtime segments, cut consented recordings with
  the same VAD settings you run with.
- `train` splits by speaker, session and room together, so no speaker,
  session or room shows up in two splits. It fits on *train*, calibrates on
  *select*, and picks the skip threshold on *select*: the largest threshold
  whose false-skip rate (over `assistant_directed` and `uncertain` rows)
  stays within `--max-false-skip`. It reports once on an untouched *holdout*.
  The report covers false-skip rate with a 95% upper bound, skip rate per
  class, worst speaker/mic/room/session, latency, the rules-only baseline on
  the same rows, and, for each assistant-directed segment it would have
  skipped, the features that pushed it there.
- Every number comes from the runtime `Gate` given the stored vectors, so
  what's evaluated is the code that runs.

**Rolling it out:** set `intent.gate.learned: true` with `mode: shadow`, run
it, and use `gate_training shadow` to compare every would-skip against the
intent engine's verdict. Switch to `mode: skip` only once that comparison and
the holdout report meet your false-skip target. Even then the gate always
transcribes when:

- Juno is waiting for an answer, a confirmation or an offer, or the
  follow-up window is open;
- your enrolled voice was confidently detected (`always_transcribe_wearer`);
- features are doubtful (non-finite, or nothing voiced);
- the model is missing or broken, or has no calibrated threshold.

Setting `mode` back to `shadow` or `off` turns skipping off immediately.
Feature extraction runs on the critical path, so it's batched through
NumPy. That makes it about 5× faster than the frame-by-frame original
(about 5 ms for a 5 s segment on an M1, and about 18 ms at the 20 s maximum),
with the same output. In skip mode it doesn't run at all when a rule already
requires transcription.

---

## What's in here

```
juno_core/
  audio/
    capture.py        microphone input (sounddevice, or PortAudio)
    vad.py             voice activity detection, segments an utterance
    buffering.py        pre-speech ring buffer
    voiceprint.py      speaker verification ("was that you")
    calibration.py     the enrolment flow behind enroll.py
  intelligence/
    context.py          conversation state: recent utterances, recent turns
    features.py          acoustic features (pitch, voicing, spectral shape)
    gate.py               pre-transcription "worth listening to?" check
    gate_training.py      offline: build feature tables, train/evaluate the gate
    gate_collect.py       guided, consented recording session (keeps numbers only)
    intent.py              the addressee-detection engine itself
    followups.py           "say that again" / "tell me more", detected cheaply
    executor.py             the narrow interface for handing off long-running work
  stt/                    speech-to-text: the interface, plus three ready adapters
  llm/                    the BYOK language-model interface and four adapters
  tts/                     optional text-to-speech interface and two adapters
  pipeline.py               wires all of the above into one running loop
  config.py, events.py, observability.py    small supporting pieces

run.py            the entry point — start here
enroll.py         optional voice enrolment
tests/            python -m unittest discover tests
examples/
  connect_an_agent.py   how to hook in your own agent instead of a plain reply
config.example.yaml    copy to config.yaml
.env.example            copy to .env
REPORT.txt              the full technical writeup the numbers above are from
```

Nothing in `juno_core/` imports a specific language model, a tool registry,
or a UI. The one place a model is *used* is behind the `LanguageModel`
interface in `juno_core/llm/__init__.py`, and only for the adjudicator
described above and the default (replaceable) reply.

---

## Third-party models

Juno doesn't bundle model weights. It downloads them on first use (see
`juno_core/assets.py`), each pinned to one version and checked against a
SHA-256 checksum:

| Model | Used for | Licence | Source |
|---|---|---|---|
| Silero VAD v6.2.2, Silero Team | voice activity detection | MIT | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) |
| WeSpeaker ECAPA-TDNN512-LM, WeSpeaker team, trained on VoxCeleb | speaker verification (`enroll.py`) | CC BY 4.0 | [Wespeaker/wespeaker-ecapa-tdnn512-LM](https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM) |
| OpenAI Whisper (via `mlx-community` or `Systran` conversions) | speech-to-text, if you pick a local engine | MIT | downloaded by `mlx-whisper` / `faster-whisper` |

If you redistribute any of these, keep their licence and attribution.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Use it, fork it, put it in
something that has nothing to do with the project it came from.
