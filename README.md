# Juno

**Decides whether you're talking to it. No wake word.**

Most voice assistants need "Hey Siri" or "OK Google" because they can't tell
the difference between speech aimed at them and speech aimed at anybody else
in the room. This is the part that can tell the difference — continuous
listening, and on every utterance a decision: was that addressed to *me*, to
the person across the table, or to nobody. Nothing is sent anywhere until the
answer is yes.

This repo is *only* that decision-making technology. It ships with:

- no speech-to-text engine baked in (bring your own — two are wired up and ready)
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
was hard to get right, and it's MIT-licensed so you don't have to redo it.

Every number in this README was measured — see [REPORT.txt](REPORT.txt) for
the methodology and the honest limits (section 6 of it, specifically — read
that before you quote any of this elsewhere).

---

## Quickstart

You'll have something you can talk to in about five minutes, using the
free/local defaults (`faster-whisper` for hearing you, nothing for a language
model until you add a key). Requires Python 3.10+.

```bash
git clone <this-repo-url> juno-core
cd juno-core
pip install -e ".[faster-whisper]"

cp config.example.yaml config.yaml
cp .env.example .env

python run.py
```

Talk to it. By default it'll hear you and correctly decide whether you were
talking to it — but until you add a language-model key below, it just tells
you what it heard instead of actually answering. That's deliberate: you can
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
  SPEECH-TO-TEXT                  Yours. Two ready-made options included
  juno_core/stt/                  (juno_core/stt/faster_whisper.py, .../
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
pip install -e ".[faster-whisper]"     # recommended: local speech-to-text
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

This repo doesn't ship one — that's the "no STT baked in" part — but two are
ready to select by name, and adding your own is a small class (see below).

| | `stt.provider` | Where it lives | Needs | When to pick it |
|---|---|---|---|---|
| **faster-whisper** (recommended) | `faster_whisper` | `juno_core/stt/faster_whisper.py` | `pip install faster-whisper` | Local, offline, free. First run downloads model weights (a few hundred MB) and caches them. Runs fine on CPU. |
| OpenAI Whisper API | `openai_whisper` | `juno_core/stt/openai_whisper.py` | `OPENAI_API_KEY`, `pip install requests` | Simplest possible setup, no local model — audio leaves the machine. |

Set `stt.provider` (and, for faster-whisper, `stt.model` — `tiny.en` through
`medium.en`, bigger is slower and more accurate) in `config.yaml`.

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

Eight short prompted sentences, under a minute. This is what lets the gate
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
    retention.py       optional, consent-gated local recording for training
                        your own gate (off by default — see its docstring)
  intelligence/
    context.py          conversation state: recent utterances, recent turns
    features.py          acoustic features (pitch, voicing, spectral shape)
    gate.py               pre-transcription "worth listening to?" check
    intent.py              the addressee-detection engine itself
    followups.py           "say that again" / "tell me more", detected cheaply
    executor.py             the narrow interface for handing off long-running work
  stt/                    speech-to-text: the interface, plus two ready adapters
  llm/                    the BYOK language-model interface and four adapters
  tts/                     optional text-to-speech interface and two adapters
  pipeline.py               wires all of the above into one running loop
  config.py, events.py, observability.py    small supporting pieces

run.py            the entry point — start here
enroll.py         optional voice enrolment
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

## License

MIT — see [LICENSE](LICENSE). Use it, fork it, put it in something that has
nothing to do with the project it came from.
