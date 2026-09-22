"""A guided, consented recording session that keeps numbers, not audio.

    python -m juno_core.intelligence.gate_training collect \\
        --out data/gate/session1.csv --speakers p1,p2 --room kitchen --consent release

None of the public corpora contains people talking TO an assistant, which is
the one class the gate most needs to recognise. This fills that gap the only
honest way: a short session where people are asked, prompt by prompt, to
talk to Juno, to each other, or to stay quiet while media plays. The label
comes from the prompt, so nobody labels anything by hand afterwards.

WHAT IS KEPT
------------
Each segment goes through the same pieces the runtime uses -- the configured
capture, VAD and SegmentDetector, the voiceprint if you've enrolled, and
extract_gate_features -- and only the resulting feature row is written.
The audio is held in memory for the length of one segment and released. No
audio file is created at any point.

WHY THE CONTEXT COLUMNS ARE NEUTRAL
-----------------------------------
A staged session can't produce honest conversational timing (how long since
Juno spoke, whether the last thing was answered): any values written here
would be ones this script made up, and a model would learn the script. So
every row is recorded as a cold start. Those columns are then constant, so
they get no weight, and the model learns from the SOUND. Context stays with
the gate's rules, which came from real logs.

CONSENT
-------
Everyone who will be heard must agree before it starts, including anyone
who only walks through the room. ``--consent release`` means they agreed that
models and feature tables derived from the session may be published with an
open-source project; ``internal`` means evaluation only. Use pseudonyms for
``--speakers`` (p1, p2...), never names.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from juno_core.intelligence.features import FEATURE_NAMES, extract_gate_features
from juno_core.intelligence.gate import NEVER_SPOKEN, Acoustics, Snapshot

CONSENT_TEXT = """\
This session records from the microphone to teach Juno's pre-transcription
gate. Only numbers describing each utterance (pitch, timing, spectral shape,
duration) are kept, labelled with what you were asked to do. No audio is
saved, and nothing is uploaded.

Scope of consent: {scope}

Everyone who will be heard -- including anyone who might walk through --
must agree before it starts. Stop at any time with Ctrl+C; what was
collected up to then is kept unless you delete the output file.
"""

SCOPES = {
    "release": "the feature rows, and models trained on them, may be published "
               "with an open-source project",
    "internal": "the feature rows are used for evaluation on this machine only "
                "and are not published",
}


@dataclass(frozen=True)
class Prompt:
    label: str
    text: str
    seconds: float
    who: str = "each"        # "each": once per speaker; "all": everyone together; "none"


# Ordered to alternate classes, so fatigue and room drift don't line up with
# one label. Questions addressed to people are included on purpose: they're
# the hard negatives, since they sound most like requests.
DEFAULT_SCRIPT = (
    Prompt("assistant_directed",
           "Ask Juno questions, one at a time, pausing a few seconds between "
           "them. With and without its name: \"Juno, what's the weather "
           "tomorrow?\", \"How long do I boil an egg?\"", 50),
    Prompt("human_directed",
           "Talk to each other normally -- your day, plans, anything. Don't "
           "address Juno.", 75, who="all"),
    Prompt("assistant_directed",
           "Give Juno short commands: \"set a timer for ten minutes\", \"Juno, "
           "stop\", \"turn it up\", \"what's next?\"", 40),
    Prompt("human_directed",
           "Ask each other questions and answer them: \"did you eat yet?\", "
           "\"what time is it?\", \"can you pass that?\"", 60, who="all"),
    Prompt("background_or_media",
           "Play a podcast, audiobook or video out loud at normal volume. "
           "Nobody speaks. Prefer public-domain or CC-licensed media (e.g. "
           "LibriVox).", 75, who="none"),
    Prompt("assistant_directed",
           "From across the room, at normal volume, ask Juno something.", 40),
    Prompt("human_directed",
           "From across the room, talk to each other.", 45, who="all"),
)

SOLO_TEXT = {
    # With one person there's nobody to talk to, so a phone call stands in.
    "human_directed": "On a phone call (or pretending to be on one), talk "
                      "normally. Don't address Juno.",
}


class Collector:
    """Runs a script against a live capture and returns feature rows."""

    def __init__(self, config, *, speakers: Sequence[str], session: str, room: str,
                 mic: str, consent: str, capture=None, ask: Callable = input,
                 say: Callable = print, clock: Callable = time.monotonic,
                 observer=None) -> None:
        from juno_core.audio.vad import SegmentDetector, build_vad
        from juno_core.audio.voiceprint import VoicePrint

        if consent not in SCOPES:
            raise ValueError(f"consent must be one of {sorted(SCOPES)}")
        if not speakers:
            raise ValueError("name at least one speaker (pseudonyms: p1, p2, ...)")
        section = (lambda key: config.get(key) or {}) if config is not None else (lambda key: {})
        audio_cfg = section("audio")
        self.rate = int(audio_cfg.get("sample_rate", 16000))
        vad_cfg = _Section(section("vad"), dict(threshold=0.5, min_speech_ms=200,
                                                min_silence_ms=550, min_segment_ms=350,
                                                max_segment_ms=20000))
        self.detector = SegmentDetector(build_vad(vad_cfg, self.rate, observer), vad_cfg,
                                        self.rate, preroll=float(audio_cfg.get("preroll", 0.4)))
        self.voice = VoicePrint(section("own_voice"), observer)
        if capture is None:
            from juno_core.audio.capture import build_capture

            capture = build_capture(audio_cfg, observer)
        self.capture = capture
        self.speakers = [s.strip() for s in speakers if s.strip()]
        self.session, self.room, self.mic, self.consent = session, room, mic, consent
        self.ask, self.say, self.clock = ask, say, clock
        self.rows: list[dict] = []
        self.dropped_not_wearer = 0

    # -- the session -------------------------------------------------------

    def confirm_consent(self) -> bool:
        self.say(CONSENT_TEXT.format(scope=SCOPES[self.consent]))
        answer = self.ask(f"Has everyone who will be heard ({', '.join(self.speakers)}) "
                          f"agreed? Type yes to start: ")
        return str(answer).strip().lower() in ("yes", "y")

    def run(self, script: Sequence[Prompt] = DEFAULT_SCRIPT, scale: float = 1.0) -> list[dict]:
        steps = list(self._steps(script))
        self.capture.start()
        try:
            for n, (prompt, speaker, text) in enumerate(steps, start=1):
                who = f"{speaker}: " if prompt.who == "each" else ""
                self.say(f"\n[{n}/{len(steps)}] {who}{text}")
                self.ask("Press Enter to start (Ctrl+C to stop) ")
                self._record(prompt, speaker, prompt.seconds * scale)
        except KeyboardInterrupt:
            self.say("\nstopped early -- keeping what was collected")
        finally:
            self.capture.stop()
        return self.rows

    def _steps(self, script):
        solo = len(self.speakers) == 1
        for prompt in script:
            text = SOLO_TEXT.get(prompt.label, prompt.text) if solo and prompt.who == "all" \
                else prompt.text
            if prompt.who == "each":
                for speaker in self.speakers:
                    yield prompt, speaker, text
            elif prompt.who == "all":
                yield prompt, "+".join(self.speakers), text
            else:
                yield prompt, "", text

    def _record(self, prompt: Prompt, speaker: str, seconds: float) -> None:
        self.detector.reset()
        started = self.clock()
        before = len(self.rows)
        self.say(f"  recording for {seconds:.0f}s...")
        for frame in self.capture.frames():
            if frame.timestamp < started:
                continue                  # queued while the prompt was being read
            if frame.timestamp - started >= seconds:
                break
            segment = self.detector.push(frame)
            if segment is not None:
                row = self._row(segment, prompt.label, speaker)
                if row is not None:
                    self.rows.append(row)
                    self.say(f"  heard {row['duration']:.1f}s")
        self.detector.reset()             # an unfinished segment is dropped, not kept
        self.say(f"  {len(self.rows) - before} segments")

    def _row(self, segment, label: str, speaker: str) -> dict | None:
        estimate = None
        if self.voice.enabled:
            estimate = self.voice.estimate(segment.audio, self.rate)
            if estimate.confident and not estimate.is_wearer:
                # The runtime drops these before the gate sees them.
                self.dropped_not_wearer += 1
                return None
        confident = bool(estimate is not None and estimate.confident)
        acoustics = Acoustics(
            seconds=float(segment.duration), confidence=float(segment.confidence),
            truncated=bool(segment.truncated),
            p_own=estimate.p_own if confident else None,
            voice_confident=confident,
            is_wearer=estimate.is_wearer if confident else None,
        )
        t0 = time.perf_counter()
        vec = extract_gate_features(segment.audio, self.rate, acoustics,
                                    Snapshot(since_ai=NEVER_SPOKEN))
        feature_ms = (time.perf_counter() - t0) * 1000.0
        row = {"label": label, "source": "consented", "license": "consent",
               "consent": self.consent, "speaker": speaker or "media",
               "session": self.session, "room": self.room, "mic": self.mic,
               "feature_ms": round(feature_ms, 3), "intent_verdict": ""}
        row.update({name: float(v) for name, v in zip(FEATURE_NAMES, vec)})
        return row


class _Section:
    """config.get with defaults, for the attribute access SegmentDetector uses."""

    def __init__(self, data, defaults: dict) -> None:
        self._data = dict(defaults)
        self._data.update({k: data[k] for k in data} if data else {})

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def get(self, key, default=None):
        return self._data.get(key, default)


def default_session() -> str:
    return time.strftime("session-%Y%m%d-%H%M")


def collect_main(args) -> int:
    from juno_core.intelligence.gate_training import read_rows, write_rows

    out = Path(args.out)
    if out.exists() and not args.append:
        raise SystemExit(f"{out} exists; pass --append to add this session to it")
    config = None
    if args.config or Path("config.yaml").exists():
        from juno_core.config import load_config

        config = load_config(args.config)
    mic = args.mic
    if not mic:
        from juno_core.audio.voiceprint import current_microphone

        mic = current_microphone() or "unknown"
    collector = Collector(config, speakers=args.speakers.split(","),
                          session=args.session or default_session(), room=args.room,
                          mic=mic, consent=args.consent)
    if not collector.confirm_consent():
        print("not started: consent not confirmed")
        return 1
    rows = collector.run(scale=args.scale)
    if not rows:
        print("nothing collected")
        return 1
    existing = read_rows([out]) if out.exists() else []
    out.parent.mkdir(parents=True, exist_ok=True)
    write_rows(out, existing + rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"\nwrote {len(rows)} feature rows to {out} {counts}; no audio was saved")
    if collector.dropped_not_wearer:
        print(f"({collector.dropped_not_wearer} segments were confidently not the "
              f"enrolled voice and were skipped, as the runtime would)")
    return 0
