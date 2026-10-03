"""A guided, consented session that teaches System One with real voices in a real room.

    python -m juno_core.slu collect --out data/slu/consented/s1 --speakers p1,p2 \\
        --room kitchen --consent release --encoder parakeet logmel

Synthetic speech gets the student started; it cannot tell you how the student
does on YOUR voice, YOUR room and YOUR microphone, and it says nothing about
how people sound when they really talk to a device versus to each other. This
does, the same way gate_collect.py does for the gate: prompt by prompt,
people are asked to give Juno a particular command, ask it something open,
talk to each other, or stay quiet while media plays. The prompt is the gold
label, so nobody labels anything afterwards.

WHAT IS KEPT
------------
Each segment goes through the runtime's capture, VAD, SegmentDetector and
voiceprint (if enrolled), exactly as in use, and then:

  - one pooled vector per ``--encoder`` (several, so encoders can be
    compared on the same takes later);
  - the teacher's answer (the transcript, its verdict, intent and slots) --
    the participants agreed to this being kept, and it is what the student
    is distilled from;
  - the prompt's label and provenance.

The audio itself is held in memory for one segment and released. No audio
file is created at any point. The vectors are tied to their encoder: change
the encoder and the session has to be recollected, which is the price of not
keeping recordings.

CONSENT
-------
As for the gate: everyone who will be heard agrees first. ``release`` means
derived models and tables may be published with an open-source project;
``internal`` means evaluation on this machine only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from juno_core.intelligence.gate_collect import Collector, Prompt, default_session
from juno_core.slu.schema import ASSISTANT, OPEN_REQUEST


@dataclass(frozen=True)
class SLUPrompt(Prompt):
    intent: str | None = None


# Ordered to alternate classes. Every assistant prompt names one intent, so
# the label is exact; the teacher fills in slot values from the words.
SLU_SCRIPT = (
    SLUPrompt(ASSISTANT, "Give Juno timer commands, a few seconds apart, different lengths: "
              "\"Juno, set a timer for ten minutes\", \"timer for half an hour\".", 40,
              intent="timer.set"),
    SLUPrompt("human_directed", "Talk to each other normally -- plans, your day. Don't address "
              "Juno.", 60, who="all"),
    SLUPrompt(ASSISTANT, "Tell Juno to stop, a few different ways: \"stop\", \"Juno, that's "
              "enough\", \"okay stop\".", 30, intent="stop"),
    SLUPrompt(ASSISTANT, "Ask Juno open questions: \"what's the weather tomorrow\", \"how "
              "far is the moon\", \"remind me to call mum at six\".", 45, intent=OPEN_REQUEST),
    SLUPrompt("human_directed", "Ask each other things that sound like commands: \"what time "
              "is it?\", \"can you turn it down?\", \"stop it!\", \"set a timer for the "
              "pasta\".", 50, who="all"),
    SLUPrompt(ASSISTANT, "Ask Juno the time, a few ways.", 25, intent="time.now"),
    SLUPrompt(ASSISTANT, "Ask Juno to turn it up / louder, a few ways.", 25, intent="volume.up"),
    SLUPrompt(ASSISTANT, "Ask Juno to turn it down / quieter, a few ways.", 25, intent="volume.down"),
    SLUPrompt("background_or_media", "Play a podcast or video out loud. Nobody speaks. "
              "Prefer public-domain or CC-licensed media (e.g. LibriVox).", 60, who="none"),
    SLUPrompt(ASSISTANT, "Say \"never mind\" / \"cancel that\" to Juno, a few ways.", 25,
              intent="cancel"),
    SLUPrompt(ASSISTANT, "Pause and resume playback: \"pause\", \"resume\", \"carry on\". "
              "(Mixed intents: the teacher labels which.)", 30, intent=None),
    SLUPrompt(ASSISTANT, "From across the room, give Juno any command from above.", 40,
              intent=None),
    SLUPrompt("human_directed", "From across the room, talk to each other.", 40, who="all"),
)


class SLUCollector(Collector):
    """The gate's collector, keeping vectors and teacher labels instead of features."""

    def __init__(self, config, *, encoders, teacher, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.encoders = list(encoders)
        self.teacher = teacher
        self.vectors: dict[str, list] = {e.spec: [] for e in self.encoders}
        self._prompt: SLUPrompt | None = None
        self._n = 0

    def _record(self, prompt, speaker, seconds) -> None:
        self._prompt = prompt
        super()._record(prompt, speaker, seconds)

    def _row(self, segment, label: str, speaker: str) -> dict | None:
        if self.voice.enabled:
            estimate = self.voice.estimate(segment.audio, self.rate)
            if estimate.confident and not estimate.is_wearer:
                self.dropped_not_wearer += 1
                return None
        clip_id = f"{self.session}-{self._n:04d}"
        self._n += 1
        teacher = self.teacher.label(segment.audio, self.rate).as_row()
        for encoder in self.encoders:
            self.vectors[encoder.spec].append(encoder.encode(segment.audio, self.rate))
        intent = getattr(self._prompt, "intent", None)
        return consented_row(clip_id, label, intent, speaker, session=self.session,
                             room=self.room, mic=self.mic, consent=self.consent,
                             duration=float(segment.duration), teacher=teacher)


def consented_row(clip_id: str, label: str, intent: str | None, speaker: str, *,
                  session: str, room: str, mic: str, consent: str, duration: float,
                  teacher: dict, category: str | None = None) -> dict:
    """One collected utterance: gold label from the prompt, the teacher's
    answer, provenance. No path, because there is no audio."""
    return {
        "id": clip_id, "path": None, "g_addressed": label,
        "g_intent": intent if label == ASSISTANT else None, "g_slots": {},
        "category": category or f"consented_{label}", "source": "consented",
        "license": "consent", "consent": consent, "speaker": speaker or "media",
        "session": session, "room": room, "mic": mic, "duration": round(duration, 3),
        **teacher,
    }


def write_session(out: Path, rows: list[dict], vectors: dict[str, list], meta: dict) -> dict:
    """manifest.jsonl + one vector table per encoder + session.json. No audio."""
    from juno_core.slu import data as D

    out.mkdir(parents=True, exist_ok=True)
    D.write_manifest(out / "manifest.jsonl", rows)
    ids = np.asarray([r["id"] for r in rows])
    for spec, vecs in vectors.items():
        np.savez(out / D.table_name(spec), ids=ids, X=np.asarray(vecs, np.float32),
                 encode_ms=np.zeros(len(ids)), spec=np.asarray(spec))
    counts: dict = {}
    for row in rows:
        counts[row["g_addressed"]] = counts.get(row["g_addressed"], 0) + 1
    meta = {**meta, "rows": len(rows), "counts": counts, "encoders": list(vectors),
            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def collect_main(args) -> int:
    from juno_core.slu.encoder import build_encoder
    from juno_core.slu.training import build_teacher

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; use a new directory per session")
    config = None
    if args.config or Path("config.yaml").exists():
        from juno_core.config import load_config

        config = load_config(args.config)
    mic = args.mic
    if not mic:
        from juno_core.audio.voiceprint import current_microphone

        mic = current_microphone() or "unknown"
    print("loading the teacher and encoders...")
    teacher, _ = build_teacher(args)
    encoders = [build_encoder(spec) for spec in args.encoder]
    for encoder in encoders:
        encoder.warmup()
    session = args.session or default_session()
    collector = SLUCollector(config, encoders=encoders, teacher=teacher,
                             speakers=args.speakers.split(","), session=session,
                             room=args.room, mic=mic, consent=args.consent)
    if not collector.confirm_consent():
        print("not started: consent not confirmed")
        return 1
    rows = collector.run(script=SLU_SCRIPT, scale=args.scale)
    if not rows:
        print("nothing collected")
        return 1
    meta = write_session(out, rows, collector.vectors, {
        "session": session, "room": args.room, "mic": mic, "consent": args.consent,
        "speakers": args.speakers})
    print(f"\nwrote {len(rows)} rows and {len(encoders)} vector tables to {out} "
          f"{meta['counts']}; no audio was saved")
    return 0
