"""System Two: speech-to-text, then the text-side engines, answering in the schema.

This is the expensive path Juno already had -- transcribe, decide whether it
was meant for us (intent.py, with its language-model second opinion), and
now also say WHAT was meant (parse.py) -- wrapped so that it produces the
same typed Decision System One does. It has two jobs:

  1. At runtime, it is where System One escalates to. The agent receives its
     Decision with ``source: system_two`` and the transcript attached.
  2. Offline, it is the TEACHER. ``Teacher.label`` runs it over a clip and
     returns its full, soft answer -- the probabilities, not just the top
     choice -- which is what the student is trained to reproduce.

The teacher may be stronger offline than at runtime, and should be: a bigger
recogniser (parakeet-0.6b, whisper large-v3-turbo) and the adjudicator on
every ambiguous utterance cost nothing that matters when nobody is waiting.

HOW "WHO WAS IT FOR" BECOMES THREE PROBABILITIES
------------------------------------------------
The intent engine answers one question: p(meant for the assistant). The
student is asked three-way, because "a person" and "the television" are
different reasons to ignore something and are worth telling apart in a
report. The remainder of the probability is split between them on what the
transcript looked like: a reliable transcript of words is a person; nothing
intelligible, or an unreliable transcript, leans background.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from juno_core.intelligence.context import ConversationContext
from juno_core.intelligence.intent import IntentDecision, IntentEngine
from juno_core.slu.parse import TextParse, TextParser
from juno_core.slu.schema import (
    ADDRESSEES, ASSISTANT, CORE_SCHEMA, OPEN_REQUEST, Decision, Field, Schema,
)


def addressed_distribution(decision: IntentDecision | None, text: str,
                           reliable: bool = True) -> dict[str, float]:
    """p over ADDRESSEES from the intent engine's verdict and the transcript."""
    if decision is None or not text.strip():
        return {ASSISTANT: 0.02, "human_directed": 0.18, "background_or_media": 0.80}
    p = max(0.0, min(1.0, float(decision.confidence)))
    human_share = 0.85 if reliable else 0.35
    if len(text.split()) <= 1 and not decision.ai_intent:
        human_share = min(human_share, 0.6)
    return {ASSISTANT: p, "human_directed": (1 - p) * human_share,
            "background_or_media": (1 - p) * (1 - human_share)}


class SystemTwo:
    """Turns (transcript, intent verdict) into a typed Decision."""

    def __init__(self, schema: Schema = CORE_SCHEMA, assistant_name: str = "Juno",
                 model=None, aliases=()) -> None:
        self.schema = schema
        self.parser = TextParser(schema, assistant_name=assistant_name, model=model,
                                 aliases=tuple(aliases or ()))

    def decide(self, text: str, verdict: IntentDecision | None, *, reliable: bool = True,
               turn: str | None = None, latency_ms: dict | None = None) -> Decision:
        dist = addressed_distribution(verdict, text, reliable)
        accepted = bool(verdict is not None and verdict.ai_intent)
        who = ASSISTANT if accepted else max(
            (a for a in ADDRESSEES if a != ASSISTANT), key=lambda a: dist[a])
        addressed = Field(who, dist[who], dist)
        if not accepted:
            return Decision(route="ignore", source="system_two", addressed=addressed,
                            transcript=text or None, turn=turn,
                            reason="the intent engine says it was not for the assistant",
                            latency_ms=dict(latency_ms or {}), detail=verdict)
        parsed = self.parser.parse(text)
        spec = self.schema.intent(parsed.intent)
        slots = {k: Field(v, parsed.p) for k, v in parsed.slots.items()}
        reason = f"parsed by {parsed.method}"
        if spec is not None and spec.slots and any(
                s.required and s.name not in slots for s in spec.slots):
            parsed = TextParse(OPEN_REQUEST, parsed.p, {}, parsed.method, parsed.normalised)
            slots, reason = {}, "a required slot was not said"
        return Decision(route="act", source="system_two", addressed=addressed,
                        intent=Field(parsed.intent, parsed.p), slots=slots,
                        transcript=text, turn=turn, reason=reason,
                        latency_ms=dict(latency_ms or {}), detail=verdict)


@dataclass
class TeacherLabel:
    """Everything the teacher said about one clip -- a training target."""

    transcript: str
    reliable: bool
    accepted: bool
    confidence: float
    method: str
    addressed: dict[str, float]
    intent: str | None
    intent_p: float
    slots: dict[str, Any] = field(default_factory=dict)
    stt_ms: float = 0.0
    text_ms: float = 0.0

    def as_row(self) -> dict:
        return {
            "transcript": self.transcript, "t_reliable": self.reliable,
            "t_accept": self.accepted, "t_confidence": round(self.confidence, 4),
            "t_method": self.method,
            "t_addressed": {k: round(v, 4) for k, v in self.addressed.items()},
            "t_intent": self.intent, "t_intent_p": round(self.intent_p, 4),
            "t_slots": self.slots, "stt_ms": round(self.stt_ms, 2),
            "text_ms": round(self.text_ms, 3),
        }


class Teacher:
    """STT + intent engine + parser, run cold on one clip at a time.

    Each clip is judged with a fresh conversation context: offline clips have
    no honest conversational history, and borrowing the previous clip's
    would teach the student an ordering that only exists in the manifest.
    """

    def __init__(self, stt, *, intent_config=None, schema: Schema = CORE_SCHEMA,
                 assistant_name: str = "Juno", model=None) -> None:
        self.stt = stt
        self.intent_config = intent_config or {}
        self.model = model
        self.assistant_name = assistant_name
        aliases = self.intent_config.get("assistant_aliases") or ()
        self.system_two = SystemTwo(schema or CORE_SCHEMA, assistant_name, model, aliases)

    def label(self, audio, sample_rate: int = 16000) -> TeacherLabel:
        t0 = time.perf_counter()
        transcript = self.stt.transcribe(audio, sample_rate)
        stt_ms = (time.perf_counter() - t0) * 1000.0
        text = transcript.text.strip() if transcript else ""
        t1 = time.perf_counter()
        verdict = None
        if text:
            engine = IntentEngine(self.intent_config, ConversationContext(), model=self.model,
                                  assistant_name=self.assistant_name)
            verdict = engine.classify(text, reliable=transcript.reliable,
                                      tail_reliable=transcript.tail_reliable)
        decision = self.system_two.decide(text, verdict, reliable=bool(transcript and transcript.reliable))
        # The intent head is taught on what the words say whether or not the
        # engine accepted them -- build_targets masks by p(assistant) later.
        parsed = self.system_two.parser.parse(text) if text else None
        text_ms = (time.perf_counter() - t1) * 1000.0
        return TeacherLabel(
            transcript=text,
            reliable=bool(transcript and transcript.reliable),
            accepted=decision.route == "act",
            confidence=float(verdict.confidence) if verdict else 0.0,
            method=verdict.method if verdict else "no_transcript",
            addressed=dict(decision.addressed.probs or {}),
            intent=parsed.intent if parsed else None,
            intent_p=float(parsed.p) if parsed else 0.0,
            slots=dict(parsed.slots) if parsed else {},
            stt_ms=stt_ms, text_ms=text_ms,
        )
