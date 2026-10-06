"""The cascade: speech-to-text, then the text-side engines, answering in the schema.

This is the expensive path Juno already had -- transcribe, decide whether it
was meant for us (intent.py, with its language-model second opinion), and
now also say WHAT was meant (parse.py) -- wrapped so that it produces the
same typed Decision Reflex does. It has two jobs:

  1. At runtime, it is where Reflex escalates to. The agent receives its
     Decision (``source`` is ``SOURCE_CASCADE``) and the transcript attached.
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
    ADDRESSEES, ASSISTANT, CORE_SCHEMA, OPEN_REQUEST, SOURCE_CASCADE, Decision, Field, Schema,
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


class Cascade:
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
            return Decision(route="ignore", source=SOURCE_CASCADE, addressed=addressed,
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
        return Decision(route="act", source=SOURCE_CASCADE, addressed=addressed,
                        intent=Field(parsed.intent, parsed.p), slots=slots,
                        transcript=text, turn=turn, reason=reason,
                        latency_ms=dict(latency_ms or {}), detail=verdict)


SystemTwo = Cascade   # the class's old name, for code written before the rename


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
    intent_probs: dict[str, float] | None = None     # the judge's full distribution
    teacher: str = "engine"
    extra: dict = field(default_factory=dict)        # the separate opinions, when blended

    def as_row(self) -> dict:
        row = {
            "transcript": self.transcript, "t_reliable": self.reliable,
            "t_accept": self.accepted, "t_confidence": round(self.confidence, 4),
            "t_method": self.method,
            "t_addressed": {k: round(v, 4) for k, v in self.addressed.items()},
            "t_intent": self.intent, "t_intent_p": round(self.intent_p, 4),
            "t_slots": self.slots, "stt_ms": round(self.stt_ms, 2),
            "text_ms": round(self.text_ms, 3), "t_teacher": self.teacher,
        }
        if self.intent_probs:
            row["t_intent_probs"] = {k: round(v, 4) for k, v in self.intent_probs.items()}
        row.update(self.extra)
        return row


class Teacher:
    """STT, then the words judged: run cold on one clip at a time.

    Two ways to judge the words:

      engine  the runtime's intent engine and parser, exactly as the cascade
              runs them (the first teacher -- conservative when cold);
      judge   a language model asked who it was for and what was wanted, as
              probabilities (judge.py), blended with the engine's verdict by
              ``judge_weight`` (1.0 = the judge alone). The parser still
              decides what it parses exactly, and every slot value.

    Each clip is judged with a fresh conversation context: offline clips have
    no honest conversational history, and borrowing the previous clip's
    would teach the student an ordering that only exists in the manifest.
    """

    def __init__(self, stt, *, intent_config=None, schema: Schema = CORE_SCHEMA,
                 assistant_name: str = "Juno", model=None, judge=None,
                 judge_weight: float = 0.8) -> None:
        self.stt = stt
        self.intent_config = intent_config or {}
        self.model = model
        self.assistant_name = assistant_name
        self.judge = judge
        self.judge_weight = float(judge_weight)
        aliases = self.intent_config.get("assistant_aliases") or ()
        self.cascade = Cascade(schema or CORE_SCHEMA, assistant_name, model, aliases)

    @property
    def name(self) -> str:
        if self.judge is None:
            return "engine"
        return f"{self.judge.tag}x{self.judge_weight:g}+engine"

    def label(self, audio, sample_rate: int = 16000) -> TeacherLabel:
        t0 = time.perf_counter()
        transcript = self.stt.transcribe(audio, sample_rate)
        stt_ms = (time.perf_counter() - t0) * 1000.0
        text = transcript.text.strip() if transcript else ""
        out = self.judge_text(text, reliable=bool(transcript and transcript.reliable),
                              tail_reliable=bool(transcript and transcript.tail_reliable))
        out.stt_ms = stt_ms
        return out

    def judge_text(self, text: str, reliable: bool = True,
                   tail_reliable: bool = True) -> TeacherLabel:
        """The teacher's answer for a transcript (``relabel`` uses this directly)."""
        t1 = time.perf_counter()
        verdict = None
        if text:
            engine = IntentEngine(self.intent_config, ConversationContext(), model=self.model,
                                  assistant_name=self.assistant_name)
            verdict = engine.classify(text, reliable=reliable, tail_reliable=tail_reliable)
        decision = self.cascade.decide(text, verdict, reliable=reliable)
        # The intent head is taught on what the words say whether or not the
        # engine accepted them -- build_targets masks by p(assistant) later.
        parsed = self.cascade.parser.parse(text) if text else None
        addressed = dict(decision.addressed.probs or {})
        intent = parsed.intent if parsed else None
        intent_p = float(parsed.p) if parsed else 0.0
        slots = dict(parsed.slots) if parsed else {}
        intent_probs = None
        method = verdict.method if verdict else "no_transcript"

        extra: dict = {}
        if self.judge is not None and text:
            j = self.judge.judge(text)
            w = self.judge_weight
            # Both opinions are kept, so other blends can be measured later
            # without asking the model again.
            extra = {"t_engine_addressed": {k: round(v, 4) for k, v in addressed.items()},
                     "t_judge_addressed": {k: round(v, 4) for k, v in j.addressed.items()}}
            addressed = {a: w * j.addressed[a] + (1 - w) * addressed.get(a, 0.0) for a in ADDRESSEES}
            if verdict is not None and verdict.method in ("named", "retry"):
                # The engine heard the name said TO the assistant (a vocative,
                # not a mention): at runtime that settles it, so it does here.
                addressed = {ASSISTANT: 0.98, "human_directed": 0.01,
                             "background_or_media": 0.01}
            method = f"judge+{method}"
            if j.intent:
                intent_probs = dict(j.intent)
                top = max(j.intent, key=j.intent.get)
                exact = parsed is not None and parsed.method in ("rule", "example") \
                    and parsed.intent != OPEN_REQUEST
                if exact:
                    # The rules parsed it exactly: that intent, with the judge's
                    # distribution sharpened onto it.
                    intent_probs = {k: 0.1 * v for k, v in intent_probs.items()}
                    intent_probs[intent] = intent_probs.get(intent, 0.0) + 0.9
                else:
                    spec = self.cascade.schema.intent(top)
                    needs = [s.name for s in spec.slots if s.required] if spec else []
                    if spec is not None and not any(n not in slots for n in needs):
                        intent, intent_p = top, float(j.intent[top])
                    else:
                        # A timer with no length the parser could read is not
                        # a typed request: the words are needed.
                        intent, intent_p = OPEN_REQUEST, float(j.intent.get(OPEN_REQUEST, 0.0))
                        intent_probs = None
                        slots = {}
            accepted = addressed[ASSISTANT] >= 0.5
            confidence = addressed[ASSISTANT]
        else:
            accepted = decision.route == "act"
            confidence = float(verdict.confidence) if verdict else 0.0
        return TeacherLabel(
            transcript=text, reliable=reliable, accepted=accepted, confidence=confidence,
            method=method, addressed=addressed, intent=intent, intent_p=intent_p, slots=slots,
            text_ms=(time.perf_counter() - t1) * 1000.0, intent_probs=intent_probs,
            teacher=self.name, extra=extra,
        )
