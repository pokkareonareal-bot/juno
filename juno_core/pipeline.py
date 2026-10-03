"""Wires the pieces into one running loop: microphone in, an accepted
utterance out.

This is the reference wiring, not the only correct one -- read it, then feel
free to take it apart. It reproduces the order the numbers in REPORT.txt (see
README) were measured against:

    mic -> voice activity -> [was that you? -> is it worth transcribing?]
        -> speech-to-text (yours) -> was that meant for me? -> your callback

With System One switched on (``system_one.mode``, see juno_core/slu), a
fast path sits in front of the transcription:

    mic -> voice activity -> was that you? -> SYSTEM ONE (audio only)
        ignore   -> dropped, nothing transcribed
        act      -> typed Decision straight to your on_decision handler
        escalate -> [gate] -> speech-to-text -> intent -> SYSTEM TWO
                    -> typed Decision (with the transcript) to the same handler

In ``shadow`` mode System One runs and is logged next to System Two on every
turn, and changes nothing.

Two things this file deliberately does NOT reproduce from the original
system, both noted so nobody goes looking for them:

  - The "provisional" head-start transcription (starting STT a couple hundred
    milliseconds early, during the trailing-silence hangover, to shave time
    off every answer). It's a real latency win -- REPORT.txt section 2.7 has
    the numbers -- but it needs careful thread coordination to avoid
    transcribing twice, and getting that subtly wrong is exactly the kind of
    bug this project's own testing notes warn about. Left out of a reference
    implementation on purpose; the gate and the intent engine underneath it
    are unaffected either way.
  - Spoken confirmations and escalation to a stronger model. Those are
    product decisions about what happens after an utterance is accepted, not
    part of deciding whether it was addressed to you.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

from .audio.capture import build_capture
from .audio.vad import SegmentDetector, SpeechSegment, build_vad
from .audio.voiceprint import VoicePrint
from .config import Section
from .events import (
    DECISION_DELIVERED,
    GATE_SCORED,
    INTENT_ACCEPTED,
    INTENT_IGNORED,
    SEGMENT_DROPPED,
    SLU_DECIDED,
    SLU_SYSTEM_TWO,
    SPEECH_START,
    TRANSCRIBE_STARTED,
    TRANSCRIPT_READY,
    TRANSCRIPT_REJECTED,
    Stage,
)
from .intelligence.context import ConversationContext, Utterance
from .intelligence.gate import Acoustics, Gate, Snapshot, verdicts_from
from .intelligence.intent import IntentDecision, IntentEngine
from .slu.router import ConversationState
from .slu.schema import CORE_SCHEMA, Decision, Schema
from .slu.system_one import SystemOne
from .slu.teacher import SystemTwo
from .stt import STTEngine


def _section(config, key: str) -> Section:
    value = config.get(key) if config is not None else None
    return value if value is not None else Section({})


# Called with (text, decision, context) once an utterance is accepted.
# Return the reply text, or None to say nothing (e.g. you handed it off to a
# background agent that will speak later on its own).
AcceptCallback = Callable[[str, IntentDecision, ConversationContext], "str | None"]

# Called with (decision, context) for every utterance meant for the
# assistant, whichever system answered: decision.source says which, and
# decision.transcript is None when no transcription was made. Same return
# convention as AcceptCallback. Register one of these and System One may
# act on its own; without it, there is nothing for a typed answer to go to.
DecisionCallback = Callable[[Decision, ConversationContext], "str | None"]


class JunoPipeline:
    """Owns the microphone and decides what's worth acting on.

    ``on_accept`` is the one seam you're expected to fill in -- see
    examples/connect_an_agent.py. Leave it unset and accepted utterances are
    answered directly by ``model`` (or echoed, if you haven't configured one
    either -- see llm/__init__.py's EchoModel).
    """

    def __init__(
        self,
        config: Section,
        *,
        stt: STTEngine,
        model=None,
        observer=None,
        on_accept: AcceptCallback | None = None,
        on_decision: DecisionCallback | None = None,
        schema: Schema | None = None,
        assistant_name: str | None = None,
    ) -> None:
        self.config = config
        self.observer = observer
        self.stt = stt
        self.model = model
        self.assistant_name = assistant_name or _section(config, "app").get(
            "assistant_name", "Juno")
        self.on_accept = on_accept or self.answer_with_model
        self.on_decision = on_decision

        self.context = ConversationContext()
        self._rate = int(_section(config, "audio").get("sample_rate", 16000))

        self.own_voice = VoicePrint(_section(config, "own_voice"), observer)
        self.gate = Gate(_section(_section(config, "intent"), "gate"), observer)
        self.intent = IntentEngine(
            _section(config, "intent"), self.context, model=model,
            observer=observer, assistant_name=self.assistant_name,
        )

        # System One loads its own model and schema; System Two answers in
        # whichever schema System One was trained on, so the agent sees one
        # vocabulary whichever path a turn took.
        self.system_one = SystemOne(_section(config, "system_one"), observer, schema=schema)
        self.schema = schema or getattr(self.system_one, "schema", None) or CORE_SCHEMA
        self.system_two = SystemTwo(
            self.schema, self.assistant_name, model,
            aliases=_section(config, "intent").get("assistant_aliases") or ())
        # Conversation states the host knows about and Juno cannot hear, e.g.
        # "timer_running" -- intents declare `when` against these.
        self.active_contexts: set[str] = set()

        vad_config = _section(config, "vad")
        backend = build_vad(vad_config, self._rate, observer)
        self.detector = SegmentDetector(
            backend, vad_config, self._rate,
            preroll=float(_section(config, "audio").get("preroll", 0.4)),
        )
        self.detector.on_speech_start = self._on_speech_start

        self.capture = build_capture(_section(config, "audio"), observer)
        self._busy = False
        self._running = False

    # -- the loop -----------------------------------------------------------

    def run_forever(self) -> None:
        """Blocks, listening, until stopped (Ctrl+C or `stop()`)."""
        self.stt.warmup()
        self.system_one.warmup()
        self.capture.start()
        self._running = True
        try:
            for frame in self.capture.frames():
                if not self._running:
                    break
                segment = self.detector.push(frame)
                if segment is not None:
                    self._process_segment(segment)
        finally:
            self.capture.stop()
            self._running = False

    def stop(self) -> None:
        self._running = False

    # -- one utterance --------------------------------------------------------

    def _on_speech_start(self, timestamp: float) -> None:
        self._emit(Stage.VAD, SPEECH_START, None)

    def _gate_snapshot(self) -> Snapshot:
        return Snapshot(
            since_ai=self.context.seconds_since_ai_response(),
            awaiting_answer=self.context.ai_awaiting_answer(),
            confirming=False,
            offer_pending=False,
            busy=self._busy,
            recent_verdicts=verdicts_from(self.context.utterances),
        )

    def _conversation_state(self) -> ConversationState:
        return ConversationState(
            since_ai=self.context.seconds_since_ai_response(),
            awaiting_answer=self.context.ai_awaiting_answer(),
            active=frozenset(self.active_contexts),
        )

    def _process_segment(self, segment: SpeechSegment) -> None:
        turn_id = uuid.uuid4().hex[:8]

        estimate = None
        if self.own_voice.enabled:
            estimate = self.own_voice.estimate(segment.audio, self._rate)
            if estimate.confident and not estimate.is_wearer:
                self._emit(Stage.AUDIO, SEGMENT_DROPPED, turn_id, reason="not_own_voice")
                return

        if self.system_one.enabled:
            fast = self.system_one.decide(segment.audio, self._rate,
                                          self._conversation_state(), turn=turn_id)
            acting = self.system_one.mode == "on"
            can_act = self.on_decision is not None
            effective = acting and (fast.route == "ignore" or (fast.route == "act" and can_act))
            self._emit(Stage.SLU, SLU_DECIDED, turn_id, effective=effective,
                       summary=fast.summary(), p=fast.intent.p if fast.intent else fast.addressed.p,
                       seconds=round(float(segment.duration), 3), **fast.as_json(full=True))
            if effective and fast.route == "ignore":
                self._emit(Stage.AUDIO, SEGMENT_DROPPED, turn_id, reason="system_one")
                return
            if effective and fast.route == "act":
                self.context.add_utterance(Utterance(
                    text=f"({fast.summary()})", timestamp=time.monotonic(),
                    ai_directed=True, confidence=fast.addressed.p,
                    duration=float(segment.duration)))
                self._deliver(fast)
                return

        acoustics = Acoustics(
            seconds=float(segment.duration),
            confidence=float(segment.confidence),
            truncated=bool(segment.truncated),
            p_own=(estimate.p_own if estimate is not None and estimate.confident else None),
            voice_confident=bool(estimate is not None and estimate.confident),
            is_wearer=(estimate.is_wearer if estimate is not None and estimate.confident else None),
        )
        snapshot = self._gate_snapshot()
        # The segment's audio lives only in memory: scored here, handed to
        # STT below, then released. Nothing in this loop writes it to disk --
        # the gate's log carries numbers (score, signals, optionally the
        # feature vector), never samples.
        decision = self.gate.score(acoustics, snapshot, audio=segment.audio,
                                    sample_rate=self._rate)
        self._emit(Stage.SYSTEM, GATE_SCORED, turn_id,
                   **decision.as_log(include_features=self.gate.log_features))
        if decision.skip:
            self._emit(Stage.AUDIO, SEGMENT_DROPPED, turn_id, reason="gate")
            return

        self._busy = True
        try:
            self._emit(Stage.STT, TRANSCRIBE_STARTED, turn_id)
            transcript = self.stt.transcribe(segment.audio, self._rate)
        finally:
            self._busy = False

        if not transcript or not transcript.text.strip():
            self._emit(Stage.STT, TRANSCRIPT_REJECTED, turn_id, reason="empty")
            return
        self._emit(Stage.STT, TRANSCRIPT_READY, turn_id, text=transcript.text)

        own_voice_value = (
            estimate.p_own if estimate is not None and estimate.confident else None
        )
        decision2 = self.intent.classify(
            transcript.text,
            reliable=transcript.reliable,
            spoken_at=segment.start_time,
            tail_reliable=transcript.tail_reliable,
            own_voice=own_voice_value,
        )
        self.intent.record(transcript.text, decision2, duration=segment.duration)

        slow = None
        if self.system_one.enabled or self.on_decision is not None:
            slow = self.system_two.decide(
                transcript.text, decision2, reliable=transcript.reliable, turn=turn_id,
                latency_ms={"stt": transcript.latency * 1000.0,
                            "intent": decision2.latency * 1000.0})
            self._emit(Stage.SLU, SLU_SYSTEM_TWO, turn_id, **slow.as_json())

        if not decision2.ai_intent:
            self._emit(Stage.INTENT, INTENT_IGNORED, turn_id,
                       confidence=round(decision2.confidence, 3))
            return

        self._emit(Stage.INTENT, INTENT_ACCEPTED, turn_id,
                   confidence=round(decision2.confidence, 3))
        self.context.add_user_turn(transcript.text)
        if self.on_decision is not None and slow is not None:
            self._deliver(slow, user_turn_added=True)
            return
        self._busy = True
        try:
            reply = self.on_accept(transcript.text, decision2, self.context)
        finally:
            self._busy = False
        self._finish(reply)

    def _deliver(self, decision: Decision, user_turn_added: bool = False) -> None:
        """Hand a typed Decision to the agent, from either system."""
        if not user_turn_added:
            self.context.add_user_turn(f"({decision.summary()})")
        self._emit(Stage.SLU, DECISION_DELIVERED, decision.turn, source=decision.source,
                   summary=decision.summary())
        self._busy = True
        try:
            reply = self.on_decision(decision, self.context)
        finally:
            self._busy = False
        self._finish(reply)

    def _finish(self, reply: str | None) -> None:
        if reply:
            # Stamps the follow-up-window clock too (see ConversationContext).
            # If you add real TTS, call context.note_ai_spoke() instead, once
            # playback actually finishes -- that's the more honest moment for
            # "how long ago did it last speak".
            self.context.add_assistant_turn(reply)

    # -- the default answerer, if you didn't bring your own agent -----------

    def answer_with_model(self, text: str, decision: IntentDecision,
                            context: ConversationContext) -> str:
        if self.model is None:
            return f"(heard you, but no llm.provider is configured -- see README)"
        messages = [
            {"role": "system", "content": (
                f"You are {self.assistant_name}, a spoken voice assistant. "
                f"Answer in one or two short sentences -- this is read aloud, "
                f"not read on a screen."
            )},
            *context.messages(turns=6),
        ]
        return self.model.complete(messages, max_tokens=200, temperature=0.6)

    def _emit(self, stage: Stage, name: str, turn_id: str | None, **fields) -> None:
        if self.observer is not None:
            self.observer.emit(stage, name, turn_id, **fields)
