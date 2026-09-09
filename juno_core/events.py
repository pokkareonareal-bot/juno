"""Names for the things that happen on the way to an answer.

Every stage below emits through the Observer (see observability.py) instead
of printing ad hoc, so one utterance can be reconstructed end to end and each
stage's latency measured on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Stage(str, Enum):
    AUDIO = "audio"
    VAD = "vad"
    STT = "stt"
    INTENT = "intent"
    LLM = "llm"
    SYSTEM = "system"


# Event names, kept as plain constants so a log file stays greppable.
SPEECH_START = "speech_start"
SPEECH_END = "speech_end"
SEGMENT_DROPPED = "segment_dropped"
# Before transcription: whether the expensive path is worth attempting at all.
GATE_SCORED = "gate_scored"
PROVISIONAL_STARTED = "provisional_started"
PROVISIONAL_USED = "provisional_used"
PROVISIONAL_MISSED = "provisional_missed"
SESSION_START = "session_start"
TRANSCRIBE_STARTED = "transcribe_started"
TRANSCRIPT_READY = "transcript_ready"
TRANSCRIPT_REJECTED = "transcript_rejected"
INTENT_SCORED = "intent_scored"
INTENT_ADJUDICATED = "intent_adjudicated"
INTENT_ACCEPTED = "intent_accepted"
INTENT_IGNORED = "intent_ignored"
LLM_STARTED = "llm_started"
LLM_FINISHED = "llm_finished"
ERROR = "error"


@dataclass
class Event:
    """One observation from one stage of one turn."""

    stage: Stage
    name: str
    turn_id: str | None = None
    wall_time: float = 0.0
    mono_time: float = 0.0
    fields: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        payload = {
            "t": round(self.wall_time, 6),
            "mono": round(self.mono_time, 6),
            "stage": self.stage.value,
            "event": self.name,
        }
        if self.turn_id:
            payload["turn"] = self.turn_id
        payload.update(self.fields)
        return payload
