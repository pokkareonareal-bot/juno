"""From probabilities to one of three routes: act, ignore, escalate.

The student says how likely each answer is. This decides what Juno DOES
with that, and it is deliberately the only place that decides, so every
rule about when System One may act alone is in one short file.

    ignore    p(assistant_directed) below ``ignore_below``, and nothing about
              the conversation says the next words are likely ours
    act       p(assistant_directed) at least ``act_addressed``; the top
              intent is a typed one (not open_request), at least its
              threshold, allowed in the current state; every required slot
              at least ``act_slot``
    escalate  everything else

TWO ASYMMETRIES
---------------
A wrong ``ignore`` is a missed request the slow path never sees: the person
repeats themselves, or gives up. A wrong ``act`` is worse: a timer nobody
asked for, music paused mid-sentence, Juno answering a remark to somebody
else -- the false activation the intent engine treats as three times worse
than a miss. ``escalate`` costs only time and compute. So both thresholds
are chosen on held-out data for a small error BUDGET (see training.py), and
uncertainty always lands on ``escalate``.

No thresholds, no shortcuts: a model file without them (or with them set to
null) makes every decision ``escalate``. That is the same rule the gate
follows -- a threshold has to be measured, never assumed.

WHAT CONTEXT VETOES
-------------------
``ignore`` is vetoed when the assistant just asked something, a
confirmation or offer is open, or the follow-up window is running -- the
same list as the gate's ``_context_must``, for the same reason: right after
Juno speaks, the next utterance is very likely for it, and a "yes" lost
there cannot be recovered. ``act`` needs the intent's declared state: a bare
"yes" is ``confirm.yes`` only while a question is open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from juno_core.slu.schema import ASSISTANT, OPEN_REQUEST, Field, Schema
from juno_core.slu.student import OTHER, Prediction


@dataclass
class Thresholds:
    ignore_below: float | None = None
    act_addressed: float | None = None
    act_intent: float | None = None
    act_slot: float | None = None
    per_intent: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict | None) -> "Thresholds":
        data = data or {}

        def num(key):
            value = data.get(key)
            return None if value is None else float(value)

        return cls(ignore_below=num("ignore_below"), act_addressed=num("act_addressed"),
                   act_intent=num("act_intent"), act_slot=num("act_slot"),
                   per_intent={k: float(v) for k, v in (data.get("per_intent") or {}).items()})

    def to_dict(self) -> dict:
        return {"ignore_below": self.ignore_below, "act_addressed": self.act_addressed,
                "act_intent": self.act_intent, "act_slot": self.act_slot,
                "per_intent": dict(self.per_intent)}

    @property
    def can_ignore(self) -> bool:
        return self.ignore_below is not None

    @property
    def can_act(self) -> bool:
        return None not in (self.act_addressed, self.act_intent, self.act_slot)


@dataclass(frozen=True)
class ConversationState:
    """What the conversation is doing as the utterance arrives."""

    since_ai: float = float("inf")       # seconds since the assistant last spoke
    awaiting_answer: bool = False
    confirming: bool = False
    offer_pending: bool = False
    active: frozenset = frozenset()       # e.g. {"timer_running"} -- set by the host

    def contexts(self) -> set[str]:
        out = set(self.active)
        if self.awaiting_answer or self.confirming or self.offer_pending:
            out.add("awaiting_answer")
        return out


@dataclass
class Routed:
    route: str
    reason: str
    addressed: Field
    intent: Field | None
    slots: dict[str, Field]


class Router:
    def __init__(self, schema: Schema, thresholds: Thresholds,
                 followup_window: float = 20.0) -> None:
        self.schema = schema
        self.thresholds = thresholds
        self.followup_window = followup_window

    def ignore_veto(self, state: ConversationState) -> str:
        if state.confirming:
            return "a confirmation is open"
        if state.offer_pending:
            return "an offer is waiting for an answer"
        if state.awaiting_answer:
            return "the assistant asked a question"
        if state.since_ai < self.followup_window:
            return "inside the follow-up window"
        return ""

    def intent_threshold(self, name: str) -> float | None:
        t = self.thresholds
        candidates = [t.act_intent, t.per_intent.get(name)]
        spec = self.schema.intent(name)
        if spec is not None and spec.min_confidence is not None:
            candidates.append(spec.min_confidence)
        if t.act_intent is None:
            return None
        return max(c for c in candidates if c is not None)

    def route(self, pred: Prediction, state: ConversationState | None = None) -> Routed:
        state = state or ConversationState()
        t = self.thresholds
        who, p_who = pred.top(pred.addressed)
        p_assistant = float(pred.addressed.get(ASSISTANT, 0.0))
        addressed = Field(who, p_who, dict(pred.addressed))
        intent_name, p_intent = pred.top(pred.intent) if pred.intent else (None, 0.0)
        intent = Field(intent_name, p_intent) if intent_name else None
        slots = self._slots(intent_name, pred)

        # -- ignore ----------------------------------------------------------
        if t.can_ignore and p_assistant < t.ignore_below:
            veto = self.ignore_veto(state)
            if not veto:
                return Routed("ignore", f"p(assistant)={p_assistant:.3f} < {t.ignore_below:.3f}",
                              addressed, None, {})
            return Routed("escalate", f"would ignore, but {veto}", addressed, intent, slots)

        # -- act -------------------------------------------------------------
        why_not = self._why_not_act(p_assistant, intent_name, p_intent, slots, state)
        if not why_not:
            return Routed("act", "confident and complete", addressed, intent, slots)
        return Routed("escalate", why_not, addressed, intent, slots)

    def _why_not_act(self, p_assistant, intent_name, p_intent, slots, state) -> str:
        t = self.thresholds
        if not t.can_act:
            return "model has no act thresholds"
        if p_assistant < t.act_addressed:
            return f"p(assistant)={p_assistant:.3f} < {t.act_addressed:.3f}"
        spec = self.schema.intent(intent_name) if intent_name else None
        if spec is None:
            return "no intent"
        if spec.name == OPEN_REQUEST or spec.needs_transcript:
            return f"{spec.name} needs the words"
        threshold = self.intent_threshold(spec.name)
        if p_intent < threshold:
            return f"p({spec.name})={p_intent:.3f} < {threshold:.3f}"
        missing = [c for c in spec.when if c not in state.contexts()]
        if missing:
            return f"{spec.name} only applies when {missing[0]}"
        for slot in spec.slots:
            got = slots.get(slot.name)
            if slot.required and got is None:
                return f"{spec.name} needs {slot.name}"
            if got is not None and got.value == OTHER:
                return f"{slot.name} is not one of the typed values"
            if got is not None and got.p < t.act_slot:
                return f"p({slot.name}={got.value})={got.p:.3f} < {t.act_slot:.3f}"
        return ""

    def _slots(self, intent_name: str | None, pred: Prediction) -> dict[str, Field]:
        out = {}
        if not intent_name:
            return out
        for key, dist in pred.slots.items():
            name, slot = key.split(":", 1)
            if name != intent_name or not dist:
                continue
            value, p = pred.top(dist)
            out[slot] = Field(value, p)
        return out
