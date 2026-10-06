"""The typed contract between Juno and whatever is plugged into it.

An agent DECLARES what it can do -- a list of intents, each with typed slots
-- and Juno ANSWERS in that shape, for every utterance, with a probability on
every field. The agent never has to parse prose and never has to care how the
answer was produced: by the small audio model alone (Reflex SLU), or by
speech-to-text plus the text-side engines (the cascade).

    schema = Schema.from_dict({"name": "my-agent", "intents": [
        {"name": "lights.off", "description": "turn the lights off",
         "examples": ["lights off", "turn off the lights"]},
    ]})

    decision.as_json() ->
    {
      "route": "act",                         # act | ignore | escalate
      "source": "system_one",                 # system_one = Reflex SLU | system_two = the cascade
      "addressed": {"value": "assistant_directed", "p": 0.97},
      "intent":    {"value": "timer.set",  "p": 0.91},
      "slots":     {"duration": {"value": 420, "p": 0.88}},
      "transcript": null,                      # only when the cascade ran
      ...
    }

THE THREE ROUTES
----------------
``ignore``    not meant for the assistant. Nothing downstream is called.
``act``       confident and complete: a known intent, every required slot
              filled, every probability above its threshold. The agent gets
              the JSON and no transcript, because none was made.
``escalate``  anything else -- uncertain, open-ended ("what's the capital of
              Mongolia"), or an intent that needs exact words. The cascade
              runs and the agent gets the same shape back, plus the
              transcript.

An open-ended request is not a failure of Reflex. Deciding THAT
something is a question for the agent, and handing it on, is its job;
carrying the content of the question is what words are for.

WHY THRESHOLDS LIVE HERE
------------------------
"Act on lights.off above 0.9, on timer.set above 0.8" is a product decision,
so the agent can state it per intent (``min_confidence``). The model file
carries the thresholds chosen on held-out data; the stricter of the two
wins. Neither can loosen the other.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_FORMAT = "juno-schema/1"
DECISION_FORMAT = "juno-decision/1"

ROUTES = ("act", "ignore", "escalate")

# Decision.source as it goes out. Reflex SLU used to be called "System One" and
# the cascade "System Two", and agents already read those spellings, so they stay
# what is emitted until those agents have moved. Everything in this repo goes
# through these constants: changing the spelling later means changing these two
# lines (and swapping SOURCE_ALIASES so the old spellings are still read).
SOURCE_REFLEX = "system_one"
SOURCE_CASCADE = "system_two"
SOURCES = (SOURCE_REFLEX, SOURCE_CASCADE)
# Spellings also accepted when a Decision is built, mapped to the emitted one.
SOURCE_ALIASES = {"reflex": SOURCE_REFLEX, "cascade": SOURCE_CASCADE}

# Who an utterance was for. The same names gate_training.py uses, so feature
# tables, student rows and decisions all speak one vocabulary.
ADDRESSEES = ("assistant_directed", "human_directed", "background_or_media")
ASSISTANT = "assistant_directed"

# The intent that means "meant for the assistant, but not one of the typed
# ones -- the words are needed". Always present, always escalates.
OPEN_REQUEST = "open_request"

SLOT_TYPES = ("duration", "number", "enum")

# Conversation states an intent can be bound to. A bare "yes" is an answer
# only when a question is open; outside it, it is somebody else's "yes".
CONTEXTS = ("awaiting_answer", "assistant_speaking", "timer_running")

_NAME = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")


class SchemaError(ValueError):
    """A schema an agent declared that cannot be answered in."""


@dataclass(frozen=True)
class SlotSpec:
    """One typed field of an intent.

    Reflex is a classifier, so every slot it fills is a choice among
    ``values``: for ``duration`` that is a list of seconds (the ones people
    actually say -- 7 minutes yes, 7 minutes 13 seconds no); for ``enum``
    the declared options; for ``number`` the declared integers. A value
    outside the list is not an error: Reflex cannot be confident about
    it, so the utterance escalates and the cascade fills the slot from the
    transcript, with whatever value was said.
    """

    name: str
    type: str
    values: tuple = ()
    required: bool = True

    def __post_init__(self) -> None:
        if not _NAME.match(self.name) or "." in self.name:
            raise SchemaError(f"slot name {self.name!r} must be lower_snake_case")
        if self.type not in SLOT_TYPES:
            raise SchemaError(f"slot {self.name}: type must be one of {SLOT_TYPES}")
        if not self.values:
            raise SchemaError(f"slot {self.name}: declare its values (Reflex "
                              f"chooses among them)")
        if len(set(self.values)) != len(self.values):
            raise SchemaError(f"slot {self.name}: duplicate values")

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.type, "values": list(self.values),
                "required": self.required}


@dataclass(frozen=True)
class IntentSpec:
    """One thing the agent can do, and how to recognise a request for it.

    ``examples`` are phrasings. They are what the training tools synthesise
    speech from when an agent adds an intent of its own, and what the text
    parser falls back on, so a handful of varied ones is worth more than a
    long list of near-duplicates. ``{slot}`` placeholders mark where a slot
    value goes: "set a timer for {duration}".
    """

    name: str
    description: str = ""
    examples: tuple[str, ...] = ()
    slots: tuple[SlotSpec, ...] = ()
    min_confidence: float | None = None
    needs_transcript: bool = False
    when: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _NAME.match(self.name):
            raise SchemaError(f"intent name {self.name!r} must look like 'timer.set'")
        if self.min_confidence is not None and not 0.0 < self.min_confidence <= 1.0:
            raise SchemaError(f"{self.name}: min_confidence must be in (0, 1]")
        unknown = [c for c in self.when if c not in CONTEXTS]
        if unknown:
            raise SchemaError(f"{self.name}: unknown context {unknown[0]!r} "
                              f"(known: {', '.join(CONTEXTS)})")
        names = [s.name for s in self.slots]
        if len(set(names)) != len(names):
            raise SchemaError(f"{self.name}: duplicate slot names")

    def slot(self, name: str) -> SlotSpec | None:
        return next((s for s in self.slots if s.name == name), None)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"name": self.name}
        if self.description:
            out["description"] = self.description
        if self.examples:
            out["examples"] = list(self.examples)
        if self.slots:
            out["slots"] = [s.to_dict() for s in self.slots]
        if self.min_confidence is not None:
            out["min_confidence"] = self.min_confidence
        if self.needs_transcript:
            out["needs_transcript"] = True
        if self.when:
            out["when"] = list(self.when)
        return out


@dataclass(frozen=True)
class Schema:
    """Everything the agent declared, plus the always-present open request."""

    name: str
    intents: tuple[IntentSpec, ...]
    version: str = "1"

    def __post_init__(self) -> None:
        names = [i.name for i in self.intents]
        if len(set(names)) != len(names):
            raise SchemaError("duplicate intent names")
        if OPEN_REQUEST not in names:
            object.__setattr__(self, "intents", tuple(self.intents) + (OPEN_REQUEST_SPEC,))

    # -- lookups -------------------------------------------------------------

    @property
    def intent_names(self) -> tuple[str, ...]:
        return tuple(i.name for i in self.intents)

    def intent(self, name: str) -> IntentSpec | None:
        return next((i for i in self.intents if i.name == name), None)

    def slot_keys(self) -> tuple[str, ...]:
        """Every (intent, slot) pair, as 'intent:slot' -- one head each."""
        return tuple(f"{i.name}:{s.name}" for i in self.intents for s in i.slots)

    # -- composition ---------------------------------------------------------

    def extend(self, other: "Schema") -> "Schema":
        """This schema plus the other's intents. The other wins on a clash,
        so an agent can redeclare a core intent with its own threshold."""
        theirs = {i.name for i in other.intents if i.name != OPEN_REQUEST}
        mine = tuple(i for i in self.intents if i.name not in theirs and i.name != OPEN_REQUEST)
        merged = mine + tuple(i for i in other.intents if i.name != OPEN_REQUEST)
        return Schema(name=f"{self.name}+{other.name}", intents=merged,
                      version=f"{self.version}+{other.version}")

    # -- (de)serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        return {"format": SCHEMA_FORMAT, "name": self.name, "version": self.version,
                "intents": [i.to_dict() for i in self.intents]}

    @classmethod
    def from_dict(cls, data: dict) -> "Schema":
        if not isinstance(data, dict) or "intents" not in data:
            raise SchemaError("a schema is an object with an 'intents' list")
        intents = []
        for raw in data["intents"]:
            slots = tuple(
                SlotSpec(name=s["name"], type=s["type"],
                         values=tuple(s.get("values") or ()),
                         required=bool(s.get("required", True)))
                for s in raw.get("slots") or ())
            intents.append(IntentSpec(
                name=raw["name"], description=raw.get("description", ""),
                examples=tuple(raw.get("examples") or ()), slots=slots,
                min_confidence=raw.get("min_confidence"),
                needs_transcript=bool(raw.get("needs_transcript", False)),
                when=tuple(raw.get("when") or ()),
            ))
        return cls(name=str(data.get("name", "custom")), intents=tuple(intents),
                   version=str(data.get("version", "1")))

    @classmethod
    def load(cls, path: str | Path) -> "Schema":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


OPEN_REQUEST_SPEC = IntentSpec(
    name=OPEN_REQUEST,
    description="meant for the assistant, but needs the words: a question, "
                "a request with free-form content, anything not typed",
    needs_transcript=True,
)


# -- the core schema ---------------------------------------------------------
#
# Short, closed, and chosen for what Reflex can plausibly carry from
# sound alone: commands people say in a few words, the same few ways. Things
# with open content -- "remind me to call mum", "what's the weather in
# Lisbon" -- are open requests on purpose.

# The durations people actually ask for, in seconds. A timer for 13 minutes
# is a perfectly good request; it just is not one Reflex will claim to
# have heard exactly, so it escalates.
TIMER_DURATIONS = (
    30, 60, 90, 120, 180, 240, 300, 360, 420, 480, 540, 600, 720, 900,
    1200, 1500, 1800, 2400, 2700, 3600, 5400, 7200,
)

CORE_SCHEMA = Schema(
    name="juno-core",
    version="1",
    intents=(
        IntentSpec("stop", "stop talking or stop what is playing",
                   ("stop", "stop it", "that's enough", "okay stop", "shush")),
        IntentSpec("cancel", "drop the current request",
                   ("never mind", "cancel that", "forget it", "cancel")),
        IntentSpec("confirm.yes", "answer yes to the assistant's question",
                   ("yes", "yeah", "yes please", "go ahead", "sure", "do it"),
                   when=("awaiting_answer",)),
        IntentSpec("confirm.no", "answer no to the assistant's question",
                   ("no", "no thanks", "nope", "don't", "no, leave it"),
                   when=("awaiting_answer",)),
        IntentSpec("repeat", "say the last answer again",
                   ("say that again", "repeat that", "what did you say", "come again")),
        IntentSpec("volume.up", "louder",
                   ("louder", "turn it up", "volume up", "a bit louder")),
        IntentSpec("volume.down", "quieter",
                   ("quieter", "turn it down", "volume down", "a bit quieter")),
        IntentSpec("media.pause", "pause playback", ("pause", "pause it", "pause the music")),
        IntentSpec("media.resume", "resume playback",
                   ("resume", "play", "carry on", "unpause", "keep playing")),
        IntentSpec("time.now", "tell the time",
                   ("what time is it", "what's the time", "tell me the time")),
        IntentSpec("timer.set", "start a countdown timer",
                   ("set a timer for {duration}", "timer for {duration}",
                    "start a {duration} timer", "countdown {duration}"),
                   slots=(SlotSpec("duration", "duration", TIMER_DURATIONS),)),
        IntentSpec("timer.cancel", "cancel the running timer",
                   ("cancel the timer", "stop the timer", "kill the timer"),
                   when=("timer_running",)),
    ),
)


# -- the answer ----------------------------------------------------------------

@dataclass
class Field:
    """A typed value and how sure the answer is of it."""

    value: Any
    p: float
    probs: dict[str, float] | None = None   # the full distribution, when there is one

    def as_json(self, full: bool = False) -> dict:
        out = {"value": self.value, "p": round(float(self.p), 4)}
        if full and self.probs:
            out["probs"] = {k: round(float(v), 4) for k, v in self.probs.items()}
        return out


@dataclass
class Decision:
    """One utterance, answered in the agent's schema."""

    route: str
    source: str
    addressed: Field
    intent: Field | None = None
    slots: dict[str, Field] = field(default_factory=dict)
    transcript: str | None = None
    reason: str = ""
    turn: str | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    model: str | None = None
    # The cascade's own verdict object (IntentDecision), when it ran. Not
    # serialised: the agent reads the typed fields; this is for code that
    # already speaks the old interface.
    detail: Any = None

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}")
        self.source = SOURCE_ALIASES.get(self.source, self.source)
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")

    @property
    def for_assistant(self) -> bool:
        return self.addressed.value == ASSISTANT

    def summary(self) -> str:
        """A short human-readable line: 'timer.set(duration=420)'."""
        if self.intent is None:
            return self.addressed.value
        args = ", ".join(f"{k}={v.value}" for k, v in self.slots.items())
        return f"{self.intent.value}({args})"

    def as_json(self, full: bool = False) -> dict:
        out: dict[str, Any] = {
            "format": DECISION_FORMAT,
            "route": self.route,
            "source": self.source,
            "addressed": self.addressed.as_json(full),
            "intent": self.intent.as_json(full) if self.intent is not None else None,
            "slots": {k: v.as_json(full) for k, v in self.slots.items()},
            "transcript": self.transcript,
        }
        if self.reason:
            out["reason"] = self.reason
        if self.turn:
            out["turn"] = self.turn
        if self.latency_ms:
            out["latency_ms"] = {k: round(float(v), 2) for k, v in self.latency_ms.items()}
        if self.model:
            out["model"] = self.model
        return out

    def to_json(self, full: bool = False) -> str:
        return json.dumps(self.as_json(full), allow_nan=False)


def validate_decision(data: dict, schema: Schema) -> list[str]:
    """Problems with a decision dict against a schema (empty when valid).

    For agents and tests: everything Juno emits must pass this.
    """
    problems = []
    if data.get("route") not in ROUTES:
        problems.append(f"route {data.get('route')!r} not in {ROUTES}")
    addressed = data.get("addressed") or {}
    if addressed.get("value") not in ADDRESSEES:
        problems.append(f"addressed {addressed.get('value')!r} not in {ADDRESSEES}")
    intent = data.get("intent")
    if data.get("route") == "act":
        if not intent:
            problems.append("an act decision needs an intent")
        elif schema.intent(intent.get("value")) is None:
            problems.append(f"intent {intent.get('value')!r} is not in the schema")
        else:
            spec = schema.intent(intent["value"])
            if spec.needs_transcript:
                problems.append(f"{spec.name} needs a transcript; it cannot be acted on")
            slots = data.get("slots") or {}
            for slot in spec.slots:
                if slot.required and slot.name not in slots:
                    problems.append(f"{spec.name} is missing required slot {slot.name}")
    for name, value in (data.get("slots") or {}).items():
        p = value.get("p")
        if not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
            problems.append(f"slot {name}: p must be a probability")
    for key in ("addressed", "intent"):
        node = data.get(key)
        if node and not (isinstance(node.get("p"), (int, float)) and 0.0 <= node["p"] <= 1.0):
            problems.append(f"{key}: p must be a probability")
    return problems


def describe_duration(seconds: int) -> str:
    """420 -> '7 minutes'; 90 -> '1 minute 30 seconds'; 5400 -> '1 hour 30 minutes'."""
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    for n, unit in ((hours, "hour"), (minutes, "minute"), (secs, "second")):
        if n:
            parts.append(f"{n} {unit}{'' if n == 1 else 's'}")
    return " ".join(parts) or "0 seconds"
