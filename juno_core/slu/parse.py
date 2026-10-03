"""Filling the schema from WORDS: System Two's half of the typed answer.

Once speech-to-text has run, the intent engine already says whether the
utterance was for the assistant. This says WHAT it was, in the agent's
schema: which intent, which slot values. It is also the teacher's labeller
-- every typed label the student learns from comes out of this file, so it
is written to be predictable before it is written to be clever:

  - the core intents are matched by rules over a normalised transcript,
    ANCHORED for the short commands ("stop" is a command; "I told him to
    stop" is not), unanchored only where the words cannot mean anything
    else ("what time is it", "set a timer for...");
  - an agent's own intents are matched against their declared examples,
    with ``{slot}`` placeholders as wildcards;
  - anything else is an open request. When a language model is configured
    and the agent declared intents of its own, it gets one chance to place
    the transcript in the schema, and its answer is validated against it.

Durations are parsed from words and digits alike ("seven minutes", "7 min",
"an hour and a half", "half an hour", "90 seconds").
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from juno_core.slu.schema import CORE_SCHEMA, OPEN_REQUEST, Schema

# -- normalisation -------------------------------------------------------------

_POLITE_LEAD = re.compile(
    r"^(?:(?:please|ok(?:ay)?|hey|oh|um+|uh+|so|and|right|alright|well)\s+)*"
    r"(?:(?:can|could|would|will) you\s+(?:please\s+)?|i want you to\s+|"
    r"i'?d like you to\s+)?"
)
_POLITE_TAIL = re.compile(r"\s+(?:please|thanks|thank you|for me)$")


def normalise(text: str, assistant_name: str = "juno",
              aliases: tuple[str, ...] = ()) -> str:
    """Lower case, no punctuation, no name, no politeness.

    "Juno, could you please set a timer for 7 minutes?" ->
    "set a timer for 7 minutes"
    """
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    names = [n.lower() for n in (assistant_name, *aliases) if n]
    if names:
        name_alt = "|".join(re.escape(n) for n in names)
        t = re.sub(rf"\b(?:hey|hi|ok(?:ay)?|yo)?\s*(?:{name_alt})\b", " ", t)
    t = " ".join(t.split())
    for _ in range(3):
        stripped = _POLITE_LEAD.sub("", t).strip()
        stripped = _POLITE_TAIL.sub("", stripped).strip()
        if stripped == t:
            break
        t = stripped
    return t


# -- numbers and durations -----------------------------------------------------

_UNITS = {
    "zero": 0, "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SECONDS_PER = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}

_NUMBER_WORD = (r"(?:\d+(?:\.\d+)?|(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
                r"(?:[ -](?:one|two|three|four|five|six|seven|eight|nine))?|"
                r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|an?)")
_UNIT_WORD = r"(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)"
_AMOUNT = re.compile(
    rf"\b(?P<n>{_NUMBER_WORD})(?P<half>\s+and\s+a\s+half)?\s+(?P<unit>{_UNIT_WORD})"
    rf"(?P<half2>\s+and\s+a\s+half)?\b")
_HALF_UNIT = re.compile(r"(?<!and a )\bhalf\s+(?:an?\s+)?(?P<unit>hour|minute)\b")
_QUARTER_HOUR = re.compile(r"\b(?:a\s+)?quarter\s+(?:of\s+)?an?\s+hour\b")
_THREE_QUARTERS = re.compile(r"\bthree\s+quarters\s+of\s+an\s+hour\b")


def parse_number(word: str) -> float | None:
    word = word.strip().replace("-", " ")
    if re.fullmatch(r"\d+(?:\.\d+)?", word):
        return float(word)
    parts = word.split()
    if len(parts) == 1:
        if parts[0] in _UNITS:
            return float(_UNITS[parts[0]])
        if parts[0] in _TENS:
            return float(_TENS[parts[0]])
        return None
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _UNITS:
        return float(_TENS[parts[0]] + _UNITS[parts[1]])
    return None


def parse_duration(text: str) -> int | None:
    """Seconds named in ``text``, or None if no duration is named.

    "7 minutes" -> 420, "an hour and a half" -> 5400, "half an hour" -> 1800,
    "1 hour 30 minutes" -> 5400, "two and a half minutes" -> 150.
    """
    t = text.lower().replace("-", " ")
    total = 0.0
    found = False
    # "half an hour" first, and out of the way: read left to right it is
    # "an hour", which is twice as long.
    for pattern, seconds in ((_THREE_QUARTERS, 2700), (_HALF_UNIT, None), (_QUARTER_HOUR, 900)):
        for match in pattern.finditer(t):
            total += seconds if seconds else 0.5 * _SECONDS_PER[match.group("unit")]
            found = True
        t = pattern.sub(" ", t)
    for match in _AMOUNT.finditer(t):
        n = parse_number(match.group("n"))
        if n is None:
            continue
        unit = match.group("unit").rstrip("s")
        per = _SECONDS_PER.get(unit, _SECONDS_PER.get(unit[:3], 0))
        if match.group("half") or match.group("half2"):
            n += 0.5
        total += n * per
        found = True
    return int(round(total)) if found and total > 0 else None


# -- the core rules ------------------------------------------------------------

def _anchored(*phrases: str) -> re.Pattern:
    return re.compile(r"^(?:" + "|".join(phrases) + r")$")


_RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("timer.cancel", re.compile(r"\b(?:cancel|stop|kill|delete|turn off|end)\s+(?:the\s+|my\s+|that\s+)?timer\b")),
    ("timer.set", re.compile(r"\b(?:set|start|make|put on)\s+(?:a|an|the|me a)?\s*.*\btimer\b|"
                             r"\btimer\s+(?:for|of)\b|\bcount\s*down\b|\bcountdown\b")),
    ("time.now", re.compile(r"\bwhat(?:'s| is)\s+the\s+time\b|\bwhat\s+time\s+is\s+it\b|"
                            r"\btell\s+me\s+the\s+time\b|\bgot\s+the\s+time\b")),
    ("stop", _anchored(r"stop", r"stop it", r"stop talking", r"stop that", r"okay stop",
                       r"that'?s enough", r"enough", r"shush", r"shh+", r"be quiet",
                       r"quiet", r"silence", r"shut up", r"stop stop")),
    ("cancel", _anchored(r"never ?mind", r"cancel", r"cancel that", r"cancel it",
                         r"forget it", r"forget that", r"scratch that", r"don'?t worry about it")),
    ("repeat", re.compile(r"^(?:sorry\s+)?(?:say (?:that|it) again|repeat (?:that|it)|"
                          r"what did you say|come again|pardon|sorry what|"
                          r"can you repeat (?:that|it)|one more time)$")),
    ("volume.up", re.compile(r"^(?:a (?:bit|little) )?louder$|\bturn (?:it|the volume|that) up\b|"
                             r"\bvolume up\b|\b(?:increase|raise) the volume\b|^louder please$")),
    ("volume.down", re.compile(r"^(?:a (?:bit|little) )?(?:quieter|softer)$|"
                               r"\bturn (?:it|the volume|that) down\b|\bvolume down\b|"
                               r"\b(?:decrease|lower|reduce) the volume\b")),
    ("media.pause", _anchored(r"pause", r"pause it", r"pause that", r"pause the music",
                              r"pause playback", r"pause the podcast", r"hold on pause")),
    ("media.resume", _anchored(r"resume", r"play", r"carry on", r"unpause", r"keep playing",
                               r"continue", r"continue playing", r"resume playback",
                               r"play it again", r"go on")),
    ("confirm.yes", _anchored(r"yes", r"yeah", r"yep", r"yup", r"yes please", r"sure",
                              r"go ahead", r"do it", r"please do", r"do", r"ok(?:ay)?", r"sounds good",
                              r"yeah go ahead", r"yes do it", r"absolutely", r"of course",
                              r"yeah sure", r"go for it")),
    ("confirm.no", _anchored(r"no", r"nope", r"nah", r"no thanks", r"no thank you",
                             r"don'?t", r"leave it", r"no leave it", r"don'?t bother",
                             r"no don'?t", r"not now")),
)


@dataclass
class TextParse:
    """What the words say, in schema terms."""

    intent: str
    p: float
    slots: dict[str, Any] = field(default_factory=dict)
    method: str = "rule"        # rule | example | llm | none
    normalised: str = ""


class TextParser:
    """Rules for the core schema, examples for the agent's, then (maybe) a model."""

    def __init__(self, schema: Schema = CORE_SCHEMA, assistant_name: str = "Juno",
                 model=None, timeout: float = 3.0, aliases: tuple[str, ...] = ()) -> None:
        self.schema = schema
        self.assistant_name = assistant_name
        self.aliases = tuple(aliases)
        self.model = model if model is not None and not getattr(model, "placeholder", False) else None
        self.timeout = timeout
        core = set(CORE_SCHEMA.intent_names)
        self._custom = [i for i in schema.intents
                        if i.name not in core and i.name != OPEN_REQUEST]
        self._example_patterns = [
            (spec.name, _example_regex(example, spec))
            for spec in self._custom for example in spec.examples
        ]

    def parse(self, text: str) -> TextParse:
        norm = normalise(text, self.assistant_name, self.aliases)
        if not norm:
            return TextParse(OPEN_REQUEST, 0.5, method="none", normalised=norm)

        # The agent's own examples first: an agent that declares "play" as
        # something else means its own thing.
        for name, pattern in self._example_patterns:
            match = pattern.fullmatch(norm)
            if match:
                slots = self._slots_from_match(name, match)
                return TextParse(name, 0.95, slots, method="example", normalised=norm)

        for name, pattern in _RULES:
            if self.schema.intent(name) is None or not pattern.search(norm):
                continue
            slots: dict[str, Any] = {}
            if name == "timer.set":
                seconds = parse_duration(norm)
                if seconds is None:
                    # "set a timer" with no length: the agent has to ask how
                    # long, which is a conversation, not a typed command.
                    return TextParse(OPEN_REQUEST, 0.8, method="rule", normalised=norm)
                slots["duration"] = seconds
            return TextParse(name, 1.0, slots, method="rule", normalised=norm)

        if self.model is not None and self._custom:
            via_model = self._ask_model(text)
            if via_model is not None:
                via_model.normalised = norm
                return via_model
        return TextParse(OPEN_REQUEST, 0.9, method="none", normalised=norm)

    def _slots_from_match(self, intent: str, match: re.Match) -> dict[str, Any]:
        spec = self.schema.intent(intent)
        out: dict[str, Any] = {}
        for slot in spec.slots if spec else ():
            raw = match.groupdict().get(slot.name)
            if raw is None:
                continue
            out[slot.name] = coerce_slot(slot, raw)
        return out

    def _ask_model(self, text: str) -> TextParse | None:
        """One schema-filling call. Validated; any doubt returns None."""
        intents = [{"name": i.name, "description": i.description,
                    "slots": [{"name": s.name, "type": s.type, "values": list(s.values)}
                              for s in i.slots]}
                   for i in self.schema.intents]
        messages = [
            {"role": "system", "content": (
                "You map one spoken request onto a fixed list of intents. Reply "
                "with JSON only: {\"intent\": NAME, \"slots\": {...}}. Use "
                f"\"{OPEN_REQUEST}\" when none fits. Slot values must come from "
                "the listed values.")},
            {"role": "user", "content": json.dumps({"intents": intents, "utterance": text})},
        ]
        try:
            reply = self.model.complete(messages, max_tokens=120, temperature=0.0,
                                        timeout=self.timeout)
            data = json.loads(reply[reply.index("{"): reply.rindex("}") + 1])
        except Exception:
            return None
        spec = self.schema.intent(str(data.get("intent", "")))
        if spec is None:
            return None
        slots = {}
        for slot in spec.slots:
            value = (data.get("slots") or {}).get(slot.name)
            if value is None:
                continue
            value = coerce_slot(slot, value)
            if value is not None:
                slots[slot.name] = value
        return TextParse(spec.name, 0.8, slots, method="llm")


def coerce_slot(slot, raw) -> Any:
    """A raw slot string (or value) as the slot's type, or None."""
    if slot.type == "duration":
        if isinstance(raw, (int, float)):
            return int(raw)
        return parse_duration(str(raw))
    if slot.type == "number":
        if isinstance(raw, (int, float)):
            return int(raw)
        n = parse_number(str(raw))
        return int(n) if n is not None else None
    text = str(raw).strip().lower()
    for value in slot.values:
        if str(value).lower() == text:
            return value
    return None


def _example_regex(example: str, spec) -> re.Pattern:
    """'set the lights to {level}' -> a regex with a named group per slot."""
    norm = normalise(example, "")
    pieces = re.split(r"(\{[a-z_]+\})", example.lower())
    out = []
    for piece in pieces:
        m = re.fullmatch(r"\{([a-z_]+)\}", piece)
        if m and spec.slot(m.group(1)) is not None:
            out.append(rf"(?P<{m.group(1)}>.+?)")
        else:
            words = normalise(piece, "") if piece.strip() else ""
            if words:
                out.append(re.escape(words))
    pattern = r"\s*".join(out) if out else re.escape(norm)
    return re.compile(pattern)


def render_example(example: str, values: dict[str, Any], slot_types: dict[str, str]) -> str:
    """'set a timer for {duration}' + {duration: 420} -> 'set a timer for 7 minutes'."""
    from juno_core.slu.schema import describe_duration

    def fill(match: re.Match) -> str:
        name = match.group(1)
        value = values.get(name, "")
        if slot_types.get(name) == "duration":
            return describe_duration(int(value))
        return str(value)

    return re.sub(r"\{([a-z_]+)\}", fill, example)
