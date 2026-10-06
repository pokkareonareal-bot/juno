"""What to say: the scripted text behind the synthetic training clips.

Every line carries its GOLD label -- who it is for, what intent, which slot
values -- because the script was written knowing. That is what lets an
experiment compare a student trained on the teacher's labels (distillation)
with one trained on the truth, and measure the teacher's own error.

Four kinds of line, chosen to make the problem honest rather than easy:

  assistant, typed     core commands (and the agent's own intents), said the
                       many ways people say them, with and without the name
  assistant, open      questions and requests whose content matters --
                       Reflex must hand these on, not answer them
  human                remarks, plans and questions to other people,
                       including HARD NEGATIVES: lines that are word for word
                       a command ("stop it", "what time is it, Sam?",
                       "turn it down a bit, I'm on the phone"). No audio-only
                       model can be sure of these; the right answer is to be
                       unsure and escalate, and the evaluation checks that.
  background           broadcast and recorded speech: news, adverts,
                       narration, a TV voice reading out a time or a timer

Lines are generated from templates with a seeded RNG, so a corpus is
reproducible from (schema, seed, size).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from juno_core.slu.parse import render_example
from juno_core.slu.schema import (
    ASSISTANT, CORE_SCHEMA, OPEN_REQUEST, TIMER_DURATIONS, Schema, describe_duration,
)

NAME_FORMS = ("{name}, {x}", "hey {name}, {x}", "{name} {x}", "{x}, {name}", "ok {name}, {x}")
POLITE_FORMS = ("{x}", "{x}", "{x}", "please {x}", "{x} please", "can you {x}",
                "could you {x}", "can you {x} please")

CORE_PHRASES: dict[str, tuple[str, ...]] = {
    "stop": ("stop", "stop it", "stop talking", "okay stop", "that's enough", "enough",
             "shush", "be quiet", "stop that", "quiet"),
    "cancel": ("never mind", "cancel", "cancel that", "forget it", "forget that",
               "scratch that", "don't worry about it"),
    "confirm.yes": ("yes", "yeah", "yes please", "sure", "go ahead", "do it", "yep",
                    "please do", "sounds good", "yeah go ahead", "of course", "go for it"),
    "confirm.no": ("no", "no thanks", "nope", "no thank you", "don't", "leave it",
                   "nah", "don't bother", "not now"),
    "repeat": ("say that again", "repeat that", "what did you say", "come again",
               "sorry, what did you say", "can you repeat that", "one more time"),
    "volume.up": ("louder", "turn it up", "volume up", "a bit louder", "turn the volume up",
                  "increase the volume", "a little louder"),
    "volume.down": ("quieter", "turn it down", "volume down", "a bit quieter",
                    "turn the volume down", "lower the volume", "softer"),
    "media.pause": ("pause", "pause it", "pause the music", "pause that", "pause the podcast"),
    "media.resume": ("resume", "play", "carry on", "unpause", "keep playing", "continue",
                     "resume playback"),
    "time.now": ("what time is it", "what's the time", "tell me the time",
                 "what's the time right now", "what time is it now"),
    "timer.set": ("set a timer for {duration}", "timer for {duration}",
                  "start a {duration} timer", "set a {duration} timer",
                  "countdown {duration}", "put on a timer for {duration}",
                  "start a timer for {duration}", "set the timer for {duration}"),
    "timer.cancel": ("cancel the timer", "stop the timer", "kill the timer",
                     "turn off the timer", "cancel my timer"),
}

# The intents that make sense on their own, without the name or politeness
# wrapped round them ("please louder" is not something anyone says).
NO_POLITE = {"confirm.yes", "confirm.no", "stop", "cancel", "repeat"}

OPEN_REQUESTS = (
    "what's the weather like tomorrow", "will it rain this afternoon",
    "how far away is the moon", "what's the capital of {country}",
    "who wrote {book}", "how do I make {food}", "how long do I boil an egg",
    "remind me to {chore} at {clock}", "add {item} to my shopping list",
    "send a message to {person} saying I'm running late",
    "call {person}", "play some {genre}", "play {genre} in the kitchen",
    "what's {a} times {b}", "convert {a} miles to kilometres",
    "what's on my calendar today", "when is my next meeting",
    "turn on the {room} lights", "set the thermostat to {temp} degrees",
    "what does {word} mean", "how do you spell {word}",
    "tell me a joke", "what's the news today", "how tall is {landmark}",
    "who won the match last night", "find me a recipe for {food}",
    "what year did {event} happen", "read me my messages",
    "how many grams in a pound", "translate thank you into {language}",
    "what's a good name for a cat", "open {site}", "navigate to {place}",
    "how long will it take to get to {place}", "is {place} open today",
    "what should I cook tonight", "summarise my emails",
)

HUMAN_LINES = (
    "did you remember to call your mum back", "I think we should leave at six",
    "can you pass the salt", "what do you want for dinner",
    "she said she'd be here by eight", "we need to buy more {item}",
    "how was work today", "I'm not sure, maybe next week",
    "did you see that film last night", "let's go to {place} this weekend",
    "have you fed the cat", "where did you put the keys",
    "I'll do the dishes later", "that's what I told him",
    "honestly I'm exhausted", "do you want a cup of tea",
    "are you coming to {place} on Saturday", "he told me to stop worrying",
    "I set a timer for the pasta already", "my phone's almost dead",
    "we should probably get going", "what did the doctor say",
    "can you grab the door", "it's freezing in here",
    "I can't find my glasses anywhere", "is it your turn to cook",
    "I'm going to take a shower", "the kids are asleep finally",
    "we're out of {item} again", "did {person} call you back",
)

# Word for word, things the assistant could be asked -- said to a person.
HARD_NEGATIVES = (
    "what time is it, {person}", "{person}, stop it", "stop it, {person}",
    "turn it down a bit, I'm on the phone", "can you turn it up, I can't hear the TV",
    "{person}, what's the time", "yes, I'll be there", "no, not that one",
    "never mind, I found it", "say that again, {person}", "{person}, pause the game",
    "set a timer for the oven, {person}", "{person}, cancel the order",
    "sure, go ahead", "come again, {person}", "nope, not today",
    "what's the weather like where you are", "{person}, play something good",
    "louder, {person}, I can't hear you", "okay stop, that tickles",
)

BACKGROUND_LINES = (
    "and in other news tonight, the government has announced new measures",
    "the time is now seven o'clock, here's the news", "call now for a free quote",
    "stay tuned after the break", "temperatures will reach twenty degrees by the afternoon",
    "chapter three. the house stood at the end of a long road",
    "and that's the end of the first half here at the stadium",
    "this programme contains scenes some viewers may find upsetting",
    "terms and conditions apply, see website for details",
    "back to you in the studio", "you're listening to the morning show",
    "set a timer for twenty minutes and let the dough rise", "pause the video if you need to",
    "it was a dark and stormy night", "traffic is heavy on the motorway this morning",
    "subscribe and hit the bell for more videos", "the next train to {place} is delayed",
    "welcome back to the podcast, today we're talking about sleep",
    "the recipe calls for two cups of flour", "previously on the show",
)

FILL = {
    "country": ("France", "Mongolia", "Peru", "Canada", "Kenya", "Japan", "Norway", "Chile"),
    "book": ("Pride and Prejudice", "Moby Dick", "The Hobbit", "Dracula", "Frankenstein"),
    "food": ("pancakes", "egg curry", "risotto", "banana bread", "a margarita pizza", "dal"),
    "chore": ("take the bins out", "call the dentist", "water the plants", "pay the rent",
              "pick up the kids"),
    "clock": ("five", "six thirty", "nine tomorrow", "noon", "half past four"),
    "item": ("milk", "eggs", "bread", "coffee", "toilet roll", "bananas", "rice"),
    "person": ("Sam", "Alex", "Mum", "Priya", "Tom", "Maya", "Dad", "Chris", "Jo"),
    "genre": ("jazz", "classical music", "nineties hip hop", "lo-fi beats", "the Beatles"),
    "a": ("seven", "twelve", "thirty", "fifteen", "eight"),
    "b": ("eight", "nine", "four", "eleven", "six"),
    "room": ("kitchen", "bedroom", "living room", "hallway"),
    "temp": ("nineteen", "twenty", "twenty one", "eighteen"),
    "word": ("serendipity", "ubiquitous", "ephemeral", "quixotic", "laconic"),
    "landmark": ("the Eiffel Tower", "Everest", "the Shard", "the Empire State Building"),
    "event": ("the moon landing", "the fall of the Berlin wall", "the first world war"),
    "language": ("Spanish", "Japanese", "French", "German"),
    "site": ("the BBC website", "YouTube", "my email", "Wikipedia"),
    "place": ("the station", "Brighton", "the supermarket", "Edinburgh", "the park"),
}

# Durations on the typed list, and some off it -- a timer for 13 minutes is
# still timer.set, just with a value Reflex must not claim to know.
OFF_LIST_DURATIONS = (45, 150, 200, 660, 780, 1080, 1320, 3000, 4200, 10800)


@dataclass
class Line:
    text: str
    addressed: str
    intent: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    category: str = ""

    def as_row(self) -> dict:
        return {"text": self.text, "g_addressed": self.addressed, "g_intent": self.intent,
                "g_slots": dict(self.slots), "category": self.category}


def _fill(template: str, rng: random.Random) -> str:
    out = template
    for key, options in FILL.items():
        while "{" + key + "}" in out:
            out = out.replace("{" + key + "}", rng.choice(options), 1)
    return out


def _wrap(phrase: str, intent: str, rng: random.Random, name: str) -> str:
    text = phrase
    if intent not in NO_POLITE and rng.random() < 0.35:
        text = rng.choice(POLITE_FORMS).format(x=text)
    if rng.random() < 0.45:
        text = rng.choice(NAME_FORMS).format(name=name, x=text)
    return text


def build_corpus(schema: Schema = CORE_SCHEMA, *, size: int = 2000, seed: int = 0,
                 name: str = "Juno", mix: dict[str, float] | None = None) -> list[Line]:
    """``size`` scripted lines, mixed by category (fractions sum to 1)."""
    rng = random.Random(seed)
    mix = mix or {"typed": 0.40, "open": 0.18, "human": 0.20, "hard_negative": 0.10,
                  "background": 0.12}
    counts = {k: int(round(v * size)) for k, v in mix.items()}
    lines: list[Line] = []

    typed = [i for i in schema.intents if i.name != OPEN_REQUEST and not i.needs_transcript]
    for n in range(counts.get("typed", 0)):
        spec = typed[n % len(typed)]
        phrases = CORE_PHRASES.get(spec.name) or spec.examples or (spec.name.replace(".", " "),)
        template = rng.choice(phrases)
        slots: dict[str, Any] = {}
        for slot in spec.slots:
            if slot.type == "duration":
                pool = TIMER_DURATIONS if rng.random() < 0.85 else OFF_LIST_DURATIONS
                slots[slot.name] = rng.choice(pool)
            else:
                slots[slot.name] = rng.choice(slot.values)
        text = render_example(template, slots, {s.name: s.type for s in spec.slots})
        if "duration" in slots:
            text = text.replace(describe_duration(slots["duration"]),
                                spoken_duration(slots["duration"], rng), 1)
            text = (text.replace("minutes timer", "minute timer")
                        .replace("seconds timer", "second timer")
                        .replace("hours timer", "hour timer"))
        lines.append(Line(_wrap(text, spec.name, rng, name), ASSISTANT, spec.name, slots, "typed"))

    for _ in range(counts.get("open", 0)):
        text = _fill(rng.choice(OPEN_REQUESTS), rng)
        if rng.random() < 0.4:
            text = rng.choice(NAME_FORMS).format(name=name, x=text)
        lines.append(Line(text, ASSISTANT, OPEN_REQUEST, {}, "open"))

    for _ in range(counts.get("human", 0)):
        lines.append(Line(_fill(rng.choice(HUMAN_LINES), rng), "human_directed", None, {}, "human"))
    for _ in range(counts.get("hard_negative", 0)):
        lines.append(Line(_fill(rng.choice(HARD_NEGATIVES), rng), "human_directed", None, {},
                          "hard_negative"))
    for _ in range(counts.get("background", 0)):
        lines.append(Line(_fill(rng.choice(BACKGROUND_LINES), rng), "background_or_media",
                          None, {}, "background"))
    rng.shuffle(lines)
    return lines


_SPOKEN = {
    30: ("thirty seconds", "half a minute", "30 seconds"),
    90: ("a minute and a half", "ninety seconds", "one and a half minutes"),
    150: ("two and a half minutes", "two minutes thirty seconds"),
    1800: ("half an hour", "thirty minutes", "30 minutes"),
    2700: ("forty five minutes", "three quarters of an hour"),
    3600: ("an hour", "one hour", "sixty minutes"),
    5400: ("an hour and a half", "ninety minutes", "one and a half hours"),
    7200: ("two hours", "2 hours"),
}


def spoken_duration(seconds: int, rng: random.Random) -> str:
    """One of the ways people say this length: '7 minutes', 'half an hour'."""
    options = _SPOKEN.get(int(seconds))
    if options and rng.random() < 0.7:
        return rng.choice(options)
    return describe_duration(int(seconds))
