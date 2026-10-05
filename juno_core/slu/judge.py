"""A better teacher: a language model that reads the transcript and answers in the schema.

The first teacher was the runtime's System Two run cold: the heuristic intent
engine, which is deliberately conservative without conversation context and
without its language-model second opinion. Measured against the scripted
labels it missed about half of short commands ("a little louder", "scratch
that") -- and a student distilled from it learned to miss them too.

This teacher asks a language model, for every clip, the two questions the
student is trained to answer, as multiple choice:

  who was it for?     A) Juno   B) another person   C) media / nobody
  what was wanted?    one letter per intent in the schema, plus "something else"

and reads the answer as PROBABILITIES: the model's next-token distribution
over the option letters, not a number it writes out. That is what makes it a
teacher worth distilling from -- "probably for Juno, 70%" is information a
one-hot label throws away -- and it is cheap, because nothing is generated.
The long shared instructions are run through the model once and their
key/value cache reused for every clip, so each clip costs only its own few
dozen tokens.

Local by default (Qwen3.5-4B via mlx-lm, already a few GB on disk and
offline). Any configured ``llm.provider`` works too, through ``ChatBackend``,
but an API returns text, not token probabilities, so its answers come back as
near-one-hot and are a weaker teacher; and a transcript sent to an API has
left the machine, which matters for consented sessions.

The rule parser (parse.py) still has the last word on what it can parse
exactly -- "set a timer for 7 minutes" is timer.set with 420 seconds, whatever
a model thinks -- and on slot values, which this does not guess. The model
adds what rules miss: paraphrases ("crank it up a notch") and, mostly, who
an utterance was for.
"""

from __future__ import annotations

import copy
import re
import string
import time
from dataclasses import dataclass, field

import numpy as np

from juno_core.slu.schema import ADDRESSEES, ASSISTANT, CORE_SCHEMA, OPEN_REQUEST, Schema

DEFAULT_LOCAL_MODEL = "mlx-community/Qwen3.5-4B-4bit"

SYSTEM = (
    "You label speech for training a voice assistant called {name}. {name} listens to a room "
    "all the time, with no wake word, and has to work out which speech is meant for it. "
    "People in the room also talk to each other and on the phone, and TVs, radios, videos "
    "and podcasts play. Nobody in the room is called {name}: speech that addresses {name} by "
    "name is speech to the assistant. Each utterance you see was captured on its own, with "
    "no other context, and transcribed by speech recognition, which makes mistakes -- the "
    "name in particular is often misheard, as words like {misheard}. Answer with a single "
    "letter."
)
MISHEARD = ("June", "Juneau", "Juna", "Zuno", "Zulo", "Julo", "Jumbo", "you know")

ADDRESSEE_QUESTION = (
    "Things {name} can do: tell the time, set and cancel timers, stop, pause or resume "
    "what is playing, change the volume, repeat what it said, answer questions, and "
    "handle requests like messages, reminders, music and smart-home devices.\n\n"
    "Who was this utterance most likely meant for?\n"
    "A) {name}, the voice assistant: a command, request or question for it\n"
    "B) another person, in the room or on the phone\n"
    "C) nobody: media playing (TV, radio, podcast, video), or not real speech\n\n"
)

INTENT_QUESTION = "If this was meant for {name}, which of these is it?\n"

LETTERS = string.ascii_uppercase


@dataclass
class Judgement:
    addressed: dict[str, float]
    intent: dict[str, float] = field(default_factory=dict)
    ms: float = 0.0


class LocalMLXBackend:
    """Option probabilities from a local model's next-token logits."""

    def __init__(self, repo: str = DEFAULT_LOCAL_MODEL) -> None:
        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.models.cache import make_prompt_cache
        except ImportError as exc:
            raise ImportError("the local judge needs mlx-lm on Apple Silicon: "
                              "pip install -e '.[slu]'") from exc
        self._mx = mx
        self._make_cache = make_prompt_cache
        self.repo = repo
        self.model, self.tokenizer = load(repo)
        self._prefixes: dict[str, tuple] = {}
        self.name = f"mlx:{repo}"

    def _template(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                      tokenize=False, enable_thinking=False)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                      tokenize=False)

    def choose(self, system: str, question: str, item: str, n_options: int) -> np.ndarray:
        """p over the first ``n_options`` letters, for one item after a shared question.

        Everything up to the item is a prefix whose cache is computed once per
        (system, question) and copied for each item.
        """
        mx = self._mx
        marker = "\u0000ITEM\u0000"
        full = self._template(system, question + marker) + "Answer: "
        head, tail = full.split(marker)
        key = head
        if key not in self._prefixes:
            cache = self._make_cache(self.model)
            tokens = self.tokenizer.encode(head)
            logits = self.model(mx.array(tokens)[None], cache=cache)
            mx.eval(logits)
            self._prefixes[key] = (cache, len(tokens))
        cache, _ = self._prefixes[key]
        cache = copy.deepcopy(cache)
        rest = self.tokenizer.encode(item + tail, add_special_tokens=False)
        logits = self.model(mx.array(rest)[None], cache=cache)[0, -1]
        logits = np.array(logits.astype(mx.float32))
        scores = []
        for letter in LETTERS[:n_options]:
            ids = {t for v in (letter, " " + letter) for t in self.tokenizer.encode(
                v, add_special_tokens=False)[:1]}
            scores.append(max(float(logits[i]) for i in ids))
        scores = np.asarray(scores)
        p = np.exp(scores - scores.max())
        return p / p.sum()


class ChatBackend:
    """Any juno_core.llm model: it answers in text, so the 'probabilities' are near-one-hot."""

    def __init__(self, model) -> None:
        if getattr(model, "placeholder", False):
            raise ValueError("the echo model cannot judge anything -- configure llm.provider")
        self.model = model
        self.name = f"chat:{getattr(model, 'name', 'model')}"

    def choose(self, system: str, question: str, item: str, n_options: int) -> np.ndarray:
        reply = self.model.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": question + item}],
            max_tokens=4, temperature=0.0, timeout=20.0)
        match = re.search(r"\b([A-Z])\b", reply.strip().upper()) or re.match(r"([A-Z])", reply.strip().upper())
        p = np.full(n_options, 0.1 / max(n_options - 1, 1))
        if match and LETTERS.index(match.group(1)) < n_options:
            p[LETTERS.index(match.group(1))] = 0.9
        else:
            p[:] = 1.0 / n_options
        return p / p.sum()


class LLMJudge:
    """Who an utterance was for and what was wanted, from its transcript, with probabilities."""

    def __init__(self, backend, schema: Schema = CORE_SCHEMA, assistant_name: str = "Juno",
                 intent_threshold: float = 0.2, misheard: tuple[str, ...] = ()) -> None:
        self.backend = backend
        self.schema = schema
        self.name = assistant_name
        # Below this p(for Juno), the intent question is not worth asking.
        self.intent_threshold = intent_threshold
        # The default list is for "Juno"; aliases from config add to it.
        words = list(dict.fromkeys([*misheard, *(MISHEARD if assistant_name == "Juno" else ())]))
        sounds = ", ".join(f'"{w}"' for w in words) if words else '"' + assistant_name.lower() + '"'
        self.system = SYSTEM.format(name=assistant_name, misheard=sounds)
        self.addressee_q = ADDRESSEE_QUESTION.format(name=assistant_name)
        self.intents = [i for i in schema.intents if i.name != OPEN_REQUEST] + \
            [schema.intent(OPEN_REQUEST)]
        lines = [INTENT_QUESTION.format(name=assistant_name)]
        for letter, spec in zip(LETTERS, self.intents):
            if spec.name == OPEN_REQUEST:
                lines.append(f"{letter}) something else: a question or request with its own content")
            else:
                lines.append(f"{letter}) {spec.name}: {spec.description}")
        self.intent_q = "\n".join(lines) + "\n\n"
        if len(self.intents) > len(LETTERS):
            raise ValueError("too many intents for single-letter options")
        self._memo: dict[str, Judgement] = {}

    @property
    def tag(self) -> str:
        return f"judge[{self.backend.name}]"

    def judge(self, transcript: str) -> Judgement:
        text = " ".join((transcript or "").split())
        if not text:
            return Judgement(addressed={ASSISTANT: 0.02, "human_directed": 0.18,
                                        "background_or_media": 0.80})
        key = text.lower()
        if key in self._memo:                 # templated data repeats; the answer does not change
            return self._memo[key]
        t0 = time.perf_counter()
        item = f'Utterance: "{text}"\n'
        p_who = self.backend.choose(self.system, self.addressee_q, item, 3)
        addressed = dict(zip(ADDRESSEES, map(float, p_who)))
        intent: dict[str, float] = {}
        if addressed[ASSISTANT] >= self.intent_threshold:
            p_int = self.backend.choose(self.system, self.intent_q, item, len(self.intents))
            intent = {spec.name: float(p) for spec, p in zip(self.intents, p_int)}
        out = Judgement(addressed, intent, (time.perf_counter() - t0) * 1000.0)
        self._memo[key] = out
        return out


def build_judge(spec: str, schema: Schema = CORE_SCHEMA, assistant_name: str = "Juno",
                aliases: tuple[str, ...] = ()) -> LLMJudge:
    """'mlx' / 'mlx:<repo>' for a local model, or an llm.provider name (openai, ...)."""
    if spec == "mlx" or spec.startswith("mlx:"):
        repo = spec.split(":", 1)[1] if ":" in spec else DEFAULT_LOCAL_MODEL
        return LLMJudge(LocalMLXBackend(repo), schema, assistant_name, misheard=tuple(aliases))
    from juno_core.llm import build_model

    return LLMJudge(ChatBackend(build_model({"provider": spec})), schema, assistant_name,
                    misheard=tuple(aliases))
