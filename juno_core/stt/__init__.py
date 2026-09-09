"""The shape a speech-to-text engine needs to have. Nothing else.

No engine ships in this package -- that's a deliberate choice, not a missing
piece (see the README for why, and for two or three engines that plug into
this cleanly in a few lines). Implement ``STTEngine`` against whatever you
choose -- a local model, a cloud API, anything -- and the rest of the
pipeline doesn't need to know or care which.

    class MySTT(STTEngine):
        def transcribe(self, audio, sample_rate=16000) -> Transcript:
            text = ...  # however you get text out of `audio`
            return Transcript(text=text)

That's the whole contract. Everything else on ``Transcript`` is optional and
only makes the pipeline smarter if you fill it in:

  - ``reliable=False`` says "here's a transcript, but don't act on a guess" --
    the intent engine treats an unreliable transcript as evidence AGAINST
    answering rather than a reason to.
  - ``tail_reliable`` is the same idea for just the end of the utterance,
    useful when one capture can span "talking to someone else" followed by
    "...actually, hey Juno, what's the weather" -- the average confidence
    over the whole clip says nothing about the part that matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def finite(value: Any) -> float | None:
    """The value, or None if it isn't a real number.

    Some engines return NaN for degenerate input (silence, pure noise).
    Carried forward it poisons every comparison it touches; serialised, it
    produces a bare ``NaN`` token that is not valid JSON. Better to know now.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class Transcript:
    text: str
    language: str | None = None
    # Your engine's confidence proxy, if it has one, on whatever scale it
    # natively uses. Not required -- `reliable` is what the pipeline reads.
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    duration: float = 0.0
    latency: float = 0.0
    reliable: bool = True
    tail_reliable: bool = True
    tail_logprob: float | None = None
    reject_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.text.strip())


class STTEngine:
    """Base class. Implement `transcribe`; `warmup` is optional."""

    name = "base"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        raise NotImplementedError

    def warmup(self) -> None:
        """Run one throwaway inference so the first real utterance isn't slow.

        Optional. Called once, if at all, before listening starts.
        """


def build_stt(config) -> "STTEngine":
    """Construct the configured `stt.provider`.

    `config` is the `stt:` section of config.yaml. See config.example.yaml
    for the two bundled options (`faster_whisper`, `openai_whisper`) and the
    README section "Choosing a speech-to-text engine" for how to add your
    own -- it's a ~15-line class either way.
    """
    provider = str(config.get("provider", "") or "").lower()
    if not provider:
        raise ValueError(
            "stt.provider isn't set in config.yaml. See config.example.yaml "
            "and the README section 'Choosing a speech-to-text engine'."
        )
    if provider == "faster_whisper":
        from .faster_whisper import FasterWhisperSTT
        return FasterWhisperSTT(
            model=config.get("model", "small.en"),
            device=config.get("device", "cpu"),
            compute_type=config.get("compute_type", "int8"),
        )
    if provider == "openai_whisper":
        from .openai_whisper import OpenAIWhisperSTT
        return OpenAIWhisperSTT(model=config.get("model", "whisper-1"))
    raise ValueError(
        f"unknown stt.provider {provider!r} -- expected 'faster_whisper' or "
        f"'openai_whisper', or add your own under juno_core/stt/ and extend "
        f"this factory (or just construct your STTEngine subclass directly "
        f"and pass it to JunoPipeline yourself, skipping this factory)"
    )
