"""Small text utilities the intent engine needs.

Just the one function, split out from a larger response-shaping module that
otherwise belongs to the product side (stripping markdown for speech, and so
on -- that's about *how you present an answer*, not about *whether the
utterance was meant for you*).
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])(?=\s)|(?<=[.!?])$")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_END.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])
