"""Two things people say to a voice assistant that are not questions.

"What did you say?" and "tell me more" are both about the answer that just
happened rather than about the world, and both were unanswerable: the first
because nothing kept the last answer for saying again, the second because
every answer is capped at three sentences and there was no way to ask past it.

Neither needs the language model to work out what was meant, so neither pays
for one. Repeating is free and instant -- the words already exist. Expanding
needs a new answer, but it needs it to the SAME question, so it re-asks rather
than trying to continue.
"""

from __future__ import annotations

import re

# Missing an answer is routine when the thing is worn in a room with people,
# and until now the only recovery was asking the whole question again and
# paying the full four seconds for an answer that had already been given.
_REPEAT = re.compile(
    r"\b(?:what|sorry)[,\s]*(?:did|was)\s+(?:you|that)\b"
    r"|\bsay\s+(?:that|it)\s+again\b"
    r"|\bsay\s+again\b"
    r"|\brepeat\s+(?:that|it|yourself)?\b"
    r"|\bcome\s+again\b"
    r"|\bi\s+(?:did\s*n[o']?t|missed|couldn'?t)\s+(?:catch|hear|get)\s+(?:that|you|it)\b"
    r"|\bone\s+more\s+time\b",
    re.IGNORECASE,
)

# Answers are capped at three sentences and sixty words, which is right almost
# always and wrong exactly when somebody is interested.
_EXPAND = re.compile(
    r"\btell\s+me\s+more\b"
    r"|\b(?:go|carry)\s+on\b"
    r"|\bmore\s+detail\b|\bin\s+more\s+detail\b"
    r"|\bexpand\s+on\s+(?:that|it|this)\b"
    r"|\belaborate\b"
    r"|\bwhat\s+else\b"
    r"|\bthe\s+long\s+version\b"
    r"|\bkeep\s+going\b",
    re.IGNORECASE,
)


# Words that can sit around a trigger without making it a new question.
_FILLER = re.compile(
    r"\b(?:ok|okay|right|so|well|hmm|um|uh|please|then|now|juno|yeah|and|"
    r"a|the|bit|little|just|can|you|sorry)\b|[^\w\s]",
    re.IGNORECASE,
)


def _is_only(pattern: re.Pattern, text: str) -> bool:
    """Is this utterance essentially just the trigger and nothing else?

    "go on" is a request to continue. "go on then, what time is it?" is a
    question, and treating it as a request to continue answers the previous
    one instead -- the trigger has to be the whole point of the sentence, not
    a thing said on the way to one.
    """
    match = pattern.search(text or "")
    if not match:
        return False
    rest = (text[: match.start()] + " " + text[match.end():])
    leftover = _FILLER.sub(" ", rest).split()
    return len(leftover) <= 1


def wants_repeat(text: str) -> bool:
    return _is_only(_REPEAT, text)


def wants_more_detail(text: str) -> bool:
    return _is_only(_EXPAND, text)


# Given to the model when the answer it already gave was too short. The
# question is asked again rather than continued, because "carry on from where
# you stopped" reliably produces something that only makes sense next to an
# answer the listener has half forgotten.
EXPAND_INSTRUCTION = (
    "Answer this again, at more length. The previous answer was too short. "
    "Give the detail that was left out rather than repeating what was already "
    "said, and do not refer to the earlier answer."
)
