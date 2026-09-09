"""Lightweight conversation state.

Two things are tracked, for two different consumers:

* an utterance log (everything heard, whether or not it was aimed at the
  assistant) -- the intent engine reads this for temporal and conversational
  cues, and it is what makes a follow-up like "and in Celsius?" resolvable;
* a turn history (user/assistant pairs the assistant actually participated in)
  -- this is what gets rendered into the LLM prompt.

Both are bounded by age and by count. Nothing is written to disk from here;
persistence is the observer's business and is subject to privacy.* config.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]


@dataclass
class Utterance:
    text: str
    timestamp: float
    reliable: bool = True
    ai_directed: bool | None = None
    confidence: float | None = None
    duration: float = 0.0

    @property
    def age(self) -> float:
        return time.monotonic() - self.timestamp


@dataclass
class Turn:
    role: Role
    content: str
    timestamp: float = field(default_factory=time.monotonic)
    tool_name: str | None = None
    tool_call_id: str | None = None


class ConversationContext:
    def __init__(self, history_ttl: float = 900.0, max_utterances: int = 40, max_turns: int = 40):
        self.history_ttl = history_ttl
        self._utterances: deque[Utterance] = deque(maxlen=max_utterances)
        self._turns: deque[Turn] = deque(maxlen=max_turns)
        self._last_ai_response_at: float | None = None
        self._last_ai_text: str = ""
        self._last_tool_used: str | None = None

    # -- writes -----------------------------------------------------------

    def add_utterance(self, utterance: Utterance) -> None:
        self._utterances.append(utterance)
        self.prune()

    def add_user_turn(self, text: str) -> None:
        self._turns.append(Turn("user", text))

    def add_assistant_turn(self, text: str) -> None:
        now = time.monotonic()
        self._turns.append(Turn("assistant", text, timestamp=now))
        self._last_ai_response_at = now
        self._last_ai_text = text

    def note_tool_used(self, name: str) -> None:
        """Record that a tool ran, without keeping its output.

        Tool results belong to the turn that requested them. Replaying them
        into later turns costs tokens, invites the model to answer from a
        stale lookup, and -- for APIs that model tool use structurally --
        produces an incoherent transcript: a result with no preceding call.
        The agent keeps them in its own working message list for the length
        of one turn, which is exactly as long as they are true.
        """
        self._last_tool_used = name

    @property
    def last_tool_used(self) -> str | None:
        return self._last_tool_used

    def note_ai_spoke(self) -> None:
        """Mark the end of playback, which is what the follow-up window runs from."""
        self._last_ai_response_at = time.monotonic()

    # -- reads ------------------------------------------------------------

    def prune(self) -> None:
        cutoff = time.monotonic() - self.history_ttl
        while self._utterances and self._utterances[0].timestamp < cutoff:
            self._utterances.popleft()
        while self._turns and self._turns[0].timestamp < cutoff:
            self._turns.popleft()

    @property
    def utterances(self) -> list[Utterance]:
        return list(self._utterances)

    def recent_utterances(self, n: int = 3) -> list[Utterance]:
        return list(self._utterances)[-n:]

    def overheard(self, within: float = 180.0, limit: int = 4,
                  max_chars: int = 90) -> list[str]:
        """Recent speech that was heard but not answered.

        The point of keeping it: "we should make egg curry" and, a minute
        later, "Juno, how do I make one?" are one thought, and the second is
        unanswerable without the first. Whisper already transcribed both --
        historically about three quarters of everything it transcribed was
        scored and thrown away -- so this is a matter of keeping what has
        already been paid for rather than listening harder.

        Only utterances the engine did not act on are returned; the ones it
        answered are already in the turn history as themselves. Newest last,
        and each is clipped, because this goes in front of every prompt and
        the model reads it as background, not as instructions.
        """
        cutoff = time.monotonic() - within
        out: list[str] = []
        for utterance in self._utterances:
            if utterance.ai_directed:
                continue
            if utterance.timestamp < cutoff or not utterance.reliable:
                continue
            text = " ".join(utterance.text.split())
            if len(text.split()) < 2:
                continue
            out.append(text[:max_chars])
        return out[-limit:]

    def seconds_since_ai_response(self, at: float | None = None) -> float:
        """Seconds from the assistant's last words to ``at`` (default: now).

        Callers pass the moment the user started speaking, so a long utterance
        is not treated as a late one.
        """
        if self._last_ai_response_at is None:
            return float("inf")
        return max(0.0, (at if at is not None else time.monotonic()) - self._last_ai_response_at)

    @property
    def last_ai_text(self) -> str:
        return self._last_ai_text

    def last_user_text(self) -> str:
        """The most recent thing the user said, before whatever is being
        handled now. "Think harder about that" has to name what "that" was,
        and the current utterance is not added to the record until the agent
        starts work on it."""
        for turn in reversed(self._turns):
            if turn.role == "user" and turn.content.strip():
                return turn.content
        return ""

    def ai_awaiting_answer(self) -> bool:
        """Did the assistant's last utterance end in a question?

        If so, the very next thing said is far more likely to be aimed at it.
        """
        return self._last_ai_text.strip().endswith("?")

    def messages(self, turns: int = 6) -> list[dict[str, Any]]:
        """Render recent turns in the chat format the model layer expects."""
        self.prune()
        return [
            {"role": turn.role, "content": turn.content}
            for turn in list(self._turns)[-turns * 2 :]
            if turn.role != "tool"
        ]

    def clear(self) -> None:
        self._utterances.clear()
        self._turns.clear()
        self._last_ai_response_at = None
        self._last_ai_text = ""
        self._last_tool_used = None
