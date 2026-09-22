"""What's happening, in a sentence a human can read.

Every stage of the pipeline reports through an Observer instead of printing
ad hoc. This one does two things with what it's told: writes a machine-
readable line to a log file (one JSON object per event, so a session can be
replayed or measured later), and prints a short, plain-language line to the
terminal for the events a person actually cares about while it's running.

Nothing here is required. Pass ``observer=None`` anywhere in this package and
that call site just skips reporting -- these are all optional collaborators,
never load-bearing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from .events import Event, Stage

# Friendly, one-line phrasing for the events worth a person's attention.
# Anything not listed here either stays silent (routine/internal) or falls
# back to a generic "stage: event" line in verbose mode.
_FRIENDLY = {
    "speech_start": "listening...",
    "gate_scored_skip": "(quiet moment, not transcribing -- saves the wait)",
    "transcribe_started": "transcribing...",
    "transcript_ready": "heard: {text!r}",
    "transcript_rejected": "couldn't make that out clearly, ignoring it",
    "intent_accepted": "-> that was meant for me ({confidence:.0%} confident)",
    "intent_ignored": "-> not talking to me, staying quiet",
    "llm_started": "thinking...",
    "llm_finished": "done",
    "error": "something went wrong: {detail}",
    "backend_fallback": "(voice detection model unavailable -- using the simpler "
                        "energy detector instead: {reason})",
}


class Observer:
    """Collects events, prints the human-relevant ones, logs the rest."""

    def __init__(
        self,
        log_path: str | Path | None = None,
        *,
        console: bool = True,
        verbose: bool = False,
        stream: TextIO = sys.stdout,
    ) -> None:
        self.console = console
        self.verbose = verbose
        self._stream = stream
        self._log: TextIO | None = None
        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log = path.open("a", encoding="utf-8")

    def emit(self, stage: Stage, name: str, turn_id: str | None = None,
              **fields: Any) -> None:
        now = time.time()
        event = Event(stage=stage, name=name, turn_id=turn_id,
                       wall_time=now, mono_time=time.monotonic(), fields=fields)
        if self._log is not None:
            self._log.write(json.dumps(event.to_json(), default=str) + "\n")
            self._log.flush()
        if self.console:
            self._print(name, fields)

    def _print(self, name: str, fields: dict) -> None:
        key = name
        if name == "gate_scored" and fields.get("skip"):
            key = "gate_scored_skip"
        elif name == "gate_scored":
            return  # scored but not skipped: nothing a person needs to see
        template = _FRIENDLY.get(key)
        if template is not None:
            try:
                line = template.format(**fields)
            except (KeyError, ValueError):
                line = template
            print(line, file=self._stream, flush=True)
        elif self.verbose:
            extra = " ".join(f"{k}={v}" for k, v in fields.items())
            print(f"[{name}] {extra}".strip(), file=self._stream, flush=True)

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None


class SilentObserver(Observer):
    """An Observer that logs nothing and prints nothing. For tests, mostly."""

    def __init__(self) -> None:
        super().__init__(log_path=None, console=False, verbose=False)
