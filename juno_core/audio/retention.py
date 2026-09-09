"""Keeping what the microphone heard, on this machine, so a gate can be taught.

OFF UNLESS SOMEBODY SAID YES. Segment audio is speech from whoever was in the
room, most of whom never agreed to be recorded. It stays in one directory on
this disk, it is never copied anywhere by anything in this repository, and the
whole directory can be deleted to stop. Every run rewrites a README saying so.

WHY IT EXISTS
-------------
The pre-transcription gate scores a segment from acoustics and conversation.
Its rules were written against six days of event logs, which carry duration,
VAD confidence and timing but no audio -- so anything that needs the sound
itself (pitch, rate, spectral tilt, a learned model) has nothing to learn
from. This keeps the segment and, once the intent engine has decided, the
label: what it was, what was said, and whether the assistant answered. That
pair is a training example. A few hundred of them is a dataset.

WHAT IS KEPT PER SEGMENT
------------------------
    data/segments/YYYYMMDD/<turn>.wav     16-bit mono, at the capture rate
    data/segments/YYYYMMDD/<turn>.json    what was known before transcription,
                                          then what the engine decided after

The JSON is written twice: first with the acoustics and the conversational
snapshot, then updated with the verdict. A segment that never reaches the
engine -- dropped, skipped -- is labelled with why, which is the more useful
half of the dataset.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import wave
from pathlib import Path

import numpy as np

README = """Audio segments the assistant heard, kept for teaching its gate.

Kept on this machine only. Nothing in the repository sends, copies or uploads
anything in this directory. Delete the directory to stop; it is recreated
empty if retention is still switched on in config/config.yaml
(audio.retain_segments).

Each .wav is one utterance as the microphone heard it, including anyone else
who was in the room. The .json beside it records what the assistant knew
before it transcribed the utterance and what it decided afterwards.
"""


class SegmentStore:
    def __init__(self, root, enabled: bool = False, max_mb: float = 500.0,
                 observer=None) -> None:
        self.root = Path(root)
        self.enabled = bool(enabled)
        self.max_bytes = int(float(max_mb) * 1024 * 1024)
        self._observer = observer
        self._lock = threading.Lock()
        self._paths: dict[str, Path] = {}
        self.kept = 0
        self.labelled = 0
        if self.enabled:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                (self.root / "README.txt").write_text(README, encoding="utf-8")
            except OSError:
                self.enabled = False

    def keep(self, turn_id: str, audio: np.ndarray, sample_rate: int,
             **known) -> Path | None:
        """Write the audio and what was known before transcription."""
        if not self.enabled or turn_id is None:
            return None
        day = self.root / time.strftime("%Y%m%d")
        try:
            day.mkdir(parents=True, exist_ok=True)
            path = day / f"{turn_id}.wav"
            pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(int(sample_rate))
                handle.writeframes((pcm * 32767.0).astype("<i2").tobytes())
            record = {
                "turn": turn_id,
                "kept_at": time.time(),
                "sample_rate": int(sample_rate),
                "seconds": round(len(pcm) / float(sample_rate), 3),
                **known,
            }
            # allow_nan=False so an infinity or a NaN is a loud failure here
            # rather than a bare Infinity in the file -- which Python reads
            # back and no other parser will, in what is meant to be a training
            # set somebody else's tooling will open.
            path.with_suffix(".json").write_text(
                json.dumps(record, indent=1, allow_nan=False), encoding="utf-8")
        except (OSError, ValueError) as exc:
            self._emit("retention_failed", error=type(exc).__name__)
            return None
        with self._lock:
            self._paths[turn_id] = path
            self.kept += 1
        return path

    def label(self, turn_id: str, **verdict) -> bool:
        """Add what the engine decided to a segment kept earlier."""
        if not self.enabled or turn_id is None:
            return False
        with self._lock:
            path = self._paths.pop(turn_id, None)
        if path is None:
            return False
        sidecar = path.with_suffix(".json")
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            record.update(verdict)
            record["labelled_at"] = time.time()
            sidecar.write_text(json.dumps(record, indent=1, allow_nan=False),
                               encoding="utf-8")
        except (OSError, ValueError) as exc:
            self._emit("retention_failed", error=type(exc).__name__)
            return False
        with self._lock:
            self.labelled += 1
        return True

    def prune(self) -> int:
        """Oldest days first, until the directory is under its cap."""
        if not self.enabled or not self.root.exists():
            return 0
        removed = 0
        days = sorted(p for p in self.root.iterdir() if p.is_dir())
        total = self._size()
        while total > self.max_bytes and days:
            oldest = days.pop(0)
            size = sum(f.stat().st_size for f in oldest.rglob("*") if f.is_file())
            shutil.rmtree(oldest, ignore_errors=True)
            total -= size
            removed += 1
        if removed:
            self._emit("retention_pruned", days=removed)
        return removed

    def _size(self) -> int:
        try:
            return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
        except OSError:
            return 0

    def snapshot(self) -> dict:
        return {"enabled": self.enabled, "kept": self.kept,
                "labelled": self.labelled, "root": str(self.root)}

    def _emit(self, name: str, **fields) -> None:
        if self._observer is None:
            return
        from juno_core.events import Stage

        self._observer.emit(Stage.AUDIO, name, None, **fields)
