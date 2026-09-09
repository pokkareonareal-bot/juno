"""Audio buffering primitives.

``PrerollBuffer`` keeps a short rolling window of audio from *before* the VAD
fired, so utterances are not clipped at the start. Voice activity detectors
need a few frames of speech before they are confident, and without pre-roll
the first syllable is routinely lost -- which the STT then guesses at.

``SegmentAccumulator`` collects frames for the duration of one utterance.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from juno_core.audio.capture import AudioFrame


class PrerollBuffer:
    """Fixed-duration ring buffer of the most recent frames."""

    def __init__(self, sample_rate: int, seconds: float) -> None:
        self.sample_rate = sample_rate
        self.seconds = seconds
        self._frames: deque[AudioFrame] = deque()
        self._samples = 0
        self._capacity = int(sample_rate * seconds)

    def push(self, frame: AudioFrame) -> None:
        self._frames.append(frame)
        self._samples += frame.samples.size
        while self._samples > self._capacity and self._frames:
            dropped = self._frames.popleft()
            self._samples -= dropped.samples.size

    def drain(self) -> list[AudioFrame]:
        """Take everything held and reset. Called at SPEECH_START."""
        frames = list(self._frames)
        self.clear()
        return frames

    def clear(self) -> None:
        self._frames.clear()
        self._samples = 0

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def duration(self) -> float:
        return self._samples / self.sample_rate


class SegmentAccumulator:
    """Collects the frames belonging to a single utterance."""

    def __init__(self, sample_rate: int, max_seconds: float) -> None:
        self.sample_rate = sample_rate
        self._max_samples = int(sample_rate * max_seconds)
        self._frames: list[AudioFrame] = []
        self._samples = 0
        self.truncated = False

    def extend(self, frames: list[AudioFrame]) -> None:
        for frame in frames:
            self.push(frame)

    def push(self, frame: AudioFrame) -> None:
        if self._samples >= self._max_samples:
            self.truncated = True
            return
        self._frames.append(frame)
        self._samples += frame.samples.size
        # Flag it on the frame that fills the buffer, not on the next one.
        # SegmentDetector closes the segment as soon as ``is_full`` goes true,
        # so a later push never happens and the flag was never set: every
        # segment that hit the cap was reported as though it had ended
        # naturally. That made a real failure -- the VAD gluing a whole
        # conversation into one 20-second segment -- invisible in the logs.
        if self._samples >= self._max_samples:
            self.truncated = True

    def audio(self) -> np.ndarray:
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([f.samples for f in self._frames])

    @property
    def start_time(self) -> float:
        return self._frames[0].timestamp if self._frames else 0.0

    @property
    def end_time(self) -> float:
        if not self._frames:
            return 0.0
        last = self._frames[-1]
        return last.timestamp + last.samples.size / self.sample_rate

    @property
    def duration(self) -> float:
        return self._samples / self.sample_rate

    @property
    def is_full(self) -> bool:
        return self._samples >= self._max_samples

    def reset(self) -> None:
        self._frames.clear()
        self._samples = 0
        self.truncated = False
