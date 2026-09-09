"""Voice activity detection.

VAD answers exactly one question: *is someone speaking right now?* It does not
and must not try to decide whether that speech is aimed at the assistant --
that is the intent engine's job (design sections 6 and 9).

Two backends:

* ``SileroVAD``   -- the 2 MB Silero ONNX model. Accurate, ~0.1 ms per frame.
* ``EnergyVAD``   -- adaptive noise-floor gate. No download, no dependency
                     beyond numpy. Noticeably worse in noise; it exists so the
                     pipeline still runs before assets are fetched.

``SegmentDetector`` turns per-frame probabilities into discrete utterances
using onset and hangover thresholds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from juno_core.audio.buffering import PrerollBuffer, SegmentAccumulator
from juno_core.audio.capture import AudioFrame

SILERO_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
DEFAULT_SILERO_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "models" / "silero_vad.onnx"
)


@dataclass
class SpeechSegment:
    """One candidate utterance, ready for transcription."""

    audio: np.ndarray
    start_time: float
    end_time: float
    confidence: float
    truncated: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class VADBackend:
    """Frame-in, speech-probability-out."""

    name = "base"

    def probability(self, frame: np.ndarray) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear internal state. Called between utterances and after unmuting."""


class EnergyVAD(VADBackend):
    """Adaptive-threshold RMS gate.

    Tracks the noise floor with a slow percentile estimate and calls speech
    when the frame sits far enough above it. Crude, but it does adapt to a
    room instead of relying on a fixed magic number.
    """

    name = "energy"

    def __init__(self, sensitivity: float = 3.0, floor_window: int = 100) -> None:
        self.sensitivity = sensitivity
        self._floor: deque[float] = deque(maxlen=floor_window)
        self._noise = 1e-3

    def probability(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)) + 1e-9)
        self._floor.append(rms)
        if len(self._floor) >= 10:
            # 20th percentile of recent frames approximates "the room, quiet".
            self._noise = max(float(np.percentile(self._floor, 20)), 1e-5)
        ratio = rms / (self._noise * self.sensitivity)
        # Squash to (0, 1) with a soft knee around ratio == 1.
        return float(np.clip((np.log10(ratio + 1e-9) + 0.5) / 1.2, 0.0, 1.0))

    def reset(self) -> None:
        self._floor.clear()
        self._noise = 1e-3


class SileroVAD(VADBackend):
    """Silero VAD v4/v5 through onnxruntime.

    Input signature differs between releases (v4 uses separate h/c tensors,
    v5 a single packed state), so the graph is introspected rather than
    assumed.
    """

    name = "silero"
    REQUIRED_SAMPLES = {16000: 512, 8000: 256}
    # v5 expects each window to be prefixed with the tail of the previous one.
    # Without this the model returns ~0.001 on clear speech -- it fails silent
    # rather than loud, so the whole pipeline would simply never trigger.
    CONTEXT_SAMPLES = {16000: 64, 8000: 32}

    def __init__(self, model_path: str | Path = DEFAULT_SILERO_PATH, sample_rate: int = 16000):
        import onnxruntime as ort

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD model not found at {model_path}. "
                f"Run scripts/fetch_assets.py, or set vad.backend: energy."
            )
        if sample_rate not in self.REQUIRED_SAMPLES:
            raise ValueError(f"Silero supports 8k/16k, not {sample_rate}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.sample_rate = sample_rate
        self.window = self.REQUIRED_SAMPLES[sample_rate]
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._v5 = "state" in self._input_names
        self._context_size = self.CONTEXT_SAMPLES[sample_rate] if self._v5 else 0
        self._pending = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        if self._v5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._context = np.zeros(self._context_size, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def _infer(self, window: np.ndarray) -> float:
        if self._context_size:
            model_input = np.concatenate([self._context, window])
            self._context = window[-self._context_size :].copy()
        else:
            model_input = window
        feeds = {"input": model_input.reshape(1, -1).astype(np.float32)}
        if "sr" in self._input_names:
            feeds["sr"] = np.array(self.sample_rate, dtype=np.int64)
        if self._v5:
            feeds["state"] = self._state
            out, self._state = self._session.run(None, feeds)
        else:
            feeds["h"] = self._h
            feeds["c"] = self._c
            out, self._h, self._c = self._session.run(None, feeds)
        return float(np.asarray(out).reshape(-1)[0])

    def probability(self, frame: np.ndarray) -> float:
        """Accepts any frame length; buffers to the model's exact window size."""
        self._pending = np.concatenate([self._pending, frame.astype(np.float32)])
        prob = 0.0
        seen = False
        while self._pending.size >= self.window:
            window, self._pending = (
                self._pending[: self.window],
                self._pending[self.window :],
            )
            # If a caller feeds oversized frames, the loudest sub-window wins:
            # missing speech onset is worse than a marginally early trigger.
            prob = max(prob, self._infer(window))
            seen = True
        return prob if seen else 0.0


def build_vad(config, sample_rate: int, observer=None) -> VADBackend:
    """Construct the configured backend, degrading to energy VAD on failure."""
    backend = str(config.get("backend", "silero")).lower()
    if backend == "energy":
        return EnergyVAD()
    if backend != "silero":
        raise ValueError(f"unknown vad backend {backend!r}")
    try:
        return SileroVAD(sample_rate=sample_rate)
    except Exception as exc:
        if observer:
            from juno_core.events import Stage

            observer.emit(
                Stage.VAD,
                "backend_fallback",
                requested="silero",
                using="energy",
                reason=str(exc),
            )
        return EnergyVAD()


class _Phase(Enum):
    SILENCE = "silence"
    SPEECH = "speech"


class SegmentDetector:
    """Frame stream in, ``SpeechSegment`` out.

        SILENCE -> SPEECH_START -> SPEECH -> SPEECH_END

    Onset requires ``min_speech_ms`` of voiced frames, which rejects coughs,
    door clicks and single-frame noise. Closure requires ``min_silence_ms`` of
    quiet -- the dominant term in how responsive the assistant feels, because
    nothing downstream can start until the utterance is declared finished.
    """

    def __init__(self, backend: VADBackend, config, sample_rate: int, preroll: float = 0.4):
        self.backend = backend
        self.sample_rate = sample_rate
        self.threshold = float(config.threshold)
        self._frame_s = None
        self.min_speech_ms = float(config.min_speech_ms)
        self.min_silence_ms = float(config.min_silence_ms)
        # Start transcribing before the segment is declared over.
        #
        # Whisper pads every input to a thirty-second window, so appending the
        # remaining silence changes nothing it sees: measured over four
        # utterances, transcribing at 250 ms of trailing silence returned a
        # byte-identical transcript to 550 ms, at the same cost. The hangover
        # is therefore time that can be spent transcribing rather than waited
        # out -- and it is the one part of the wait that no amount of faster
        # hardware would remove.
        self.provisional_silence_ms = float(
            config.get("provisional_silence_ms", 0) or 0
        )
        self._provisional_sent = False
        self.min_segment_s = float(config.min_segment_ms) / 1000.0
        self.max_segment_s = float(config.max_segment_ms) / 1000.0
        self._preroll = PrerollBuffer(sample_rate, preroll)
        self._accumulator = SegmentAccumulator(sample_rate, self.max_segment_s)
        self._phase = _Phase.SILENCE
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._probs: list[float] = []
        self.on_speech_start = None  # optional callback(timestamp)
        # optional callback(audio, samples) -- a head start, not a segment
        self.on_provisional = None

    def reset(self) -> None:
        self.backend.reset()
        self._preroll.clear()
        self._accumulator.reset()
        self._phase = _Phase.SILENCE
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._probs.clear()

    def push(self, frame: AudioFrame) -> SpeechSegment | None:
        """Feed one frame. Returns a segment on the frame that closes it."""
        if self._frame_s is None:
            self._frame_s = frame.samples.size / self.sample_rate
        frame_ms = 1000.0 * frame.samples.size / self.sample_rate

        prob = self.backend.probability(frame.samples)
        voiced = prob >= self.threshold

        if self._phase is _Phase.SILENCE:
            self._preroll.push(frame)
            if voiced:
                self._voiced_ms += frame_ms
                self._probs.append(prob)
                if self._voiced_ms >= self.min_speech_ms:
                    self._open(frame)
            else:
                self._voiced_ms = 0.0
                self._probs.clear()
            return None

        # -- in speech --
        self._accumulator.push(frame)
        self._probs.append(prob)
        if voiced:
            self._silence_ms = 0.0
            self._provisional_sent = False
        else:
            self._silence_ms += frame_ms
            # Enough quiet that this has probably ended, but not enough to say
            # so. Hand the audio over to be transcribed while the rest of the
            # hangover runs; if speech resumes, the caller throws the result
            # away and has lost only a transcription nobody waited for.
            if (
                self.provisional_silence_ms
                and not self._provisional_sent
                and self._silence_ms >= self.provisional_silence_ms
                and self.on_provisional is not None
            ):
                self._provisional_sent = True
                audio = self._accumulator.audio()
                if audio.size:
                    self.on_provisional(audio, audio.size)
            if self._silence_ms >= self.min_silence_ms:
                return self._close()
        if self._accumulator.is_full:
            return self._close()
        return None

    def _open(self, frame: AudioFrame) -> None:
        self._phase = _Phase.SPEECH
        self._silence_ms = 0.0
        self._provisional_sent = False
        self._accumulator.reset()
        # Replay the pre-roll so the leading consonant survives.
        self._accumulator.extend(self._preroll.drain())
        if self.on_speech_start:
            self.on_speech_start(frame.timestamp)

    def _close(self) -> SpeechSegment | None:
        audio = self._accumulator.audio()
        duration = self._accumulator.duration
        start = self._accumulator.start_time
        end = self._accumulator.end_time
        truncated = self._accumulator.truncated
        voiced_probs = [p for p in self._probs if p >= self.threshold]
        confidence = float(np.mean(voiced_probs)) if voiced_probs else 0.0

        self._phase = _Phase.SILENCE
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._provisional_sent = False
        self._probs.clear()
        self._accumulator.reset()
        self._preroll.clear()

        # Trailing silence is part of the buffer but not part of the utterance.
        speech_duration = duration - (self.min_silence_ms / 1000.0)
        if speech_duration < self.min_segment_s:
            return None
        return SpeechSegment(
            audio=audio,
            start_time=start,
            end_time=end,
            confidence=confidence,
            truncated=truncated,
            meta={"speech_duration": round(speech_duration, 3)},
        )

    @property
    def in_speech(self) -> bool:
        return self._phase is _Phase.SPEECH

    @property
    def speech_seconds(self) -> float:
        """How long the utterance in progress has been going."""
        return self._accumulator.duration if self.in_speech else 0.0

    def partial(self, seconds: float | None = None) -> np.ndarray | None:
        """The audio so far, mid-utterance.

        Lets something look at the opening of a sentence while the rest is
        still being said -- which is the only way to react to being addressed
        before the speaker has finished addressing you.
        """
        if not self.in_speech:
            return None
        audio = self._accumulator.audio()
        if seconds is not None:
            audio = audio[: int(seconds * self.sample_rate)]
        return audio if audio.size else None
