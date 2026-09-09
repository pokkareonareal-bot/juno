"""Local, offline speech-to-text via faster-whisper (CTranslate2).

RECOMMENDED if you want nothing to leave the machine, ever -- pair it with
llm/ollama.py for a fully offline pipeline. Runs fine on CPU; faster with a
GPU. First run downloads the model weights once (a few hundred MB, cached
under ~/.cache).

Needs:  pip install faster-whisper
"""

from __future__ import annotations

import re
import time

import numpy as np

from . import STTEngine, Transcript, finite

_LOOP = re.compile(r"^(.+?)\1{2,}$")


class FasterWhisperSTT(STTEngine):
    name = "faster-whisper"

    def __init__(self, model: str = "small.en", device: str = "cpu",
                 compute_type: str = "int8") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "stt.provider is 'faster_whisper' but the package isn't "
                "installed. Run: pip install faster-whisper"
            ) from exc
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        started = time.monotonic()
        segments, info = self._model.transcribe(
            np.asarray(audio, dtype=np.float32), language="en", vad_filter=False,
        )
        pieces = list(segments)
        text = " ".join(s.text.strip() for s in pieces).strip()
        avg_logprob = finite(
            sum(s.avg_logprob for s in pieces) / len(pieces) if pieces else None
        )
        reliable = avg_logprob is None or avg_logprob > -1.0
        return Transcript(
            text=text,
            language=getattr(info, "language", None),
            avg_logprob=avg_logprob,
            no_speech_prob=finite(getattr(info, "no_speech_prob", None)),
            duration=len(audio) / float(sample_rate),
            latency=time.monotonic() - started,
            reliable=reliable,
            tail_reliable=reliable,
        )
