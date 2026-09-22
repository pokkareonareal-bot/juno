"""Local, offline speech-to-text on Apple Silicon via mlx-whisper.

RECOMMENDED on an M-series Mac. MLX is Apple's array framework for Apple
Silicon: it runs Whisper on the GPU through Metal, in unified memory.
faster-whisper's CTranslate2 backend has no Metal support, so on a Mac it
runs on the CPU only. On Linux, Windows or an Intel Mac, MLX isn't available;
use faster_whisper there (``stt.provider: auto`` picks for you).

First run downloads the converted weights from Hugging Face once (the
mlx-community repos, a few hundred MB for small.en) and caches them under
~/.cache/huggingface. After that nothing leaves the machine.

Needs:  pip install mlx-whisper      (Apple Silicon, macOS 13.5+)
"""

from __future__ import annotations

import platform
import sys
import time

import numpy as np

from . import STTEngine, Transcript, finite

# The same short names faster-whisper takes, so `stt.model` means the same
# thing whichever provider is picked. Anything else is passed through as a
# Hugging Face repo or a local path.
MODELS = {
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def resolve_model(name: str) -> str:
    return MODELS.get(name, name)


class MLXWhisperSTT(STTEngine):
    name = "mlx-whisper"

    def __init__(self, model: str = "small.en") -> None:
        if not apple_silicon():
            raise RuntimeError(
                "stt.provider is 'mlx_whisper', which needs an Apple Silicon Mac. "
                "Use faster_whisper here (or stt.provider: auto).")
        try:
            import mlx_whisper
        except ImportError as exc:
            raise ImportError(
                "stt.provider is 'mlx_whisper' but the package isn't "
                "installed. Run: pip install mlx-whisper"
            ) from exc
        self._mlx_whisper = mlx_whisper
        self.model = resolve_model(model)

    def warmup(self) -> None:
        # Loads the weights and compiles the Metal kernels, which is most of
        # the first call's cost -- better paid before anyone is speaking.
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        started = time.monotonic()
        pcm = np.asarray(audio, dtype=np.float32)
        result = self._mlx_whisper.transcribe(
            pcm,
            path_or_hf_repo=self.model,
            language="en",
            verbose=None,
            # One utterance at a time: no earlier text worth conditioning on,
            # and conditioning is how a hallucinated loop feeds itself.
            condition_on_previous_text=False,
        )
        pieces = result.get("segments") or []
        text = str(result.get("text") or "").strip()
        avg_logprob = finite(
            sum(float(s.get("avg_logprob", 0.0)) for s in pieces) / len(pieces)
            if pieces else None
        )
        no_speech = finite(max((float(s.get("no_speech_prob", 0.0)) for s in pieces),
                               default=None) if pieces else None)
        tail = finite(pieces[-1].get("avg_logprob")) if pieces else None
        reliable = avg_logprob is None or avg_logprob > -1.0
        return Transcript(
            text=text,
            language=result.get("language"),
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech,
            duration=pcm.size / float(sample_rate),
            latency=time.monotonic() - started,
            reliable=reliable,
            tail_reliable=tail is None or tail > -1.0,
            tail_logprob=tail,
        )
