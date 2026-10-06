"""Local, offline speech-to-text on Apple Silicon via parakeet-mlx.

NVIDIA's Parakeet models (FastConformer encoder, TDT/CTC decoder; weights
CC BY 4.0) running on the Mac's GPU through MLX. Much faster than Whisper
at similar English accuracy -- measured on an M1 for a 2.6 s command:
45 ms for the 110M model and 85 ms for 0.6B, against 354 ms for Whisper
small.en and 1350 ms for large-v3-turbo. English only.

It is also the speech recogniser Reflex's default encoder comes from
(juno_core/slu/encoder.py), which is what makes it the honest baseline for
benchmarking Reflex: "skip STT" means less when STT is already cheap.

    stt:
      provider: parakeet
      model: parakeet-110m        # or parakeet-0.6b, or any HF repo / path

Needs:  pip install parakeet-mlx      (Apple Silicon)
CC BY requires attribution: "Parakeet, NVIDIA, CC BY 4.0".
"""

from __future__ import annotations

import time

import numpy as np

from . import STTEngine, Transcript

MODELS = {
    "parakeet-110m": "mlx-community/parakeet-tdt_ctc-110m",
    "parakeet-0.6b": "mlx-community/parakeet-tdt-0.6b-v2",
    "parakeet-0.6b-v3": "mlx-community/parakeet-tdt-0.6b-v3",
}


def resolve_model(name: str) -> str:
    return MODELS.get(name, name)


class ParakeetSTT(STTEngine):
    name = "parakeet"

    def __init__(self, model: str = "parakeet-110m") -> None:
        from .mlx_whisper import apple_silicon

        if not apple_silicon():
            raise RuntimeError("stt.provider 'parakeet' needs an Apple Silicon Mac")
        try:
            import mlx.core as mx
            from parakeet_mlx import from_pretrained
            from parakeet_mlx.audio import get_logmel
        except ImportError as exc:
            raise ImportError("stt.provider is 'parakeet' but the package isn't installed. "
                              "Run: pip install parakeet-mlx") from exc
        self._mx = mx
        self._get_logmel = get_logmel
        self.model_name = resolve_model(model)
        self.model = from_pretrained(self.model_name)
        self.name = f"parakeet ({model})"

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        started = time.monotonic()
        mx = self._mx
        pcm = np.asarray(audio, dtype=np.float32).ravel()
        if pcm.size < 4000:
            pcm = np.pad(pcm, (0, 4000 - pcm.size))
        mel = self._get_logmel(mx.array(pcm), self.model.preprocessor_config)
        result = self.model.generate(mel.astype(mx.bfloat16))[0]
        tokens = result.tokens
        text = result.text.strip()
        confidence = float(np.mean([t.confidence for t in tokens])) if tokens else None
        tail = tokens[-max(1, len(tokens) // 3):] if tokens else []
        tail_conf = float(np.mean([t.confidence for t in tail])) if tail else None
        return Transcript(
            text=text,
            language="en",
            duration=pcm.size / float(sample_rate),
            latency=time.monotonic() - started,
            reliable=confidence is None or confidence >= 0.5,
            tail_reliable=tail_conf is None or tail_conf >= 0.5,
            meta={"confidence": confidence},
        )
