"""Audio in, one vector out: what the student listens with.

The student does not learn to hear from scratch. It borrows the first half
of a speech recogniser -- the ENCODER, which turns sound into a sequence of
frames that already carry phonetic and lexical information -- and skips the
second half, the DECODER, which turns those frames into words one token at
a time. Words are exactly what Reflex is trying not to make.

Measured on an M1 (2.6 s command, median of 5, see benchmarks in the README):

    parakeet-tdt_ctc-110m   encoder 17 ms    full transcription  45 ms
    parakeet-tdt-0.6b-v2    encoder 64 ms    full transcription  85 ms
    whisper small.en                          full transcription 354 ms
    whisper large-v3-turbo                    full transcription 1350 ms

Three encoders, so the comparison can be made rather than assumed:

  ``parakeet``  NVIDIA's FastConformer (CC BY 4.0 weights), via parakeet-mlx.
                Variable-length input natively. ``layers`` runs only the
                first N blocks -- an early exit, cheaper, with less lexical
                detail in the frames. The default.
  ``whisper``   OpenAI Whisper's encoder (MIT), via mlx-whisper. Whisper
                pads every clip to 30 s; here the positional embedding is
                cut to the clip's length instead (the trick whisper.cpp
                calls audio_ctx), which is what makes it affordable at all.
  ``logmel``    No pretrained model: log-mel statistics in numpy. The
                baseline that answers "is the pretrained encoder doing the
                work, or would any acoustic features do?"
  ``gate``      The 20 hand-made features the pre-STT gate uses (pitch,
                voicing, spectral shape, duration; context held at a cold
                start). The same student and thresholds on top of them is the
                fairest comparison with the existing learned gate.

All three return frames; ``pool`` turns frames into a fixed-length vector
(mean, standard deviation and maximum over time, optionally also the mean of
each third, which keeps a little of the order). The pooled vector is what
the student's heads read and what training tables store. Never audio.

A vector is only meaningful to a student trained on the same encoder, so
every encoder has a ``spec`` string that the model file records and the
runtime checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

POOLINGS = ("stats", "stats_thirds")

DEFAULT_PARAKEET = "mlx-community/parakeet-tdt_ctc-110m"
DEFAULT_WHISPER = "mlx-community/whisper-small.en-mlx"


class AudioEncoder:
    """Frames from audio. Subclasses implement ``frames``."""

    name = "base"
    dim = 0

    def __init__(self, pooling: str = "stats") -> None:
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}")
        self.pooling = pooling

    @property
    def spec(self) -> str:
        raise NotImplementedError

    def frames(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        raise NotImplementedError

    def encode(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """The pooled vector for one clip."""
        return pool(self.frames(audio, sample_rate), self.pooling)

    @property
    def vector_dim(self) -> int:
        return self.dim * (3 if self.pooling == "stats" else 6)

    def warmup(self) -> None:
        """Pay compilation and first-call costs before anyone speaks."""
        noise = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.01
        self.encode(noise)


def pool(frames: np.ndarray, pooling: str = "stats") -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[0] == 0:
        raise ValueError("pool needs a (time, dim) array with at least one frame")
    parts = [frames.mean(axis=0), frames.std(axis=0), frames.max(axis=0)]
    if pooling == "stats_thirds":
        for chunk in np.array_split(frames, 3, axis=0):
            parts.append(chunk.mean(axis=0) if chunk.shape[0] else frames.mean(axis=0))
    return np.concatenate(parts).astype(np.float32)


def _as_pcm(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != 16000:
        raise ValueError(f"encoders take 16 kHz audio, not {sample_rate} Hz")
    pcm = np.asarray(audio, dtype=np.float32).ravel()
    # Shorter than one analysis window and every encoder below produces
    # nothing; pad with silence rather than fail on a clipped "no".
    if pcm.size < 4000:
        pcm = np.pad(pcm, (0, 4000 - pcm.size))
    return pcm


class ParakeetEncoder(AudioEncoder):
    name = "parakeet"

    def __init__(self, repo: str = DEFAULT_PARAKEET, layers: int | None = None,
                 pooling: str = "stats") -> None:
        super().__init__(pooling)
        try:
            import mlx.core as mx
            from parakeet_mlx import from_pretrained
            from parakeet_mlx.audio import get_logmel
        except ImportError as exc:
            raise ImportError("the parakeet encoder needs parakeet-mlx on Apple Silicon: "
                              "pip install -e '.[slu]'") from exc
        self._mx = mx
        self._get_logmel = get_logmel
        self.repo = repo
        self.model = from_pretrained(repo)
        encoder = self.model.encoder
        total = len(encoder.layers)
        self.layers = total if layers is None else max(1, min(int(layers), total))
        self.dim = int(self.model.encoder_config.d_model)

    @property
    def spec(self) -> str:
        return f"parakeet:{self.repo}@L{self.layers}:{self.pooling}"

    def frames(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        mx = self._mx
        pcm = _as_pcm(audio, sample_rate)
        mel = self._get_logmel(mx.array(pcm), self.model.preprocessor_config)
        mel = mel.astype(mx.bfloat16)
        enc = self.model.encoder
        lengths = mx.full((1,), mel.shape[-2], dtype=mx.int64)
        x, _ = enc.pre_encode(mel, lengths)
        pos_emb = None
        if enc.pos_enc is not None:
            x, pos_emb = enc.pos_enc(x, offset=0)
        for layer in enc.layers[: self.layers]:
            x = layer(x, pos_emb=pos_emb)
        out = x[0].astype(mx.float32)
        mx.eval(out)
        return np.array(out)


class WhisperEncoder(AudioEncoder):
    name = "whisper"

    def __init__(self, repo: str = DEFAULT_WHISPER, layers: int | None = None,
                 pooling: str = "stats") -> None:
        super().__init__(pooling)
        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_whisper.audio import log_mel_spectrogram
            from mlx_whisper.load_models import load_model
        except ImportError as exc:
            raise ImportError("the whisper encoder needs mlx-whisper on Apple Silicon") from exc
        self._mx, self._nn = mx, nn
        self._log_mel = log_mel_spectrogram
        self.repo = repo
        self.model = load_model(repo, dtype=mx.float16)
        self.n_mels = int(self.model.dims.n_mels)
        total = len(self.model.encoder.blocks)
        self.layers = total if layers is None else max(1, min(int(layers), total))
        self.dim = int(self.model.dims.n_audio_state)

    @property
    def spec(self) -> str:
        return f"whisper:{self.repo}@L{self.layers}:{self.pooling}"

    def frames(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        mx, nn = self._mx, self._nn
        pcm = _as_pcm(audio, sample_rate)
        mel = self._log_mel(pcm, n_mels=self.n_mels)
        max_frames = 3000
        mel = mel[:max_frames]
        enc = self.model.encoder
        x = mel[None].astype(mx.float16)
        x = nn.gelu(enc.conv1(x))
        x = nn.gelu(enc.conv2(x))
        x = x + enc._positional_embedding[: x.shape[1]]
        for block in enc.blocks[: self.layers]:
            x, _, _ = block(x)
        if self.layers == len(enc.blocks):
            x = enc.ln_post(x)
        out = x[0].astype(mx.float32)
        mx.eval(out)
        return np.array(out)


class LogMelEncoder(AudioEncoder):
    """64 log-mel bands and their deltas, numpy only. The no-pretraining baseline."""

    name = "logmel"

    def __init__(self, n_mels: int = 64, pooling: str = "stats") -> None:
        super().__init__(pooling)
        self.n_mels = n_mels
        self.n_fft = 400
        self.hop = 160
        self.dim = n_mels * 2
        self._fb = _mel_filterbank(16000, self.n_fft, n_mels)
        self._window = np.hanning(self.n_fft).astype(np.float32)

    @property
    def spec(self) -> str:
        return f"logmel:{self.n_mels}:{self.pooling}"

    def frames(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        pcm = _as_pcm(audio, sample_rate)
        peak = float(np.max(np.abs(pcm))) or 1.0
        pcm = pcm / peak                                  # level-invariant
        n = 1 + (pcm.size - self.n_fft) // self.hop
        idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(n)[:, None]
        spec = np.abs(np.fft.rfft(pcm[idx] * self._window, axis=1)) ** 2
        mel = np.log(spec @ self._fb.T + 1e-6)
        mel -= mel.mean(axis=0, keepdims=True)            # per-clip channel norm
        delta = np.diff(mel, axis=0, prepend=mel[:1])
        return np.concatenate([mel, delta], axis=1).astype(np.float32)


class GateFeatureEncoder(AudioEncoder):
    """The gate's 20 features as a vector -- no frames, already pooled."""

    name = "gate"

    def __init__(self, pooling: str = "stats") -> None:
        super().__init__(pooling)
        from juno_core.intelligence.features import FEATURE_NAMES

        self.dim = len(FEATURE_NAMES)

    @property
    def spec(self) -> str:
        return "gate:20"

    @property
    def vector_dim(self) -> int:
        return self.dim

    def encode(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        from juno_core.intelligence.features import extract_gate_features
        from juno_core.intelligence.gate import NEVER_SPOKEN, Acoustics, Snapshot

        pcm = np.asarray(audio, dtype=np.float32).ravel()
        seconds = pcm.size / float(sample_rate)
        return np.asarray(extract_gate_features(pcm, sample_rate, Acoustics(seconds, 1.0),
                                                Snapshot(since_ai=NEVER_SPOKEN)), np.float32)


def _mel_filterbank(rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    bins = n_fft // 2 + 1
    freqs = np.linspace(0, rate / 2, bins)
    points = mel_to_hz(np.linspace(hz_to_mel(20.0), hz_to_mel(rate / 2), n_mels + 2))
    fb = np.zeros((n_mels, bins), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = points[i], points[i + 1], points[i + 2]
        up = (freqs - lo) / max(mid - lo, 1e-9)
        down = (hi - freqs) / max(hi - mid, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


@dataclass(frozen=True)
class EncoderChoice:
    kind: str = "parakeet"
    repo: str | None = None
    layers: int | None = None
    pooling: str = "stats"


def parse_spec(spec: str) -> EncoderChoice:
    """'parakeet:repo@L8:stats' -> EncoderChoice. The inverse of ``.spec``."""
    kind, _, rest = spec.partition(":")
    if kind == "gate":
        return EncoderChoice("gate")
    if kind == "logmel":
        n_mels, _, pooling = rest.partition(":")
        return EncoderChoice("logmel", repo=n_mels or "64", pooling=pooling or "stats")
    repo_layers, _, pooling = rest.rpartition(":")
    repo, _, layers = repo_layers.rpartition("@L")
    return EncoderChoice(kind, repo=repo or None, layers=int(layers) if layers else None,
                         pooling=pooling or "stats")


def build_encoder(choice: EncoderChoice | str) -> AudioEncoder:
    if isinstance(choice, str):
        choice = parse_spec(choice) if ":" in choice else EncoderChoice(choice)
    if choice.kind == "parakeet":
        return ParakeetEncoder(choice.repo or DEFAULT_PARAKEET, choice.layers, choice.pooling)
    if choice.kind == "whisper":
        return WhisperEncoder(choice.repo or DEFAULT_WHISPER, choice.layers, choice.pooling)
    if choice.kind == "logmel":
        return LogMelEncoder(int(choice.repo or 64), choice.pooling)
    if choice.kind == "gate":
        return GateFeatureEncoder()
    raise ValueError(f"unknown encoder {choice.kind!r} (parakeet | whisper | logmel | gate)")


def time_encoder(encoder: AudioEncoder, seconds: float = 2.5, runs: int = 5) -> float:
    """Median milliseconds to encode one clip of this length."""
    clip = np.random.default_rng(1).standard_normal(int(seconds * 16000)).astype(np.float32) * 0.05
    encoder.encode(clip)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        encoder.encode(clip)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000.0)
