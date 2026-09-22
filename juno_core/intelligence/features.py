"""Acoustic and conversational feature extraction for pre-transcription gating.

Pure NumPy only. Level and distance are explicitly NOT features: all acoustic
features are strictly scale-invariant so speech from across the room is not
penalised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

SAMPLE_RATE = 16000
FRAME_LEN = 480      # 30 ms
HOP = 160            # 10 ms
N_FFT = 512
FMIN_HZ = 70.0
FMAX_HZ = 350.0

FEATURE_NAMES = (
    "f0_mean",
    "f0_std",
    "f0_slope",
    "voiced_ratio",
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_tilt_log",
    "zcr",
    "pause_before",
    "duration",
    "vad_confidence",
    "p_own",
    "voice_confident",
    "is_wearer",
    "since_ai_log",
    "ignored_run",
    "accepted_recently",
    "awaiting_answer",
    "confirming",
    "offer_pending",
)


@dataclass(frozen=True)
class ProsodyFeatures:
    f0_mean: float
    f0_std: float
    f0_slope: float
    voiced_ratio: float
    spectral_centroid: float
    spectral_rolloff: float
    spectral_tilt_log: float
    zcr: float
    pause_before: float

    def as_dict(self) -> dict[str, float]:
        return {
            "f0_mean": round(self.f0_mean, 2),
            "f0_std": round(self.f0_std, 2),
            "f0_slope": round(self.f0_slope, 4),
            "voiced_ratio": round(self.voiced_ratio, 4),
            "spectral_centroid": round(self.spectral_centroid, 1),
            "spectral_rolloff": round(self.spectral_rolloff, 1),
            "spectral_tilt_log": round(self.spectral_tilt_log, 4),
            "zcr": round(self.zcr, 4),
            "pause_before": round(self.pause_before, 3),
        }


_EMPTY_PROSODY = ProsodyFeatures(
    f0_mean=0.0, f0_std=0.0, f0_slope=0.0, voiced_ratio=0.0,
    spectral_centroid=0.0, spectral_rolloff=0.0, spectral_tilt_log=0.0,
    zcr=0.0, pause_before=0.0,
)

# Frames per FFT batch. Bounds peak memory on a max-length segment (a few MB)
# while keeping each batch large enough that NumPy, not Python, does the work.
_CHUNK = 512
_WINDOW = np.hanning(FRAME_LEN)
_FREQS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _acf_size(max_lag: int) -> int:
    """Smallest 5-smooth FFT length that gives exact lags 0..max_lag.

    A circular autocorrelation of length n only wraps lags above
    n - FRAME_LEN, and pitch never looks past max_lag -- so FRAME_LEN +
    max_lag points suffice (720 at 16 kHz, against 1024 for the full-length
    correlation). 5-smooth sizes are the ones pocketfft is fast at.
    """
    n = FRAME_LEN + max_lag
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _freqs(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    cached = _FREQS_CACHE.get(sample_rate)
    if cached is None:
        freqs = np.fft.rfftfreq(N_FFT, 1.0 / sample_rate)
        cached = (freqs, freqs < 1000.0)
        _FREQS_CACHE[sample_rate] = cached
    return cached


def extract_prosody(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> ProsodyFeatures:
    """Extract scale-invariant prosody features from mono PCM audio.

    Runs before transcription, on the critical path, so every per-frame step
    is batched: frames are strided views of the signal (no copies), pitch comes
    from an FFT autocorrelation of all frames at once, and the spectrum is
    computed once per batch. The values match the frame-by-frame reference to
    floating-point precision; a model trained on either reads the same.
    """
    pcm = np.asarray(audio, dtype=np.float32).ravel()
    if pcm.size < FRAME_LEN:
        return _EMPTY_PROSODY

    # 1. Zero crossing rate
    signs = np.signbit(pcm)
    zcr = float(np.count_nonzero(signs[1:] != signs[:-1])) / (pcm.size - 1)

    n_frames = (pcm.size - FRAME_LEN) // HOP
    if n_frames <= 0:
        return ProsodyFeatures(
            f0_mean=0.0, f0_std=0.0, f0_slope=0.0, voiced_ratio=0.0,
            spectral_centroid=0.0, spectral_rolloff=0.0, spectral_tilt_log=0.0,
            zcr=zcr, pause_before=0.0,
        )
    frames = np.lib.stride_tricks.sliding_window_view(pcm, FRAME_LEN)[::HOP][:n_frames]

    # 2. Pause before speech onset (scale-invariant relative threshold)
    frame_energies = np.mean(frames * frames, axis=1)
    max_energy = float(np.max(frame_energies))
    pause_before = 0.0
    if max_energy > 1e-8:
        above = np.flatnonzero(frame_energies > 0.02 * max_energy)
        if above.size > 0:
            pause_before = float(above[0] * HOP) / float(sample_rate)

    min_lag = max(1, int(sample_rate / FMAX_HZ))
    max_lag = min(FRAME_LEN - 1, int(sample_rate / FMIN_HZ))
    acf_n = _acf_size(max_lag)
    freqs, low_mask = _freqs(sample_rate)

    pitch_parts: list[np.ndarray] = []
    centroid_parts: list[np.ndarray] = []
    rolloff_parts: list[np.ndarray] = []
    tilt_parts: list[np.ndarray] = []

    for lo in range(0, n_frames, _CHUNK):
        block = frames[lo : lo + _CHUNK].astype(np.float64)

        # 3. F0 via normalised autocorrelation, all frames at once
        centred = block - block.mean(axis=1, keepdims=True)
        e0 = np.einsum("ij,ij->i", centred, centred)
        spec = np.fft.rfft(centred, n=acf_n, axis=1)
        acf = np.fft.irfft(spec.real ** 2 + spec.imag ** 2, n=acf_n, axis=1)
        lags = acf[:, min_lag : max_lag + 1]
        best = np.argmax(lags, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            peak = lags[np.arange(lags.shape[0]), best] / e0
        voiced = (e0 >= 1e-8) & (peak >= 0.35)
        pitch_parts.append(float(sample_rate) / (min_lag + best[voiced]))

        # 4. Spectral shape (centroid, rolloff, tilt)
        power = np.fft.rfft(block * _WINDOW, n=N_FFT, axis=1)
        power = power.real ** 2 + power.imag ** 2
        total = power.sum(axis=1)
        keep = total >= 1e-8
        if not np.any(keep):
            continue
        power, total = power[keep], total[keep]
        centroid_parts.append(power @ freqs / total)
        cum = np.cumsum(power, axis=1)
        idx = np.minimum(np.sum(cum < (0.85 * total)[:, None], axis=1), freqs.size - 1)
        rolloff_parts.append(freqs[idx])
        p_low = power[:, low_mask].sum(axis=1)
        tilt_parts.append(p_low / (total - p_low + 1e-8))

    pitches = np.concatenate(pitch_parts)
    voiced_ratio = pitches.size / max(1, n_frames)
    if pitches.size:
        f0_mean = float(np.mean(pitches))
        f0_std = float(np.std(pitches))
        if pitches.size >= 4:
            q_len = max(1, pitches.size // 4)
            q1 = float(np.mean(pitches[:q_len]))
            q4 = float(np.mean(pitches[-q_len:]))
            f0_slope = (q4 - q1) / max(1.0, f0_mean)
        else:
            f0_slope = 0.0
    else:
        f0_mean = f0_std = f0_slope = 0.0

    def _mean(parts: list[np.ndarray]) -> float:
        return float(np.mean(np.concatenate(parts))) if parts else 0.0

    mean_tilt = _mean(tilt_parts)
    return ProsodyFeatures(
        f0_mean=f0_mean,
        f0_std=f0_std,
        f0_slope=f0_slope,
        voiced_ratio=voiced_ratio,
        spectral_centroid=_mean(centroid_parts),
        spectral_rolloff=_mean(rolloff_parts),
        spectral_tilt_log=math.log1p(max(0.0, mean_tilt)),
        zcr=zcr,
        pause_before=pause_before,
    )


def extract_gate_features(
    audio: np.ndarray | None,
    sample_rate: int,
    acoustics,
    snapshot,
) -> np.ndarray:
    """Combine prosody, acoustics, and snapshot into a 20-dim feature vector."""
    if audio is not None and len(audio) > 0:
        prosody = extract_prosody(audio, sample_rate)
    else:
        prosody = _EMPTY_PROSODY

    # Conversational run
    run = 0
    recent = getattr(snapshot, "recent_verdicts", ())
    for verdict in reversed(recent):
        if verdict is False:
            run += 1
        else:
            break
    ignored_run = min(float(run), 5.0)
    accepted_recently = 1.0 if recent and recent[-1] is True else 0.0

    since_ai = getattr(snapshot, "since_ai", 1e9)
    since_ai_val = 3600.0 if since_ai is None or since_ai > 3600.0 else float(since_ai)
    since_ai_log = math.log1p(max(0.0, since_ai_val))

    # Voiceprint estimates
    p_own = getattr(acoustics, "p_own", None)
    p_own_val = 0.5 if p_own is None else float(p_own)
    voice_conf = 1.0 if getattr(acoustics, "voice_confident", False) else 0.0
    is_wearer_attr = getattr(acoustics, "is_wearer", None)
    is_wearer = 0.5 if is_wearer_attr is None else (1.0 if is_wearer_attr else 0.0)

    vec = [
        prosody.f0_mean,
        prosody.f0_std,
        prosody.f0_slope,
        prosody.voiced_ratio,
        prosody.spectral_centroid,
        prosody.spectral_rolloff,
        prosody.spectral_tilt_log,
        prosody.zcr,
        prosody.pause_before,
        float(getattr(acoustics, "seconds", 0.0)),
        float(getattr(acoustics, "confidence", 0.0)),
        p_own_val,
        voice_conf,
        is_wearer,
        since_ai_log,
        ignored_run,
        accepted_recently,
        1.0 if getattr(snapshot, "awaiting_answer", False) else 0.0,
        1.0 if getattr(snapshot, "confirming", False) else 0.0,
        1.0 if getattr(snapshot, "offer_pending", False) else 0.0,
    ]
    return np.asarray(vec, dtype=np.float32)
