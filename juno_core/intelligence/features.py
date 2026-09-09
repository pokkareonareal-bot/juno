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


def extract_prosody(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> ProsodyFeatures:
    """Extract scale-invariant prosody features from mono PCM audio."""
    pcm = np.asarray(audio, dtype=np.float32).ravel()
    if pcm.size < FRAME_LEN:
        return ProsodyFeatures(
            f0_mean=0.0, f0_std=0.0, f0_slope=0.0, voiced_ratio=0.0,
            spectral_centroid=0.0, spectral_rolloff=0.0, spectral_tilt_log=0.0,
            zcr=0.0, pause_before=0.0,
        )

    # 1. Zero crossing rate
    signs = np.signbit(pcm)
    zcr = float(np.mean(np.abs(np.diff(signs)))) if signs.size > 1 else 0.0

    # 2. Pause before speech onset (scale-invariant relative threshold)
    n_frames = (pcm.size - FRAME_LEN) // HOP
    frame_energies = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * HOP
        f = pcm[start : start + FRAME_LEN]
        frame_energies[i] = np.mean(f * f)

    max_energy = float(np.max(frame_energies)) if frame_energies.size > 0 else 0.0
    pause_before = 0.0
    if max_energy > 1e-8:
        threshold = 0.02 * max_energy
        above = np.where(frame_energies > threshold)[0]
        if above.size > 0:
            pause_before = float(above[0] * HOP) / float(sample_rate)

    # 3. F0 pitch tracking via autocorrelation
    min_lag = max(1, int(sample_rate / FMAX_HZ))
    max_lag = min(FRAME_LEN - 1, int(sample_rate / FMIN_HZ))
    pitches: list[float] = []

    for i in range(n_frames):
        start = i * HOP
        frame = pcm[start : start + FRAME_LEN]
        frame = frame - np.mean(frame)
        e0 = float(np.dot(frame, frame))
        if e0 < 1e-8:
            continue
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(frame) - 1 :]
        r_norm = corr / e0
        if max_lag + 1 > len(r_norm):
            continue
        lag_window = r_norm[min_lag : max_lag + 1]
        best_offset = int(np.argmax(lag_window))
        peak_lag = min_lag + best_offset
        peak_val = float(r_norm[peak_lag])
        if peak_val >= 0.35:
            pitches.append(float(sample_rate) / peak_lag)

    voiced_ratio = len(pitches) / max(1, n_frames)
    if pitches:
        f0_mean = float(np.mean(pitches))
        f0_std = float(np.std(pitches))
        if len(pitches) >= 4:
            q_len = max(1, len(pitches) // 4)
            q1 = float(np.mean(pitches[:q_len]))
            q4 = float(np.mean(pitches[-q_len:]))
            f0_slope = (q4 - q1) / max(1.0, f0_mean)
        else:
            f0_slope = 0.0
    else:
        f0_mean = 0.0
        f0_std = 0.0
        f0_slope = 0.0

    # 4. Spectral features (centroid, rolloff, tilt)
    window = np.hanning(FRAME_LEN)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sample_rate)
    low_mask = freqs < 1000.0

    centroids: list[float] = []
    rolloffs: list[float] = []
    tilts: list[float] = []

    for i in range(n_frames):
        start = i * HOP
        frame = pcm[start : start + FRAME_LEN] * window
        power = np.abs(np.fft.rfft(frame, n=N_FFT)) ** 2
        total_p = float(np.sum(power))
        if total_p < 1e-8:
            continue
        centroid = float(np.sum(freqs * power) / total_p)
        centroids.append(centroid)

        cum = np.cumsum(power)
        idx = int(np.searchsorted(cum, 0.85 * total_p))
        rolloffs.append(float(freqs[min(idx, len(freqs) - 1)]))

        p_low = float(np.sum(power[low_mask]))
        p_high = float(np.sum(power[~low_mask]))
        tilts.append(p_low / (p_high + 1e-8))

    mean_centroid = float(np.mean(centroids)) if centroids else 0.0
    mean_rolloff = float(np.mean(rolloffs)) if rolloffs else 0.0
    mean_tilt = float(np.mean(tilts)) if tilts else 0.0
    spectral_tilt_log = math.log1p(max(0.0, mean_tilt))

    return ProsodyFeatures(
        f0_mean=f0_mean,
        f0_std=f0_std,
        f0_slope=f0_slope,
        voiced_ratio=voiced_ratio,
        spectral_centroid=mean_centroid,
        spectral_rolloff=mean_rolloff,
        spectral_tilt_log=spectral_tilt_log,
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
        prosody = ProsodyFeatures(
            f0_mean=0.0, f0_std=0.0, f0_slope=0.0, voiced_ratio=0.0,
            spectral_centroid=0.0, spectral_rolloff=0.0, spectral_tilt_log=0.0,
            zcr=0.0, pause_before=0.0,
        )

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
