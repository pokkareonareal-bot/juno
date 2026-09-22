"""Pre-STT gate: feature speed-up fidelity, safety fallbacks, offline tooling.

Synthetic signals only -- no recorded speech ships with this repository.
Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import re
import tempfile
import time
import unittest
import wave
from pathlib import Path

import numpy as np

from juno_core.intelligence import features as F
from juno_core.intelligence import gate_training as T
from juno_core.intelligence.gate import NEVER_SPOKEN, Acoustics, Gate, Snapshot

ROOT = Path(__file__).resolve().parent.parent


def reference_prosody(audio, sample_rate=16000):
    """The original frame-by-frame implementation, kept as the oracle."""
    pcm = np.asarray(audio, dtype=np.float32).ravel()
    if pcm.size < F.FRAME_LEN:
        return F._EMPTY_PROSODY
    signs = np.signbit(pcm)
    zcr = float(np.mean(signs[1:] != signs[:-1]))
    n_frames = (pcm.size - F.FRAME_LEN) // F.HOP
    energies = np.array([np.mean(pcm[i * F.HOP:i * F.HOP + F.FRAME_LEN] ** 2)
                         for i in range(n_frames)], dtype=np.float32)
    pause = 0.0
    if energies.size and energies.max() > 1e-8:
        above = np.where(energies > 0.02 * energies.max())[0]
        if above.size:
            pause = float(above[0] * F.HOP) / sample_rate
    min_lag = max(1, int(sample_rate / F.FMAX_HZ))
    max_lag = min(F.FRAME_LEN - 1, int(sample_rate / F.FMIN_HZ))
    pitches, cents, rolls, tilts = [], [], [], []
    window = np.hanning(F.FRAME_LEN)
    freqs = np.fft.rfftfreq(F.N_FFT, 1.0 / sample_rate)
    low = freqs < 1000.0
    for i in range(n_frames):
        frame = pcm[i * F.HOP:i * F.HOP + F.FRAME_LEN].astype(np.float64)
        c = frame - frame.mean()
        e0 = float(np.dot(c, c))
        if e0 >= 1e-8:
            r = np.correlate(c, c, mode="full")[c.size - 1:] / e0
            lag = min_lag + int(np.argmax(r[min_lag:max_lag + 1]))
            if r[lag] >= 0.35:
                pitches.append(sample_rate / lag)
        power = np.abs(np.fft.rfft(frame * window, n=F.N_FFT)) ** 2
        total = power.sum()
        if total < 1e-8:
            continue
        cents.append(float(np.sum(freqs * power) / total))
        idx = int(np.searchsorted(np.cumsum(power), 0.85 * total))
        rolls.append(float(freqs[min(idx, freqs.size - 1)]))
        tilts.append(power[low].sum() / (power[~low].sum() + 1e-8))
    if pitches:
        f0_mean, f0_std = float(np.mean(pitches)), float(np.std(pitches))
        q = max(1, len(pitches) // 4)
        slope = ((np.mean(pitches[-q:]) - np.mean(pitches[:q])) / max(1.0, f0_mean)
                 if len(pitches) >= 4 else 0.0)
    else:
        f0_mean = f0_std = slope = 0.0
    return F.ProsodyFeatures(
        f0_mean=f0_mean, f0_std=f0_std, f0_slope=float(slope),
        voiced_ratio=len(pitches) / max(1, n_frames),
        spectral_centroid=float(np.mean(cents)) if cents else 0.0,
        spectral_rolloff=float(np.mean(rolls)) if rolls else 0.0,
        spectral_tilt_log=math.log1p(max(0.0, float(np.mean(tilts)) if tilts else 0.0)),
        zcr=zcr, pause_before=pause)


def synthetic_voice(seconds, f0=140.0, sr=16000, seed=0, level=0.1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.1 * np.sin(2 * np.pi * 0.5 * t))) / sr
    harmonics = sum(np.sin(k * phase) / k for k in range(1, 8))
    envelope = (np.sin(2 * np.pi * 1.5 * t) > -0.3).astype(float)
    return (level * harmonics * envelope
            + 0.005 * rng.standard_normal(t.size)).astype(np.float32)


class ProsodyFidelity(unittest.TestCase):
    def assertSame(self, a, b):
        for key, va in a.as_dict().items():
            vb = b.as_dict()[key]
            self.assertAlmostEqual(va, vb, delta=max(1e-3, 1e-3 * abs(vb)), msg=key)

    def test_matches_reference(self):
        rng = np.random.default_rng(1)
        cases = [
            synthetic_voice(0.5), synthetic_voice(3.0, f0=210, seed=2),
            synthetic_voice(12.0, f0=95, seed=3),
            (0.1 * rng.standard_normal(16000)).astype(np.float32),
            np.zeros(8000, dtype=np.float32),
            np.concatenate([np.zeros(4000), synthetic_voice(1.0)]).astype(np.float32),
            synthetic_voice(0.05)[: F.FRAME_LEN],
            synthetic_voice(0.05)[: F.FRAME_LEN + F.HOP + 1],
            np.zeros(100, dtype=np.float32),
        ]
        for audio in cases:
            self.assertSame(F.extract_prosody(audio), reference_prosody(audio))

    def test_scale_invariant(self):
        a = synthetic_voice(2.0)
        self.assertSame(F.extract_prosody(a), F.extract_prosody(a * 0.05))

    def test_faster_than_reference(self):
        audio = synthetic_voice(10.0)
        F.extract_prosody(audio)
        t0 = time.perf_counter()
        F.extract_prosody(audio)
        fast = time.perf_counter() - t0
        t0 = time.perf_counter()
        reference_prosody(audio)
        slow = time.perf_counter() - t0
        self.assertLess(fast, slow / 2)


def model_dict(threshold=0.4, bias=-3.0, weights=None):
    n = len(F.FEATURE_NAMES)
    return {"feature_names": list(F.FEATURE_NAMES),
            "weights": weights if weights is not None else [0.0] * n,
            "bias": bias, "scaler_mean": [0.0] * n, "scaler_scale": [1.0] * n,
            "threshold": threshold}


COLD = Snapshot(since_ai=NEVER_SPOKEN, recent_verdicts=(False, False))
STRANGER = Acoustics(seconds=3.0, confidence=0.95)  # voice not verified


def learned_gate(mode="skip", model=None, **config):
    gate = Gate({"mode": mode, "learned": True, "model_path": "/dev/null/none", **config})
    if model is not None:
        gate.load_model(model)
    return gate


class GateSafety(unittest.TestCase):
    def test_learned_skip_when_everything_agrees(self):
        d = learned_gate(model=model_dict()).score(STRANGER, COLD, audio=synthetic_voice(3.0))
        self.assertLess(d.learned_confidence, 0.4)
        self.assertTrue(d.skip)
        self.assertGreater(d.feature_latency, 0.0)

    def test_context_always_transcribes(self):
        gate = learned_gate(model=model_dict())
        for snapshot in (Snapshot(since_ai=NEVER_SPOKEN, awaiting_answer=True),
                         Snapshot(since_ai=NEVER_SPOKEN, confirming=True),
                         Snapshot(since_ai=NEVER_SPOKEN, offer_pending=True),
                         Snapshot(since_ai=3.0)):
            d = gate.score(STRANGER, snapshot, audio=synthetic_voice(3.0))
            self.assertFalse(d.would_skip)
            self.assertTrue(d.reason)
            # skip mode, answer already known: no extraction on the critical path
            self.assertIsNone(d.features)
            self.assertEqual(d.feature_latency, 0.0)

    def test_enrolled_voice_always_transcribes(self):
        wearer = Acoustics(seconds=3.0, confidence=0.95, p_own=0.9,
                           voice_confident=True, is_wearer=True)
        d = learned_gate(model=model_dict()).score(wearer, COLD, audio=synthetic_voice(3.0))
        self.assertFalse(d.would_skip)
        self.assertIn("enrolled voice", d.reason)
        d = learned_gate(model=model_dict(), always_transcribe_wearer=False).score(
            wearer, COLD, audio=synthetic_voice(3.0))
        self.assertTrue(d.would_skip)

    def test_no_threshold_no_skip(self):
        d = learned_gate(model=model_dict(threshold=None)).score(
            STRANGER, COLD, audio=synthetic_voice(3.0))
        self.assertIsNotNone(d.learned_confidence)
        self.assertFalse(d.would_skip)
        self.assertIn("threshold", d.reason)

    def test_missing_or_broken_model_transcribes(self):
        d = learned_gate().score(STRANGER, COLD, audio=synthetic_voice(3.0))
        self.assertFalse(d.would_skip)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "m.json"
            bad.write_text(json.dumps(dict(model_dict(), weights=[1.0])))
            gate = Gate({"mode": "skip", "learned": True, "model_path": str(bad)})
            self.assertIsNone(gate.model_data)
            self.assertTrue(gate.model_error)
            self.assertFalse(gate.score(STRANGER, COLD, audio=synthetic_voice(3.0)).skip)

    def test_doubtful_features_transcribe(self):
        gate = learned_gate(model=model_dict())
        self.assertFalse(gate.score(STRANGER, COLD, audio=np.zeros(48000, np.float32)).skip)
        self.assertFalse(gate.score(STRANGER, COLD, audio=np.zeros(0, np.float32)).skip)
        vec = F.extract_gate_features(synthetic_voice(3.0), 16000, STRANGER, COLD)
        vec[0] = np.nan
        self.assertFalse(gate.score(STRANGER, COLD, features=vec).skip)

    def test_shadow_never_skips_and_logs_numbers(self):
        gate = learned_gate(mode="shadow", model=model_dict(), log_features=True)
        d = gate.score(STRANGER, COLD, audio=synthetic_voice(3.0))
        self.assertTrue(d.would_skip)
        self.assertFalse(d.skip)
        log = d.as_log(include_features=True)
        json.dumps(log, allow_nan=False)
        self.assertEqual(set(log["features"]), set(F.FEATURE_NAMES))
        for key in ("learned_confidence", "would_skip", "signals", "feature_ms", "model_ms"):
            self.assertIn(key, log)

    def test_off_scores_nothing(self):
        d = Gate({"mode": "off"}).score(STRANGER, COLD, audio=synthetic_voice(1.0))
        self.assertFalse(d.skip)
        self.assertIsNone(d.features)


class NoAudioOnDisk(unittest.TestCase):
    def test_runtime_has_no_audio_writer(self):
        self.assertFalse((ROOT / "juno_core" / "audio" / "retention.py").exists())
        for path in (ROOT / "juno_core").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"wave\.open\(([^,]+),\s*['\"]wb['\"]", text):
                self.assertEqual(match.group(1).strip(), "buffer",
                                 f"{path}: audio written somewhere other than memory")
            self.assertNotRegex(text, r"soundfile|sf\.write|\.tofile\(")


def write_wav(path, audio, rate=16000):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


class DataPolicy(unittest.TestCase):
    def test_licences(self):
        self.assertEqual(T.resolve_provenance({"source": "ami"})[1], "CC-BY-4.0")
        self.assertEqual(T.resolve_provenance({"source": "common_voice"})[1], "CC0-1.0")
        with self.assertRaises(T.DataPolicyError):
            T.resolve_provenance({"source": "custom", "license": "CC-BY-NC-4.0"})
        with self.assertRaises(T.DataPolicyError):
            T.resolve_provenance({"source": "ami", "license": "CC0-1.0"})
        with self.assertRaises(T.DataPolicyError):
            T.resolve_provenance({"source": "consented"})
        with self.assertRaises(T.DataPolicyError):
            T.resolve_provenance({"source": "some_podcast"})
        self.assertEqual(T.resolve_provenance(
            {"source": "custom", "license": "CC-BY-4.0"})[1], "CC-BY-4.0")

    def test_trainable(self):
        self.assertTrue(T.trainable({"source": "ami", "license": "CC-BY-4.0"}))
        self.assertFalse(T.trainable({"source": "shadow_log", "license": "none"}, True))
        internal = {"source": "consented", "license": "consent", "consent": "internal"}
        self.assertFalse(T.trainable(internal))
        self.assertTrue(T.trainable(internal, allow_internal=True))
        self.assertFalse(T.provenance_summary([internal])["distributable"])

    def test_build_rows_keeps_numbers_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_wav(tmp / "public.wav", synthetic_voice(2.0))
            write_wav(tmp / "mine.wav", synthetic_voice(2.0, seed=4))
            manifest = [
                {"path": "public.wav", "label": "human_directed", "source": "ami",
                 "speaker": "MEE001", "session": "ES2002a"},
                {"path": "mine.wav", "label": "assistant_directed", "source": "consented",
                 "consent": "release", "speaker": "p1", "session": "s1", "start": "0.2",
                 "end": "1.8", "since_ai": "4", "awaiting_answer": "yes"},
                {"path": "public.wav", "label": "human_directed", "source": "custom",
                 "license": "CC-BY-NC-4.0"},
            ]
            errors = []
            rows = T.build_rows(manifest, base_dir=tmp, delete_source=True,
                                on_error=lambda i, m: errors.append(i))
            self.assertEqual(len(rows), 2)
            self.assertEqual(errors, [3])
            self.assertTrue((tmp / "public.wav").exists())        # public data untouched
            self.assertFalse((tmp / "mine.wav").exists())         # consented clip released
            self.assertAlmostEqual(rows[1]["duration"], 1.6, places=2)
            self.assertEqual(rows[1]["awaiting_answer"], 1.0)
            T.write_rows(tmp / "rows.csv", rows)
            back = T.read_rows([tmp / "rows.csv"])
            self.assertEqual(len(back), 2)
            self.assertEqual(sorted(p.suffix for p in tmp.iterdir()), [".csv", ".wav"])

    def test_context_round_trip(self):
        acoustics = Acoustics(seconds=2.5, confidence=0.93, p_own=0.8,
                              voice_confident=True, is_wearer=False)
        snapshot = Snapshot(since_ai=42.0, awaiting_answer=True, recent_verdicts=(False, False))
        vec = F.extract_gate_features(None, 16000, acoustics, snapshot)
        a2, s2 = T.context_from_features(vec)
        self.assertAlmostEqual(a2.seconds, 2.5, places=4)
        self.assertEqual((a2.voice_confident, a2.is_wearer), (True, False))
        self.assertAlmostEqual(s2.since_ai, 42.0, places=2)
        self.assertTrue(s2.awaiting_answer)
        self.assertEqual(s2.recent_verdicts, (False, False))


def quiet_main(argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return T.main(argv)


def synthetic_rows(n=900, seed=0):
    """Feature rows with a learnable signal -- a stand-in for a real table."""
    rng = np.random.default_rng(seed)
    rows = []
    labels = rng.choice(["assistant_directed", "human_directed", "background_or_media",
                         "uncertain"], size=n, p=[0.3, 0.35, 0.3, 0.05])
    for i, label in enumerate(labels):
        directed = label == "assistant_directed"
        vec = dict.fromkeys(F.FEATURE_NAMES, 0.0)
        vec.update(
            f0_mean=rng.normal(170 if directed else 150, 25),
            f0_std=abs(rng.normal(20, 5)), voiced_ratio=rng.uniform(0.3, 0.8),
            spectral_centroid=rng.normal(1500, 200), spectral_rolloff=rng.normal(2800, 300),
            spectral_tilt_log=rng.normal(4.5, 0.3), zcr=rng.uniform(0.1, 0.3),
            duration=float(np.clip(rng.normal(2.0 if directed else 5.0, 1.0), 0.4, 20)),
            vad_confidence=rng.uniform(0.9, 1.0) if directed else rng.uniform(0.6, 1.0),
            p_own=0.5, voice_confident=0.0, is_wearer=0.5,
            since_ai_log=math.log1p(3600.0),
            ignored_run=float(rng.integers(0, 2 if directed else 5)),
        )
        rows.append({"label": str(label), "source": "ami", "license": "CC-BY-4.0",
                     "consent": "", "speaker": f"spk{i % 60}", "session": f"ses{i % 45}",
                     "room": "", "mic": f"mic{i % 3}", "feature_ms": "3.0",
                     "intent_verdict": "", **vec})
    return rows


class Training(unittest.TestCase):
    def test_split_is_disjoint(self):
        rows = synthetic_rows()
        assignment = T.group_split(rows, (0.6, 0.2, 0.2), ("speaker", "session"))
        self.assertEqual(T.check_disjoint(rows, assignment, ("speaker", "session")), [])
        self.assertEqual(len(set(assignment)), 3)

    def test_threshold_respects_budget(self):
        p = np.array([0.05, 0.1, 0.2, 0.3, 0.6, 0.02, 0.9])
        pos = np.array([True, True, True, True, True, False, False])
        may = np.array([True, True, False, True, True, True, True])
        self.assertEqual(T.choose_threshold(p, may, pos, 0.0), 0.05)
        self.assertEqual(T.choose_threshold(p, may, pos, 0.2), 0.1)
        self.assertEqual(T.choose_threshold(p, may, pos, 0.9), 0.5)   # ceiling

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            T.write_rows(tmp / "rows.csv", synthetic_rows())
            out = tmp / "gate_model.json"
            # By default "uncertain" rows are protected like requests; these
            # are random, so the default must end up refusing to skip them.
            quiet_main(["train", "--rows", str(tmp / "rows.csv"), "--out", str(out),
                        "--group-by", "speaker,session", "--max-false-skip", "0.05"])
            select = json.loads(out.read_text())["metrics"]["select"]["classes"]
            n = select["assistant_directed"]["n"] + select["uncertain"]["n"]
            k = select["assistant_directed"]["skipped"] + select["uncertain"]["skipped"]
            self.assertLessEqual(k / n, 0.05)

            code = quiet_main(["train", "--rows", str(tmp / "rows.csv"), "--out", str(out),
                               "--group-by", "speaker,session", "--max-false-skip", "0.05",
                               "--ignore-uncertain"])
            model = json.loads(out.read_text())
            self.assertIsNotNone(model["threshold"])
            self.assertTrue(model["provenance"]["distributable"])
            self.assertEqual(model["provenance"]["sources"][0]["license"], "CC-BY-4.0")
            select = model["metrics"]["select"]
            self.assertLessEqual(select["false_skip_rate"], 0.05)
            self.assertGreater(select["classes"]["human_directed"]["skip_rate"], 0.0)
            self.assertIn(code, (0, 2))
            gate = Gate({"mode": "shadow", "learned": True, "model_path": str(out)})
            self.assertIsNotNone(gate.model_data)
            self.assertEqual(quiet_main(["evaluate", "--rows", str(tmp / "rows.csv"),
                                     "--model", str(out)]), 0)

    def test_refuses_untrainable_rows(self):
        rows = synthetic_rows(300)
        for r in rows:
            r.update(source="shadow_log", license="none")
        with tempfile.TemporaryDirectory() as tmp:
            T.write_rows(Path(tmp) / "rows.csv", rows)
            with self.assertRaises(SystemExit):
                quiet_main(["train", "--rows", str(Path(tmp) / "rows.csv"),
                        "--out", str(Path(tmp) / "m.json")])

    def test_shadow_log_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "events.jsonl"
            feats = {n: 0.0 for n in F.FEATURE_NAMES}
            lines = [
                {"event": "gate_scored", "turn": "a", "would_skip": True, "latency_ms": 2.0,
                 "feature_ms": 1.5, "features": feats},
                {"event": "intent_accepted", "turn": "a"},
                {"event": "gate_scored", "turn": "b", "would_skip": True, "latency_ms": 1.0},
                {"event": "intent_ignored", "turn": "b"},
                {"event": "gate_scored", "turn": "c", "would_skip": False, "latency_ms": 1.0},
            ]
            log.write_text("\n".join(json.dumps(x) for x in lines))
            turns = T.read_shadow_log(log)
            rep = T.shadow_report(turns)
            self.assertEqual(rep["scored"], 3)
            self.assertEqual(rep["would_skip"], 2)
            self.assertEqual([d["turn"] for d in rep["would_skip_but_accepted"]], ["a"])
            rows = T.shadow_rows(turns)
            self.assertEqual(len(rows), 1)
            self.assertFalse(T.trainable(rows[0], allow_internal=True))


if __name__ == "__main__":
    unittest.main()
