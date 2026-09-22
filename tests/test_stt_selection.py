"""STT provider selection and the mlx-whisper adapter, with a stand-in module
(no model download, runs on any machine)."""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import numpy as np

from juno_core import stt
from juno_core.stt import mlx_whisper as M


def fake_mlx(result):
    module = types.ModuleType("mlx_whisper")
    module.calls = []

    def transcribe(audio, **kwargs):
        module.calls.append((audio, kwargs))
        return result

    module.transcribe = transcribe
    return module


RESULT = {
    "text": " What's the weather tomorrow? ",
    "language": "en",
    "segments": [
        {"avg_logprob": -0.2, "no_speech_prob": 0.01},
        {"avg_logprob": -1.4, "no_speech_prob": 0.05},
    ],
}


class MLXAdapter(unittest.TestCase):
    def build(self, result=RESULT, model="small.en"):
        module = fake_mlx(result)
        with mock.patch.dict(sys.modules, {"mlx_whisper": module}), \
                mock.patch.object(M, "apple_silicon", return_value=True):
            return M.MLXWhisperSTT(model=model), module

    def test_maps_result(self):
        engine, module = self.build()
        t = engine.transcribe(np.zeros(32000, dtype=np.float64))
        self.assertEqual(t.text, "What's the weather tomorrow?")
        self.assertAlmostEqual(t.avg_logprob, -0.8)
        self.assertTrue(t.reliable)
        self.assertFalse(t.tail_reliable)          # the last segment is the doubtful one
        self.assertAlmostEqual(t.duration, 2.0)
        audio, kwargs = module.calls[0]
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(kwargs["path_or_hf_repo"], "mlx-community/whisper-small.en-mlx")
        self.assertEqual(kwargs["language"], "en")
        self.assertFalse(kwargs["condition_on_previous_text"])

    def test_empty_and_passthrough_model(self):
        engine, _ = self.build({"text": "", "segments": []}, model="/models/my-whisper")
        self.assertEqual(engine.model, "/models/my-whisper")
        t = engine.transcribe(np.zeros(1600, dtype=np.float32))
        self.assertFalse(t)
        self.assertIsNone(t.avg_logprob)

    def test_refuses_off_apple_silicon(self):
        with mock.patch.object(M, "apple_silicon", return_value=False):
            with self.assertRaises(RuntimeError):
                M.MLXWhisperSTT()


class AutoSelection(unittest.TestCase):
    def test_apple_silicon_with_mlx(self):
        with mock.patch.object(M, "apple_silicon", return_value=True), \
                mock.patch("importlib.util.find_spec", return_value=object()):
            self.assertEqual(stt._local_provider(), "mlx_whisper")

    def test_apple_silicon_without_mlx(self):
        with mock.patch.object(M, "apple_silicon", return_value=True), \
                mock.patch("importlib.util.find_spec", return_value=None):
            self.assertEqual(stt._local_provider(), "faster_whisper")

    def test_elsewhere(self):
        with mock.patch.object(M, "apple_silicon", return_value=False):
            self.assertEqual(stt._local_provider(), "faster_whisper")

    def test_build_auto_constructs_mlx(self):
        module = fake_mlx(RESULT)
        with mock.patch.dict(sys.modules, {"mlx_whisper": module}), \
                mock.patch.object(M, "apple_silicon", return_value=True), \
                mock.patch.object(stt, "_local_provider", return_value="mlx_whisper"):
            engine = stt.build_stt({"provider": "auto", "model": "base.en"})
        self.assertIsInstance(engine, M.MLXWhisperSTT)
        self.assertEqual(engine.model, "mlx-community/whisper-base.en-mlx")

    def test_unknown_provider(self):
        with self.assertRaises(ValueError):
            stt.build_stt({"provider": "nope"})


if __name__ == "__main__":
    unittest.main()
