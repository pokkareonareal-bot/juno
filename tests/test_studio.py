"""Juno Studio: the server's guards, and the try / collect / evaluate flows.

Headless: synthetic audio stands in for the microphone, a fake teacher for
speech-to-text, and the numpy encoders for the MLX ones.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from juno_core.config import Section
from juno_core.slu.schema import ADDRESSEES, CORE_SCHEMA
from juno_core.slu.teacher import TeacherLabel

from test_pre_stt_gate import synthetic_voice


class FakeTeacher:
    def label(self, audio, rate=16000):
        return TeacherLabel(transcript="juno what time is it", reliable=True, accepted=True,
                            confidence=0.9, method="heuristic",
                            addressed={a: 1 / 3 for a in ADDRESSEES}, intent="time.now", intent_p=1.0)


CONFIG = Section({
    "audio": {"sample_rate": 16000, "channels": 1, "frame_ms": 32},
    "vad": {"backend": "energy", "threshold": 0.5, "min_speech_ms": 200, "min_silence_ms": 400,
            "min_segment_ms": 350, "max_segment_ms": 20000},
})


def wait_for(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


class StudioBase(unittest.TestCase):
    def setUp(self):
        from juno_core.slu.studio.engine import Studio
        from juno_core.slu.studio.server import make_server

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        clips = [synthetic_voice(1.2, seed=1), synthetic_voice(1.6, f0=200, seed=2)]
        self.studio = Studio(CONFIG, root=self.root, encoders=("logmel", "gate"),
                             simulate=clips, realtime=False, teacher=FakeTeacher())
        self.studio.load(background=False)
        self.server = make_server(self.studio, port=0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.studio.stop_listening()
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def call(self, path, body=None, token=True, host=None, ctype="application/json"):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data=None if body is None else json.dumps(body).encode(),
                                     method="GET" if body is None else "POST")
        if token:
            req.add_header("X-Studio-Token", self.server.token)
        if host:
            req.add_header("Host", host)
        if body is not None:
            req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
                return resp.status, (json.loads(raw) if resp.headers.get_content_type() == "application/json" else raw)
        except urllib.error.HTTPError as err:
            raw = err.read()
            try:
                return err.code, json.loads(raw)
            except ValueError:
                return err.code, raw


class Guards(StudioBase):
    def test_token_host_and_content_type(self):
        status, page = self.call("/", token=False)
        self.assertEqual(status, 200)
        self.assertIn(self.server.token.encode(), page)          # handed to its own page only
        self.assertEqual(self.call("/api/status", token=False)[0], 403)
        self.assertEqual(self.call("/api/status", host="evil.example")[0], 421)
        self.assertEqual(self.call("/api/try/start", {}, token=False)[0], 403)
        self.assertEqual(self.call("/api/try/start", {}, ctype="text/plain")[0], 415)
        status, data = self.call("/api/status")
        self.assertEqual(status, 200)
        self.assertTrue(data["result"]["ready"])
        self.assertEqual(self.call("/static/../server.py", token=False)[0], 404)


class Collecting(StudioBase):
    def test_consented_session_keeps_vectors_not_audio(self):
        status, data = self.call("/api/session/start", {"speakers": ["p1"], "room": "lab",
                                                         "consent": "internal", "confirmed": False})
        self.assertEqual(status, 400)
        self.assertIn("agree", data["error"])
        status, data = self.call("/api/session/start", {"speakers": ["p1"], "room": "lab",
                                                         "consent": "internal", "confirmed": True})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["result"]["steps"][0]["speaker"], "p1")

        self.assertEqual(self.call("/api/session/free", {"label": "assistant_directed",
                                                         "intent": "time.now", "speaker": "p1"})[0], 200)
        self.assertTrue(wait_for(lambda: len(self.studio.session.rows) >= 3))
        self.call("/api/session/stop", {})
        rows = self.studio.session.rows
        first = rows[0]["id"]
        self.assertEqual(self.call("/api/session/discard", {"id": first})[0], 200)
        self.assertNotIn(first, [r["id"] for r in self.studio.session.rows])

        status, data = self.call("/api/session/finish", {})
        self.assertEqual(status, 200, data)
        out = Path(data["result"]["path"])
        names = sorted(p.name for p in out.iterdir())
        self.assertIn("manifest.jsonl", names)
        self.assertIn("logmel__64__stats.npz", names)
        self.assertIn("gate__20.npz", names)
        self.assertFalse([n for n in names if n.endswith(".wav")])
        manifest = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
        with np.load(out / "gate__20.npz") as z:
            self.assertEqual(list(z["ids"]), [r["id"] for r in manifest])
        for r in manifest:
            self.assertEqual((r["g_addressed"], r["g_intent"], r["consent"], r["source"]),
                             ("assistant_directed", "time.now", "internal", "consented"))
        sessions = self.call("/api/sessions")[1]["result"]
        self.assertEqual(sessions[0]["rows"], len(manifest))


class Trying(StudioBase):
    def test_compare_label_and_save(self):
        q = self.studio.events.subscribe()
        self.assertEqual(self.call("/api/try/start", {})[0], 200)
        utterances = []

        def got_one():
            while not q.empty():
                kind, data = q.get_nowait()
                if kind == "utterance":
                    utterances.append(data)
            return bool(utterances)

        self.assertTrue(wait_for(got_one))
        self.call("/api/listen/stop", {})
        u = utterances[0]
        self.assertEqual(u["agreement"], "no_model")                 # no model in this root
        self.assertEqual(u["system_two"]["intent"], "time.now")
        self.assertEqual(self.call("/api/try/label", {"id": u["id"], "addressed": "human_directed"})[0], 200)
        # Labelling again replaces, rather than duplicates.
        self.call("/api/try/label", {"id": u["id"], "addressed": "assistant_directed", "intent": "time.now"})
        self.assertEqual(len(self.studio.labelled), 1)
        status, data = self.call("/api/try/save", {"consent": "release", "room": "lab", "speaker": "p1"})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["result"]["rows"], 1)
        self.assertEqual(self.studio.labelled, [])

    def test_evaluate_a_model_on_a_session(self):
        from juno_core.slu.student import build_targets, fit

        # A session to evaluate on...
        self.call("/api/session/start", {"speakers": ["p1"], "room": "lab", "consent": "internal",
                                          "confirmed": True})
        self.call("/api/session/free", {"label": "assistant_directed", "intent": "time.now", "speaker": "p1"})
        self.assertTrue(wait_for(lambda: len(self.studio.session.rows) >= 2))
        self.call("/api/session/stop", {})
        session = self.call("/api/session/finish", {})[1]["result"]["path"]
        # ...and a (meaningless) model on the same encoder.
        rows = [{"g_addressed": a, "g_intent": "time.now" if a == "assistant_directed" else None}
                for a in ADDRESSEES * 10]
        X = np.random.default_rng(0).standard_normal((len(rows), 20)).astype(np.float32)
        T = build_targets(rows, CORE_SCHEMA, "gold")
        model = fit(X, T, X, T, schema=CORE_SCHEMA, encoder_spec="gate:20", hidden=8, epochs=3)
        (self.root / "models").mkdir()
        model.save(self.root / "models" / "toy.npz")
        status, data = self.call("/api/evaluate", {"model": "models/toy.npz", "sessions": [session]})
        self.assertEqual(status, 200, data)
        report = data["result"]["report"]
        self.assertGreaterEqual(report["rows"], 2)
        self.assertIn("false_activation", report)
        # A session recorded without the model's encoder is refused with a reason.
        status, data = self.call("/api/evaluate", {"model": "models/toy.npz", "sessions": []})
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
