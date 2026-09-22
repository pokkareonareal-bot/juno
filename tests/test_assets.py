"""Model downloads: pinned, checksummed, cached, and failing loudly."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from juno_core import assets

PAYLOAD = b"not really an onnx model" * 100


def fake_asset(payload=PAYLOAD):
    return dataclasses.replace(assets.ASSETS["silero_vad"],
                               sha256=hashlib.sha256(payload).hexdigest(), size=len(payload))


class Download(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"JUNO_MODELS_DIR": self.tmp.name})
        self.env.start()
        self.legacy = mock.patch.object(assets, "LEGACY_DIR", Path(self.tmp.name) / "none")
        self.legacy.start()

    def tearDown(self):
        self.legacy.stop()
        self.env.stop()
        self.tmp.cleanup()

    def serve(self, payload=PAYLOAD, sha_of=PAYLOAD):
        return (mock.patch.dict(assets.ASSETS, {"silero_vad": fake_asset(sha_of)}),
                mock.patch("urllib.request.urlopen", return_value=io.BytesIO(payload)))

    def test_downloads_verifies_and_caches(self):
        registry, net = self.serve()
        with registry, net as urlopen:
            path = assets.ensure("silero_vad", quiet=True)
            self.assertEqual(path, Path(self.tmp.name) / "silero_vad.onnx")
            self.assertEqual(path.read_bytes(), PAYLOAD)
            assets.ensure("silero_vad", quiet=True)          # cached: no second fetch
            self.assertEqual(urlopen.call_count, 1)
            self.assertEqual(os.listdir(self.tmp.name), ["silero_vad.onnx"])

    def test_checksum_mismatch_keeps_nothing(self):
        registry, net = self.serve(payload=b"tampered", sha_of=PAYLOAD)
        with registry, net:
            with self.assertRaises(assets.AssetError):
                assets.ensure("silero_vad", quiet=True)
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_offline_says_what_to_do(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("no network")):
            with self.assertRaises(assets.AssetError) as caught:
                assets.ensure("ecapa", quiet=True)
        self.assertIn("python -m juno_core.assets", str(caught.exception))

    def test_existing_explicit_path_is_trusted(self):
        mine = Path(self.tmp.name) / "mine.onnx"
        mine.write_bytes(b"x")
        with mock.patch("urllib.request.urlopen") as urlopen:
            self.assertEqual(assets.ensure("silero_vad", mine, quiet=True), mine)
            urlopen.assert_not_called()

    def test_pins_are_exact(self):
        for asset in assets.ASSETS.values():
            self.assertNotIn("/master/", asset.url)
            self.assertNotIn("/main/", asset.url)
            self.assertEqual(len(asset.sha256), 64)

    def test_vad_falls_back_visibly_when_offline(self):
        from juno_core.audio.vad import EnergyVAD, build_vad

        events = []
        observer = mock.Mock(emit=lambda *a, **k: events.append((a, k)))
        with mock.patch("urllib.request.urlopen", side_effect=OSError("no network")):
            backend = build_vad({"backend": "silero"}, 16000, observer)
        self.assertIsInstance(backend, EnergyVAD)
        self.assertEqual(events[0][0][1], "backend_fallback")

    def test_voiceprint_downloads_only_its_own_default(self):
        from juno_core.audio.voiceprint import VoicePrint

        with mock.patch("urllib.request.urlopen") as urlopen:
            custom = VoicePrint({"model": str(Path(self.tmp.name) / "absent.onnx")})
            self.assertFalse(custom.available)
            urlopen.assert_not_called()
        with mock.patch("urllib.request.urlopen", side_effect=OSError("offline")):
            self.assertFalse(VoicePrint({}).available)


if __name__ == "__main__":
    unittest.main()
