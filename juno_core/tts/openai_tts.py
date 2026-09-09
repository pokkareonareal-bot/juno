"""OpenAI's text-to-speech API. Cross-platform, needs an API key and a
speaker.

Needs OPENAI_API_KEY (see .env.example) and: pip install requests
Playback needs one of: afplay (macOS, built in), ffplay (ffmpeg), or you can
swap _play() for however you already play audio.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from . import TTSEngine

ENDPOINT = "https://api.openai.com/v1/audio/speech"


class OpenAITTS(TTSEngine):
    name = "openai"

    def __init__(self, voice: str = "alloy", api_key: str | None = None) -> None:
        self.voice = voice
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "tts.provider is 'openai' but OPENAI_API_KEY isn't set. "
                "Put it in .env (see .env.example) or export it."
            )

    def speak(self, text: str) -> None:
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "tts.provider is 'openai' but requests isn't installed. "
                "Run: pip install requests"
            ) from exc

        response = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": "tts-1", "voice": self.voice, "input": text},
            timeout=30,
        )
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
            handle.write(response.content)
            path = handle.name
        self._play(path)

    def _play(self, path: str) -> None:
        for player in ("afplay", "ffplay"):
            if shutil.which(player):
                args = [player, path] if player == "afplay" else \
                    [player, "-nodisp", "-autoexit", "-loglevel", "quiet", path]
                subprocess.run(args, check=False)
                return
        print(f"(saved the reply's audio to {path} -- no player found to "
              f"play it automatically; install ffmpeg, or open the file)")
