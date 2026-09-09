"""Cloud speech-to-text via OpenAI's Whisper API.

RECOMMENDED if you want the simplest possible setup and don't mind audio
leaving the machine. No local model download, no GPU consideration.

Needs OPENAI_API_KEY (see .env.example) and: pip install requests
"""

from __future__ import annotations

import io
import time
import wave

import numpy as np

from . import STTEngine, Transcript

ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"


class OpenAIWhisperSTT(STTEngine):
    name = "openai-whisper"

    def __init__(self, model: str = "whisper-1", api_key: str | None = None) -> None:
        import os
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "stt.provider is 'openai_whisper' but OPENAI_API_KEY isn't "
                "set. Put it in .env (see .env.example) or export it."
            )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "stt.provider is 'openai_whisper' but requests isn't "
                "installed. Run: pip install requests"
            ) from exc

        started = time.monotonic()
        buffer = io.BytesIO()
        pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes((pcm * 32767.0).astype("<i2").tobytes())
        buffer.seek(0)

        response = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"file": ("utterance.wav", buffer, "audio/wav")},
            data={"model": self.model},
            timeout=30,
        )
        response.raise_for_status()
        text = response.json().get("text", "").strip()
        return Transcript(
            text=text,
            duration=len(audio) / float(sample_rate),
            latency=time.monotonic() - started,
        )
