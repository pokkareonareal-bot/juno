"""The shape a text-to-speech engine needs to have, if you want one at all.

Entirely optional. Nothing in juno_core/pipeline.py calls this -- by default
an accepted utterance's reply is just returned as text (run.py prints it).
If you want it spoken, implement `TTSEngine` and pass it to run.py, or call
one of the two bundled adapters below.

    class MyTTS(TTSEngine):
        def speak(self, text: str) -> None:
            ...  # however you turn `text` into sound
"""

from __future__ import annotations


class TTSEngine:
    name = "base"

    def speak(self, text: str) -> None:
        raise NotImplementedError


def build_tts(config):
    """Construct the configured `tts.provider`, or None if unset (silent)."""
    provider = str(config.get("provider", "") or "").lower()
    if not provider or provider == "none":
        return None
    if provider == "say":
        from .system_say import SayTTS
        return SayTTS(voice=config.get("voice"))
    if provider == "openai":
        from .openai_tts import OpenAITTS
        return OpenAITTS(voice=config.get("voice", "alloy"))
    raise ValueError(
        f"unknown tts.provider {provider!r} -- expected 'say' or 'openai', "
        f"or leave it unset to just print replies"
    )
