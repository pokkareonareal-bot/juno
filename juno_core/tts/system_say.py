"""macOS's built-in `say` command. Zero setup, zero API key, macOS only.

The fastest way to hear an actual answer out loud with nothing to configure.
Not recommended beyond trying the pipeline out -- it's a system utility, not
a production voice.
"""

from __future__ import annotations

import shutil
import subprocess

from . import TTSEngine


class SayTTS(TTSEngine):
    name = "say"

    def __init__(self, voice: str | None = None) -> None:
        if shutil.which("say") is None:
            raise RuntimeError("tts.provider is 'say' but this isn't macOS "
                               "(no `say` command found)")
        self.voice = voice

    def speak(self, text: str) -> None:
        command = ["say"]
        if self.voice:
            command += ["-v", self.voice]
        command.append(text)
        subprocess.run(command, check=False)
