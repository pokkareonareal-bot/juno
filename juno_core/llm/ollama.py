"""A local model via Ollama. No API key, no cloud, nothing leaves the machine.

Needs Ollama running (https://ollama.com) and a model pulled, e.g.:
    ollama pull llama3.2

If you want the whole pipeline to run with nothing sent over the network at
all, this is the adapter for that -- pair it with a local STT engine (see
README, "Choosing a speech-to-text engine") for a fully offline setup.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import BaseModel, ModelError, ModelResponse


class OllamaModel(BaseModel):
    name = "ollama"

    def __init__(self, model: str = "llama3.2",
                 base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, messages, *, max_tokens=512, temperature=0.7,
                 timeout=30.0) -> ModelResponse:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ModelError(
                f"couldn't reach Ollama at {self.base_url} ({exc.reason}). "
                f"Is it running? (`ollama serve`, or just open the app)"
            ) from exc

        return ModelResponse(
            text=result.get("message", {}).get("content", ""),
            finish_reason="stop" if result.get("done") else "length",
            raw=result,
        )
