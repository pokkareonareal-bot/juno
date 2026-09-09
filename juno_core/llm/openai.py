"""OpenAI, via a plain HTTPS call -- no SDK, so this package stays light.

Needs OPENAI_API_KEY (see .env.example). Get one at
https://platform.openai.com/api-keys
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from . import BaseModel, ModelError, ModelResponse

ENDPOINT = "https://api.openai.com/v1/chat/completions"


class OpenAIModel(BaseModel):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ModelError(
                "llm.provider is 'openai' but OPENAI_API_KEY isn't set. "
                "Put it in .env (see .env.example) or export it."
            )

    def generate(self, messages, *, max_tokens=512, temperature=0.7,
                 timeout=30.0) -> ModelResponse:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")
        request = urllib.request.Request(
            ENDPOINT, data=body, method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ModelError(f"OpenAI returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ModelError(f"couldn't reach OpenAI: {exc.reason}") from exc

        choice = payload["choices"][0]
        usage = payload.get("usage", {})
        return ModelResponse(
            text=choice["message"].get("content", "") or "",
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw=payload,
        )
