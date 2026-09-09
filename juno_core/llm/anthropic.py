"""Anthropic (Claude), via a plain HTTPS call.

Needs ANTHROPIC_API_KEY (see .env.example). Get one at
https://console.anthropic.com/settings/keys

Anthropic's Messages API takes the system prompt as its own top-level field
rather than as a "system"-role message, so that one message (if present) is
pulled out of `messages` before the rest goes across unchanged.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from . import BaseModel, ModelError, ModelResponse

ENDPOINT = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicModel(BaseModel):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ModelError(
                "llm.provider is 'anthropic' but ANTHROPIC_API_KEY isn't set. "
                "Put it in .env (see .env.example) or export it."
            )

    def generate(self, messages, *, max_tokens=512, temperature=0.7,
                 timeout=30.0) -> ModelResponse:
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        turns = [m for m in messages if m.get("role") in ("user", "assistant")]
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": turns,
        }
        if system:
            payload["system"] = system
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            ENDPOINT, data=body, method="POST",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ModelError(f"Anthropic returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ModelError(f"couldn't reach Anthropic: {exc.reason}") from exc

        text = "".join(
            block.get("text", "") for block in result.get("content", [])
            if block.get("type") == "text"
        )
        usage = result.get("usage", {})
        return ModelResponse(
            text=text,
            finish_reason=result.get("stop_reason", "stop") or "stop",
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            raw=result,
        )
