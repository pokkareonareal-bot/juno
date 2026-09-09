"""Google Gemini, via a plain HTTPS call.

Needs GOOGLE_API_KEY (see .env.example). Get one at
https://aistudio.google.com/apikey

Gemini's API uses "model" instead of "assistant" for the model's own turns,
and takes the system prompt as its own field, so messages are translated on
the way in.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from . import BaseModel, ModelError, ModelResponse

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiModel(BaseModel):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ModelError(
                "llm.provider is 'gemini' but GOOGLE_API_KEY isn't set. "
                "Put it in .env (see .env.example) or export it."
            )

    def generate(self, messages, *, max_tokens=512, temperature=0.7,
                 timeout=30.0) -> ModelResponse:
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages if m.get("role") in ("user", "assistant")
        ]
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        body = json.dumps(payload).encode("utf-8")
        url = ENDPOINT.format(model=self.model) + f"?key={self.api_key}"
        request = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ModelError(f"Gemini returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ModelError(f"couldn't reach Gemini: {exc.reason}") from exc

        try:
            candidate = result["candidates"][0]
            text = "".join(p.get("text", "") for p in candidate["content"]["parts"])
            finish = candidate.get("finishReason", "STOP")
        except (KeyError, IndexError) as exc:
            raise ModelError(f"unexpected Gemini response shape: {result}") from exc
        usage = result.get("usageMetadata", {})
        return ModelResponse(
            text=text,
            finish_reason=str(finish).lower(),
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
            raw=result,
        )
