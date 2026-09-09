"""The one thing the pipeline needs from a language model: text in, text out.

This interface is used in exactly two places once an utterance is accepted:

  1. Optionally, *before* acceptance -- the intent engine's LLM adjudicator
     consults it for a second opinion, but only on the utterances its fast
     heuristic scoring finds genuinely ambiguous (on the shipped eval set,
     that's about one utterance in five; the rest never touch the network).
  2. *After* acceptance -- your own code calls it (or calls your own agent,
     which probably calls something like it) to actually answer.

Nothing downstream of ``LanguageModel`` cares which provider is behind it.
Bring whichever key you have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ModelResponse:
    text: str
    finish_reason: str = "stop"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: Any = None


class ModelError(RuntimeError):
    """The provider was reachable but refused or failed the request."""


class LanguageModel(Protocol):
    """Implement `generate`. `complete` is free once you have.

    Every adapter in this package (openai.py, anthropic.py, gemini.py,
    ollama.py) implements exactly this, so switching providers is a one-line
    config change -- see config.example.yaml's `llm:` section.
    """

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        timeout: float = 30.0,
    ) -> ModelResponse:
        ...

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        timeout: float = 30.0,
    ) -> str:
        ...


class BaseModel:
    """Shared plumbing for the bundled adapters: `complete` from `generate`."""

    name = "base"

    def generate(self, messages, *, max_tokens=512, temperature=0.7,
                 timeout=30.0) -> ModelResponse:
        raise NotImplementedError

    def complete(self, messages, *, max_tokens=512, temperature=0.7,
                 timeout=30.0) -> str:
        return self.generate(messages, max_tokens=max_tokens,
                              temperature=temperature, timeout=timeout).text


class EchoModel(BaseModel):
    """No API key, no network, no dependency. Repeats the question back.

    The default when nothing is configured, so `python run.py` produces a
    working (if unhelpful) end-to-end pipeline the moment you clone the repo
    -- you can hear the addressee detection working before you've decided
    which real model to point it at.
    """

    name = "echo"

    def generate(self, messages, *, max_tokens=512, temperature=0.7,
                 timeout=30.0) -> ModelResponse:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        text = (
            f"(no language model configured -- you said: {last_user!r}. "
            f"Set llm.provider in config.yaml and an API key in .env to get "
            f"a real answer here.)"
        )
        return ModelResponse(text=text)


def build_model(config, api_key: str | None = None) -> BaseModel:
    """Construct the configured `llm.provider`, or EchoModel if unset.

    `config` is the `llm:` section of config.yaml (a Section or a dict-like
    with `.get`). `api_key`, if not passed explicitly, is read from the
    environment variable each adapter documents (see .env.example).
    """
    provider = str(config.get("provider", "echo") or "echo").lower()
    model = config.get("model")
    if provider in ("echo", "none", ""):
        return EchoModel()
    if provider == "openai":
        from .openai import OpenAIModel
        return OpenAIModel(model=model or "gpt-4o-mini", api_key=api_key)
    if provider == "anthropic":
        from .anthropic import AnthropicModel
        return AnthropicModel(model=model or "claude-sonnet-5", api_key=api_key)
    if provider == "gemini" or provider == "google":
        from .gemini import GeminiModel
        return GeminiModel(model=model or "gemini-2.5-flash", api_key=api_key)
    if provider == "ollama":
        from .ollama import OllamaModel
        return OllamaModel(model=model or "llama3.2",
                            base_url=config.get("base_url", "http://localhost:11434"))
    raise ModelError(
        f"unknown llm.provider {provider!r} -- expected one of: "
        f"echo, openai, anthropic, gemini, ollama"
    )
