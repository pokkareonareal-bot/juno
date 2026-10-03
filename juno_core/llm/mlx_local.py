"""A local model on Apple Silicon via mlx-lm. No API key, no server, nothing leaves the machine.

    llm:
      provider: mlx
      model: mlx-community/Qwen3.5-4B-4bit     # the default; any mlx-lm chat model works

The weights download once from Hugging Face and are cached. Like Ollama, this
makes a fully offline pipeline possible; unlike Ollama there is nothing to
install or run beside Juno. It is also the model System One's teacher uses
(juno_core/slu/judge.py), so the two share one download.

MLX keeps its streams per thread, and the pipeline may call this from more
than one, so the model lives on a single thread of its own and every call is
handed to it.

Needs:  pip install mlx-lm      (Apple Silicon)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from . import BaseModel, ModelError, ModelResponse

DEFAULT_MODEL = "mlx-community/Qwen3.5-4B-4bit"


class MLXModel(BaseModel):
    name = "mlx"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        import importlib.util

        if importlib.util.find_spec("mlx_lm") is None:
            raise ModelError("llm.provider is 'mlx' but mlx-lm isn't installed. "
                             "Run: pip install mlx-lm")
        self.model = model
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-llm")
        self._loaded = None
        # Load now, in the background: the adjudicator's timeout is a few
        # seconds, and the first load takes about that long on its own.
        self._thread.submit(self._load)

    def _load(self):
        if self._loaded is None:
            from mlx_lm import load

            self._loaded = load(self.model)
        return self._loaded

    def _generate(self, messages, max_tokens, temperature) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = self._load()
        try:
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                   tokenize=False, enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                   tokenize=False)
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                        sampler=make_sampler(temp=max(0.0, float(temperature))))

    def generate(self, messages, *, max_tokens=512, temperature=0.7,
                 timeout=30.0) -> ModelResponse:
        future = self._thread.submit(self._generate, messages, max_tokens, temperature)
        try:
            text = future.result(timeout=timeout)
        except FutureTimeout as exc:
            raise ModelError(f"the local model took longer than {timeout:.0f} s") from exc
        return ModelResponse(text=text.strip())
