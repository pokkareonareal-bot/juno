#!/usr/bin/env python3
"""Example: connecting your own agent to the pipeline.

    python examples/connect_an_agent.py     (needs config.yaml + .env, same as run.py)

juno_core.pipeline.JunoPipeline calls exactly one function once it has
decided an utterance was meant for you:

    on_accept(text: str, decision: IntentDecision, context: ConversationContext) -> str | None

Everything upstream of that call is the addressee-detection technology this
package ships -- deciding WHETHER you were being spoken to, with no wake
word. Everything downstream of it is yours: call a language model once, run
a tool-calling loop, call out to LangChain / AutoGen / a framework you
already have, or -- as below -- do something on the device directly and only
reach for a language model when nothing more specific matches.

This example adds two tiny actions (say the time, open a site in the default
browser) and falls back to whatever `llm.provider` is configured in
config.yaml for everything else. It's deliberately small -- the point is the
shape of the seam, not a tool-calling framework. Wire in a real one the same
way: replace the body of `my_agent` and leave the rest of this file alone.
"""

from __future__ import annotations

import re
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from juno_core.intelligence.context import ConversationContext  # noqa: E402
from juno_core.intelligence.intent import IntentDecision  # noqa: E402

_TIME = re.compile(r"what(?:'s| is) the time|what time is it", re.IGNORECASE)
_OPEN = re.compile(r"\bopen\s+([a-z0-9.-]+\.[a-z]{2,})\b", re.IGNORECASE)


def my_agent(text: str, decision: IntentDecision, context: ConversationContext,
             *, model) -> str:
    """The whole seam, in one function.

    `text` -- what Juno decided was meant for it.
    `decision` -- how confident it was, and why (see IntentDecision).
    `context` -- the conversation so far; `context.messages()` renders it in
      the [{"role": ..., "content": ...}] shape every chat API expects.
    `model` -- whatever `llm.provider` is configured (juno_core.llm). Swap
      this parameter for your own agent's entry point and this function is
      the entire integration.
    """
    if _TIME.search(text):
        return f"It's {datetime.now().strftime('%H:%M')}."

    match = _OPEN.search(text)
    if match:
        site = match.group(1)
        webbrowser.open(f"https://{site}")
        return f"Opening {site}."

    # Not one of ours -- fall back to the configured language model, same as
    # JunoPipeline's own default would.
    messages = [
        {"role": "system", "content": "You are a spoken voice assistant. "
                                       "Answer in one short sentence -- this "
                                       "is read aloud, not read on a screen."},
        *context.messages(turns=6),
    ]
    return model.complete(messages, max_tokens=150)


def main() -> None:
    from juno_core.config import load_config
    from juno_core.llm import build_model
    from juno_core.observability import Observer
    from juno_core.pipeline import JunoPipeline
    from juno_core.stt import build_stt

    config = load_config()
    stt = build_stt(config.get("stt") or {})
    model = build_model(config.get("llm") or {})
    observer = Observer(log_path=ROOT / "logs" / "events.jsonl")

    pipeline = JunoPipeline(
        config, stt=stt, model=model, observer=observer,
        on_accept=lambda text, decision, context: my_agent(
            text, decision, context, model=model),
    )

    print("Connected a two-action example agent.")
    print('Try: "what time is it", "open wikipedia.org", or anything else.\n')
    try:
        pipeline.run_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
