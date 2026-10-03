#!/usr/bin/env python3
"""Example: connecting your own agent to the pipeline.

    python examples/connect_an_agent.py     (needs config.yaml + .env, same as run.py)

There are two seams. Use whichever fits your agent.

1. TEXT -- the original one. Once Juno decides an utterance was meant for
   you, it calls

       on_accept(text: str, decision: IntentDecision, context) -> str | None

2. TYPED -- the System One one. Your agent DECLARES what it can do (a schema
   of intents with typed slots), and Juno calls

       on_decision(decision: Decision, context) -> str | None

   for every utterance meant for you, answered in that schema:

       {"route": "act", "source": "system_one",
        "addressed": {"value": "assistant_directed", "p": 0.97},
        "intent": {"value": "lights.off", "p": 0.93}, "slots": {},
        "transcript": null}

   ``source`` says how it was decided. "system_one" means from the audio
   alone, with no transcription (``transcript`` is null); "system_two" means
   speech-to-text ran and ``transcript`` holds the words. The shape is the
   same either way, so your agent has one code path.

   An agent's own intents are understood by System Two straight away (from
   their examples, or by your language model if one is configured). System
   One only knows the intents it was trained on, and an utterance it has no
   class for could land in the wrong one -- so given a schema with intents
   its model lacks, it switches itself off (and says why) until it is
   retrained on the extended schema (README, "System One"). System Two keeps
   answering in your schema meanwhile.

This example declares two intents of its own on top of the core ones, and
handles a few things locally. Everything else goes to whatever llm.provider
is configured.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from juno_core.intelligence.context import ConversationContext  # noqa: E402
from juno_core.slu.schema import CORE_SCHEMA, Decision, Schema  # noqa: E402

# What this agent can do, beyond the core intents. Write it in JSON if you
# prefer -- `python -m juno_core.slu schema --out my_schema.json` prints the
# core one to start from.
MY_SCHEMA = CORE_SCHEMA.extend(Schema.from_dict({
    "name": "example-agent",
    "intents": [
        {"name": "lights.off", "description": "turn the lights off",
         "examples": ["lights off", "turn off the lights", "turn the lights off"],
         "min_confidence": 0.95},
        {"name": "lights.set", "description": "set the lights to a level",
         "examples": ["set the lights to {level}", "lights {level}", "make it {level}"],
         "slots": [{"name": "level", "type": "enum", "values": ["dim", "medium", "bright"]}]},
    ],
}))


def my_agent(decision: Decision, context: ConversationContext, *, model) -> str | None:
    """The whole typed seam, in one function."""
    intent = decision.intent.value if decision.intent else None
    slots = {name: f.value for name, f in decision.slots.items()}

    if intent == "time.now":
        return f"It's {datetime.now().strftime('%H:%M')}."
    if intent == "lights.off":
        return "Lights off."                      # call your smart-home API here
    if intent == "lights.set":
        return f"Lights to {slots['level']}."

    # Open-ended: there is a transcript (System Two ran). Ask the model.
    if decision.transcript:
        messages = [
            {"role": "system", "content": "You are a spoken voice assistant. Answer "
                                           "in one short sentence -- this is read aloud."},
            *context.messages(turns=6),
        ]
        return model.complete(messages, max_tokens=150)
    return None


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
        config, stt=stt, model=model, observer=observer, schema=MY_SCHEMA,
        on_decision=lambda decision, context: my_agent(decision, context, model=model),
    )
    if pipeline.system_one.error:
        print(f"(System One off: {pipeline.system_one.error})")

    print("Connected an example agent with two intents of its own.")
    print('Try: "lights off", "set the lights to dim", "what time is it", or anything else.\n')
    try:
        pipeline.run_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
