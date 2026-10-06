#!/usr/bin/env python3
"""Run the pipeline end to end: microphone in, an answer out.

    python run.py                    uses config.yaml
    python run.py --config x.yaml    uses a different file
    python run.py --quiet            only print what was heard and answered

Before running this: copy config.example.yaml to config.yaml and .env.example
to .env, then follow README.md -- it's four short steps (pick an STT engine,
pick an LLM provider, optionally enrol your voice, run this).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_dotenv(path: Path) -> None:
    """The three lines of .env this project needs don't justify a dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


class CoreActions:
    """The core schema's intents, done locally. A reference, not a product:
    replace any branch with the real thing (your media player, your timers)."""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.timer = None

    def handle(self, decision, context) -> str | None:
        """A reply, "" for handled-and-silent, or None for "not mine"."""
        import threading
        from datetime import datetime

        from juno_core.slu.schema import describe_duration

        intent = decision.intent.value if decision.intent else None
        slots = {k: v.value for k, v in decision.slots.items()}
        if intent == "time.now":
            return f"It's {datetime.now().strftime('%H:%M')}."
        if intent == "timer.set" and "duration" in slots:
            if self.timer is not None:
                self.timer.cancel()
            seconds = int(slots["duration"])

            def ring():
                self.pipeline.active_contexts.discard("timer_running")
                print(f"\n  ** timer done ({describe_duration(seconds)}) **\n")

            self.timer = threading.Timer(seconds, ring)
            self.timer.daemon = True
            self.timer.start()
            self.pipeline.active_contexts.add("timer_running")
            return f"Timer set for {describe_duration(seconds)}."
        if intent == "timer.cancel" and self.timer is not None:
            self.timer.cancel()
            self.timer = None
            self.pipeline.active_contexts.discard("timer_running")
            return "Timer cancelled."
        if intent == "repeat":
            return context.last_ai_text or "I haven't said anything yet."
        if intent in ("stop", "cancel"):
            return "" if intent == "stop" else "Okay."       # "" = handled, say nothing
        if intent in ("volume.up", "volume.down", "media.pause", "media.resume"):
            return f"({intent} -- wire this to your player)"
        return None


def banner(assistant_name: str, stt_name: str, llm_name: str, tts_name: str | None) -> None:
    print("=" * 64)
    print(f"  {assistant_name} is listening. No wake word -- just talk.")
    print(f"  hearing you with : {stt_name}")
    print(f"  thinking with    : {llm_name}")
    print(f"  speaking with    : {tts_name or 'nothing -- answers print here'}")
    print("  Ctrl+C to stop.")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--quiet", action="store_true",
                        help="print only what was heard and answered, nothing else")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    sys.path.insert(0, str(ROOT))

    from juno_core.config import load_config
    from juno_core.llm import build_model
    from juno_core.observability import Observer
    from juno_core.pipeline import JunoPipeline
    from juno_core.stt import build_stt
    from juno_core.tts import build_tts

    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"\n{exc}\n")
        raise SystemExit(1)

    try:
        stt = build_stt(config.get("stt") or {})
    except (ValueError, ImportError, RuntimeError) as exc:
        print(f"\nCouldn't set up speech-to-text:\n  {exc}\n")
        raise SystemExit(1)

    try:
        model = build_model(config.get("llm") or {})
    except Exception as exc:  # noqa: BLE001 - surfacing a clear setup error beats a traceback
        print(f"\nCouldn't set up the language model:\n  {exc}\n")
        raise SystemExit(1)

    tts = None
    tts_config = config.get("tts")
    if tts_config and tts_config.get("provider"):
        try:
            tts = build_tts(tts_config)
        except Exception as exc:  # noqa: BLE001
            print(f"(text-to-speech unavailable, continuing without it: {exc})\n")

    log_path = ROOT / "logs" / "events.jsonl"
    observer = Observer(log_path=log_path, console=not args.quiet, verbose=False)

    pipeline = JunoPipeline(config, stt=stt, model=model, observer=observer)

    def on_accept(text, decision, context):
        reply = pipeline.answer_with_model(text, decision, context)
        print(f"\n  you said : {text}")
        print(f"  answer   : {reply}\n")
        if tts is not None and reply:
            try:
                tts.speak(reply)
            except Exception as exc:  # noqa: BLE001
                print(f"  (couldn't speak that: {exc})")
        return reply

    pipeline.on_accept = on_accept

    if pipeline.reflex.enabled:
        # Typed decisions, from either path. Reflex's arrive with no
        # transcript at all -- the core intents are handled right here, with
        # no speech-to-text and no language model; anything open-ended came
        # through the cascade and goes to the model as before.
        actions = CoreActions(pipeline)

        def on_decision(decision, context):
            heard = decision.transcript or f"(no transcript) {decision.summary()}"
            print(f"\n  you said : {heard}   [{decision.source}]")
            reply = actions.handle(decision, context)
            if reply is None and decision.transcript:
                reply = pipeline.answer_with_model(decision.transcript, decision.detail, context)
            if reply:
                print(f"  answer   : {reply}\n")
                if tts is not None:
                    try:
                        tts.speak(reply)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  (couldn't speak that: {exc})")
            return reply

        pipeline.on_decision = on_decision
        print(f"Reflex SLU: {pipeline.reflex.mode} ({pipeline.reflex.model.tag})")
    elif pipeline.reflex.error:
        print(f"(Reflex SLU unavailable, running as before: {pipeline.reflex.error})")

    banner(pipeline.assistant_name, stt.name, model.name if model else "nothing",
           tts.name if tts else None)
    print("Warming up speech-to-text (first run may take a moment)...")
    try:
        pipeline.run_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:  # noqa: BLE001 - clear message beats a raw traceback here
        print(f"\nStopped because of an error: {exc}\n")
        raise
    finally:
        observer.close()


if __name__ == "__main__":
    main()
