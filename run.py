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
