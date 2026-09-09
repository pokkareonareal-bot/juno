#!/usr/bin/env python3
"""Teach it your voice. Optional, but it's most of what makes the gate work
well with more than one person in the room -- see README.md, "Teaching it
your voice", and REPORT.txt section 4.2 for the measured accuracy (97.9% on
the bundled eval, worst case still +0.213 of separation).

    python enroll.py

Eight short prompted sentences, about a minute of talking. Nothing is kept
except 192 numbers derived from your voice (no audio), written to the path
named in config.yaml's own_voice.profile. Ctrl+C any time to stop without
changing anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    sys.path.insert(0, str(ROOT))

    import sounddevice as sd

    from juno_core.audio.calibration import Calibration
    from juno_core.config import Section, load_config

    try:
        config = load_config()
    except FileNotFoundError as exc:
        print(f"\n{exc}\n")
        raise SystemExit(1)

    config_path = Path(config.get("_meta", {}).get("path", "config.yaml"))
    own_voice_cfg = config.get("own_voice") or Section({})
    sample_rate = int((config.get("audio") or Section({})).get("sample_rate", 16000))
    clip_seconds = 4.0

    calibration = Calibration(config_path=config_path, own_voice=own_voice_cfg)
    progress = calibration.start()

    print("=" * 64)
    print("  Teaching it your voice")
    print("=" * 64)
    print("  8 short sentences. Press Enter, then say the line out loud.")
    print("  A second speaker sharpens the result but isn't required --")
    print("  you can skip that half.")
    print("  Ctrl+C any time to stop without changing anything.\n")

    try:
        while progress.stage in ("wearer", "bystander"):
            if progress.stage == "bystander":
                print(f"[{progress.collected}/{progress.wanted}] optional: have "
                      f"someone ELSE say:")
                print(f'  "{progress.prompt}"')
                answer = input(
                    "  Press Enter to record them, or type 'skip' to finish "
                    "without it: "
                ).strip().lower()
                if answer == "skip":
                    progress = calibration.skip()
                    continue
            else:
                print(f"[{progress.collected}/{progress.wanted}] say:")
                print(f'  "{progress.prompt}"')
                input("  Press Enter, then speak... ")

            print(f"  recording ({clip_seconds:.0f}s)...", end=" ", flush=True)
            clip = sd.rec(int(clip_seconds * sample_rate), samplerate=sample_rate,
                          channels=1, dtype="float32")
            sd.wait()
            print("done.")
            progress = calibration.collect(clip.reshape(-1))
            if progress.message:
                print(f"  ({progress.message})")
            print()
    except KeyboardInterrupt:
        print("\n\nStopped -- nothing was saved.")
        return

    result = progress.result or {}
    if progress.stage == "done" and result.get("measured") and result.get("separated"):
        print(f"Measured a threshold of {result['threshold']} on this microphone.")
        applied = calibration.apply()
        print(f"\n{applied.message}\n")
    elif progress.stage == "done":
        print(result.get("note") or "Could not measure a clean separation from "
              "these clips.")
        print("Own-voice gating stays off. Try again any time: python enroll.py\n")
    else:
        print(f"Stopped at stage {progress.stage!r}: {progress.message}\n")


if __name__ == "__main__":
    main()
