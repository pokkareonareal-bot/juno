"""The live benchmark: Reflex against always-transcribe, clip by clip.

    python -m juno_core.slu bench --rows data/slu/test.jsonl --model model.npz \\
        --fallback-stt parakeet-110m \\
        --baselines parakeet-110m parakeet-0.6b whisper:small.en whisper:large-v3-turbo \\
        --out bench.json --csv bench.csv

Every system sees the same clips, in the same order, after a warm-up, and
is timed end to end on this machine -- nothing here is estimated from stored
numbers (``train``/``evaluate`` do that, cheaply, for model selection; this
is the measurement for the report). Systems:

  always:<stt>          the cascade on every clip: transcribe, intent engine,
                        parser. Today's Juno, with that recogniser.
  reflex+<stt>          Reflex on every clip; the cascade with <stt> only
                        on what it escalates.

Per system: end-to-end correctness against the gold labels, false
activations, missed requests, intent and slot accuracy, STT calls (and the
share avoided), latency per utterance (p50/p95, and separately for typed
commands, where the difference is largest), and compute (wall and CPU
seconds for the whole run).

What it does not measure, and the report should say so: energy (run the
whole command under ``sudo powermetrics --samplers cpu_power,gpu_power -i 500``
and integrate, see the README), the language-model adjudicator unless
``--llm`` is given, and anything about rooms the clips were not recorded in.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from juno_core.events import CASCADE_EVENTS
from juno_core.slu import data as D
from juno_core.slu.schema import ASSISTANT, OPEN_REQUEST, SOURCE_REFLEX


def add_bench_args(s) -> None:
    s.add_argument("--rows", nargs="+", required=True, help="labelled (or gold-only) manifests")
    s.add_argument("--model", help="Reflex student (.npz); omit to run baselines only")
    s.add_argument("--fallback-stt", default="parakeet-110m",
                   help="the recogniser the cascade uses when Reflex escalates")
    s.add_argument("--baselines", nargs="*", default=["parakeet-110m", "whisper:small.en"],
                   help="always-transcribe systems: parakeet-110m | parakeet-0.6b | whisper:<model>")
    s.add_argument("--only-holdout", action="store_true",
                   help="only the clips held out when --model was trained")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--llm", help="adjudicator provider for the cascade (adds network latency)")
    s.add_argument("--name", default="Juno")
    s.add_argument("--out", help="JSON report")
    s.add_argument("--csv", help="one line per (system, clip)")


def _stt(name: str):
    from juno_core.stt import build_stt

    if name.startswith("parakeet"):
        return build_stt({"provider": "parakeet", "model": name})
    if name.startswith("whisper:"):
        return build_stt({"provider": "mlx_whisper", "model": name.split(":", 1)[1]})
    raise SystemExit(f"unknown recogniser {name!r}")


def _gold(row: dict):
    return row.get("g_addressed"), row.get("g_intent"), row.get("g_slots") or {}


def _correct(kind: str, intent, slots, gold) -> bool:
    addressed, g_intent, g_slots = gold
    if addressed != ASSISTANT:
        return kind == "ignore"
    if kind != "act" or intent != g_intent:
        return False
    if g_intent == OPEN_REQUEST:
        return True
    for name, value in g_slots.items():
        got = (slots or {}).get(name)
        try:
            if got is None or int(got) != int(value):
                return False
        except (TypeError, ValueError):
            return False
    return True


class _System:
    def __init__(self, name, run):
        self.name, self.run = name, run


def bench_main(args) -> int:
    from juno_core.slu.teacher import Teacher

    # Collected sessions keep vectors, not audio, so they cannot be replayed here.
    rows = [r for r in D.read_manifest(args.rows)
            if r.get("g_addressed") and (r.get("path") or r.get("generate"))]
    model = None
    if args.model:
        from juno_core.slu.student import StudentModel

        model = StudentModel.load(args.model)
        if args.only_holdout:
            keep = set((model.meta.get("splits") or {}).get("holdout_ids") or [])
            rows = [r for r in rows if r.get("id") in keep]
    if args.limit:
        rng = np.random.default_rng(args.seed)
        rows = [rows[k] for k in sorted(rng.choice(len(rows), min(args.limit, len(rows)), replace=False))]
    if not rows:
        raise SystemExit("no rows with gold labels to benchmark on")
    clips = [D.load_clip(r) for r in rows]
    print(f"benchmarking on {len(rows)} clips", flush=True)

    llm = None
    if args.llm:
        from juno_core.llm import build_model

        llm = build_model({"provider": args.llm})

    teachers: dict = {}

    def teacher(stt_name):
        if stt_name not in teachers:
            stt = _stt(stt_name)
            stt.warmup()
            teachers[stt_name] = Teacher(stt, model=llm, assistant_name=args.name,
                                         schema=model.schema if model else _core())
        return teachers[stt_name]

    systems = []
    for name in args.baselines:
        t = teacher(name)

        def run_always(audio, t=t):
            label = t.label(audio)
            kind = "act" if label.accepted else "ignore"
            return kind, label.intent if label.accepted else None, label.slots, True, "cascade"
        systems.append(_System(f"always:{name}", run_always))

    if model is not None:
        from juno_core.slu.reflex import Reflex

        reflex = Reflex({"mode": "on"}, model=model)
        if not reflex.enabled:
            raise SystemExit(f"Reflex failed to load: {reflex.error}")
        reflex.warmup()
        fallback = teacher(args.fallback_stt)

        def run_hybrid(audio):
            d = reflex.decide(audio)
            if d.route == "ignore":
                return "ignore", None, {}, False, "reflex"
            if d.route == "act":
                return "act", d.intent.value, {k: v.value for k, v in d.slots.items()}, False, "reflex"
            label = fallback.label(audio)
            kind = "act" if label.accepted else "ignore"
            return kind, label.intent if label.accepted else None, label.slots, True, "cascade"
        systems.append(_System(f"reflex+{args.fallback_stt}", run_hybrid))

    per_clip = []
    report = {"clips": len(rows), "machine": _machine(), "systems": {}}
    for system in systems:
        # One untimed pass over a few clips: first-call costs are not latency.
        for audio in clips[:3]:
            system.run(audio)
        wall0, cpu0 = time.perf_counter(), time.process_time()
        results = []
        for row, audio in zip(rows, clips):
            t0 = time.perf_counter()
            kind, intent, slots, stt_called, source = system.run(audio)
            ms = (time.perf_counter() - t0) * 1000.0
            ok = _correct(kind, intent, slots, _gold(row))
            results.append((row, kind, intent, slots, stt_called, source, ms, ok))
            per_clip.append({"system": system.name, "id": row.get("id"), "category": row.get("category"),
                             "source": row.get("source"), "gold_addressed": row.get("g_addressed"),
                             "gold_intent": row.get("g_intent"), "decision": kind, "intent": intent,
                             "slots": json.dumps(slots), "stt_called": stt_called,
                             "decided_by": source, "ms": round(ms, 3), "correct": ok})
        wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
        report["systems"][system.name] = _summarise(results, wall, cpu)
        s = report["systems"][system.name]
        print(f"  {system.name:32s} correct {s['correct']:.3f}  false-act {s['false_activation']['rate']}"
              f"  missed {s['missed']['rate']}  stt calls {s['stt_calls']}  p50 {s['latency_ms']['p50']} ms",
              flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.out}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_clip[0]))
            writer.writeheader()
            writer.writerows(per_clip)
        print(f"wrote {args.csv}")
    return 0


def _core():
    from juno_core.slu.schema import CORE_SCHEMA

    return CORE_SCHEMA


def _summarise(results, wall: float, cpu: float) -> dict:
    from juno_core.intelligence.gate_training import wilson_upper

    def rate(k, n):
        return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
                "upper95": round(wilson_upper(k, n), 4) if n else None}

    n = len(results)
    ms = np.asarray([r[6] for r in results])
    assist = [r for r in results if r[0].get("g_addressed") == ASSISTANT]
    non = [r for r in results if r[0].get("g_addressed") != ASSISTANT]
    typed = [r for r in assist if r[0].get("g_intent") not in (None, OPEN_REQUEST)]
    typed_ms = np.asarray([r[6] for r in typed]) if typed else np.asarray([])
    by_cat: dict = defaultdict(lambda: [0, 0])
    for r in results:
        by_cat[r[0].get("category") or "?"][0] += 1
        by_cat[r[0].get("category") or "?"][1] += int(r[7])
    stt_calls = sum(1 for r in results if r[4])
    return {
        "correct": round(float(np.mean([r[7] for r in results])), 4),
        "false_activation": rate(sum(1 for r in non if r[1] == "act"), len(non)),
        "missed": rate(sum(1 for r in assist if r[1] != "act"), len(assist)),
        "intent_accuracy_when_served": round(float(np.mean(
            [r[2] == r[0].get("g_intent") for r in assist if r[1] == "act"])), 4) if any(
            r[1] == "act" for r in assist) else None,
        "typed_correct": round(float(np.mean([r[7] for r in typed])), 4) if typed else None,
        "stt_calls": stt_calls,
        "stt_avoided": round(1.0 - stt_calls / n, 4),
        "decided_by": dict(Counter(r[5] for r in results)),
        "latency_ms": {"p50": round(float(np.percentile(ms, 50)), 2),
                       "p95": round(float(np.percentile(ms, 95)), 2),
                       "mean": round(float(ms.mean()), 2)},
        "typed_latency_ms": {"p50": round(float(np.percentile(typed_ms, 50)), 2),
                             "p95": round(float(np.percentile(typed_ms, 95)), 2)} if typed_ms.size else None,
        "compute_s": {"wall": round(wall, 3), "cpu": round(cpu, 3),
                      "wall_per_utterance_ms": round(wall / n * 1000.0, 2)},
        "correct_by_category": {k: round(v[1] / v[0], 4) for k, v in sorted(by_cat.items())},
    }


def _machine() -> dict:
    import platform
    import subprocess

    chip = ""
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        pass
    return {"platform": platform.platform(), "python": platform.python_version(), "chip": chip}


# -- the shadow log ---------------------------------------------------------------

def _reflex_ms(latency: dict | None) -> float | None:
    """Reflex's own latency from a logged decision, under whichever spelling the log used."""
    latency = latency or {}
    for key in (SOURCE_REFLEX, "reflex", "system_one"):
        if latency.get(key) is not None:
            return latency[key]
    return None


def shadow_report(path: str | Path) -> dict:
    """Reflex's decision next to the cascade's, for every turn both scored."""
    turns: dict[str, dict] = defaultdict(dict)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            turn = event.get("turn")
            if not turn:
                continue
            if event.get("event") == "slu_decided":
                turns[turn]["one"] = event
            elif event.get("event") in CASCADE_EVENTS:
                turns[turn]["two"] = event
    pairs = [(t, v["one"], v.get("two")) for t, v in turns.items() if "one" in v]
    matrix = Counter()
    dangerous, latency = [], []
    for turn, one, two in pairs:
        latency.append(_reflex_ms(one.get("latency_ms")))
        # In mode "on", the cascade only runs on escalations.
        verdict = "cascade_not_run" if two is None else two.get("route")
        matrix[(one.get("route"), verdict)] += 1
        if two is None:
            continue
        if one.get("route") == "ignore" and two.get("route") == "act":
            dangerous.append({"turn": turn, "kind": "would_have_missed",
                              "p_assistant": (one.get("addressed") or {}).get("probs", {}).get(ASSISTANT),
                              "cascade": two.get("transcript")})
        elif one.get("route") == "act":
            one_intent = (one.get("intent") or {}).get("value")
            two_intent = (two.get("intent") or {}).get("value")
            one_slots = {k: v.get("value") for k, v in (one.get("slots") or {}).items()}
            two_slots = {k: v.get("value") for k, v in (two.get("slots") or {}).items()}
            if two.get("route") != "act":
                dangerous.append({"turn": turn, "kind": "would_have_acted_on_ignored",
                                  "acted_as": one_intent, "cascade": two.get("transcript")})
            elif one_intent != two_intent or one_slots != two_slots:
                dangerous.append({"turn": turn, "kind": "would_have_acted_differently",
                                  "acted_as": [one_intent, one_slots],
                                  "cascade": [two_intent, two_slots],
                                  "transcript": two.get("transcript")})
    lat = [v for v in latency if v is not None]
    return {
        "turns": len(pairs),
        "agreement": {f"{a} / {b}": n for (a, b), n in sorted(matrix.items())},
        "would_have_avoided_stt": sum(n for (a, _), n in matrix.items() if a in ("act", "ignore")),
        "disagreements_that_matter": dangerous,
        "reflex_ms": {"p50": round(float(np.percentile(lat, 50)), 2) if lat else None,
                      "p95": round(float(np.percentile(lat, 95)), 2) if lat else None},
    }
