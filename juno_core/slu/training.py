"""Teaching Reflex SLU, offline.

    python -m juno_core.slu sources
    python -m juno_core.slu schema   --out my_schema.json           # start an agent schema
    python -m juno_core.slu synth    --out data/slu/syn --size 3000 --augment 1 --noise 150
    python -m juno_core.slu import-ami --out data/slu/real
    python -m juno_core.slu import-speech-commands --out data/slu/real
    python -m juno_core.slu label    --manifest data/slu/syn/manifest.jsonl ... \\
                                     --out data/slu/labelled.jsonl --stt parakeet-0.6b
    python -m juno_core.slu relabel  --rows data/slu/labelled.jsonl --judge mlx \
                                     --out data/slu/judged.jsonl     # a better teacher
    python -m juno_core.slu embed    --rows data/slu/labelled.jsonl --encoder parakeet \\
                                     --out data/slu/emb
    python -m juno_core.slu train    --rows data/slu/labelled.jsonl \\
                                     --embeddings data/slu/emb/<spec>.npz --out model.npz
    python -m juno_core.slu evaluate --rows test.jsonl --embeddings test.npz --model model.npz
    python -m juno_core.slu sweep    --rows ... --emb-dirs data/slu/emb --out reports/sweep
    python -m juno_core.slu collect  --out data/slu/consented/s1 --speakers p1,p2 \
                                     --room kitchen --consent release
    python -m juno_core.slu bench    --rows test.jsonl --model model.npz --baselines ...
    python -m juno_core.slu shadow   --log logs/events.jsonl
    python -m juno_core.slu studio                                  # the same, in a browser

THE PIPELINE
------------
``synth``/``import-*`` produce clips and a manifest with GOLD labels and
provenance. ``label`` runs the TEACHER (the cascade: speech-to-text, the intent
engine, the parser) over every clip and records its soft answer next to the
gold one. ``embed`` runs an encoder over every clip and stores one pooled
vector per clip -- several encoders can be embedded side by side, which is
how encoders are compared. ``train`` fits the student on the vectors, with
targets from the teacher (distillation), the gold labels, or a mix.

HOW THRESHOLDS ARE CHOSEN
-------------------------
Rows are split by speaker into train / select / holdout, so no voice is in
two splits. The student is fitted on train, early-stopped and calibrated on
select, and its thresholds are chosen on select:

  ignore_below   the largest value (capped at 0.5) whose false-ignore rate --
                 assistant-directed rows that would be dropped -- is within
                 --max-false-ignore;
  act_*          the combination that acts on the most rows while wrong acts
                 stay within --max-wrong-act of acts AND acts on speech that
                 was not for the assistant stay within --max-false-activation
                 of such rows -- both judged on an upper confidence bound
                 (--margin-z), because select is only a few speakers.

Holdout is touched once, to report. Every number in the report comes from
the runtime Router fed the stored vector, so what is evaluated is the code
that runs.

Truth, for choosing thresholds and for reporting, is the gold label where a
row has one and the teacher's verdict where it does not (``--truth`` can
force either). Offline rows are judged cold -- nothing pending, the assistant
has not spoken -- so intents bound to a conversation state (``confirm.yes``
needs an open question) escalate in these numbers. That is conservative on
purpose: the context they need does not exist in a clip.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from juno_core.intelligence.gate_training import choose_threshold, group_split, wilson_upper
from juno_core.slu import data as D
from juno_core.slu.router import ConversationState, Router, Thresholds
from juno_core.slu.schema import ADDRESSEES, ASSISTANT, CORE_SCHEMA, OPEN_REQUEST, Schema
from juno_core.slu.student import (
    OTHER, StudentModel, build_targets, expected_calibration_error, fit,
)

GRID = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


# -- joining rows and vectors ------------------------------------------------

def load_embeddings(paths: str | Path | Sequence[str | Path]) -> dict:
    """One or more vector tables from the same encoder, concatenated."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    ids, Xs, ms, spec = [], [], [], None
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            this = str(z["spec"])
            if spec is not None and this != spec:
                raise SystemExit(f"{path} is {this}, the others are {spec}: one encoder at a time")
            spec = this
            ids += [str(i) for i in z["ids"]]
            Xs.append(z["X"].astype(np.float32))
            ms.append(z["encode_ms"].astype(np.float64))
    return {"ids": ids, "X": np.concatenate(Xs), "encode_ms": np.concatenate(ms), "spec": spec}


def join(rows: Sequence[dict], emb: dict) -> tuple[list[dict], np.ndarray, np.ndarray]:
    index = {i: k for k, i in enumerate(emb["ids"])}
    kept, idx = [], []
    for row in rows:
        k = index.get(row.get("id"))
        if k is not None:
            kept.append(row)
            idx.append(k)
    if not kept:
        raise SystemExit("no row ids match the embeddings file -- embed these rows first")
    idx = np.asarray(idx)
    return kept, emb["X"][idx], emb["encode_ms"][idx]


# -- truth ---------------------------------------------------------------------

def truth(row: dict, mode: str = "auto") -> tuple[str | None, str | None, dict]:
    """(addressed, intent, slots) to judge against: gold if known, else teacher."""
    use_gold = mode == "gold" or (mode == "auto" and row.get("g_addressed"))
    if use_gold:
        addressed, intent = row.get("g_addressed"), row.get("g_intent")
        slots = row.get("g_slots") or {}
        # A prompt can fix WHO without fixing WHAT ("give Juno any command"):
        # in auto mode the teacher fills in the intent the gold label lacks.
        if mode == "auto" and addressed == ASSISTANT and not intent and row.get("t_intent"):
            intent, slots = row.get("t_intent"), row.get("t_slots") or {}
        return addressed, intent, slots
    if not row.get("t_addressed"):
        return None, None, {}
    addressed = ASSISTANT if row.get("t_accept") else (
        max((a for a in ADDRESSEES if a != ASSISTANT),
            key=lambda a: row["t_addressed"].get(a, 0.0)))
    intent = row.get("t_intent") if addressed == ASSISTANT else None
    return addressed, intent, row.get("t_slots") or {}


# -- scoring with the runtime router ---------------------------------------------

def predict_all(model: StudentModel, X: np.ndarray):
    probs = model.probabilities(X)
    preds = []
    from juno_core.slu.student import Prediction

    for r in range(X.shape[0]):
        preds.append(Prediction(
            addressed=dict(zip(model.addressees, map(float, probs["addressed"][r]))),
            intent=dict(zip(model.intents, map(float, probs["intent"][r]))),
            slots={k: dict(zip(v, map(float, probs[k][r]))) for k, v in model.slot_values.items()},
        ))
    return preds


def route_all(model: StudentModel, preds, thresholds: Thresholds, schema: Schema,
              state: ConversationState | None = None) -> list:
    router = Router(schema, thresholds)
    return [router.route(p, state) for p in preds]


def _act_correct(routed, t_addr, t_intent, t_slots) -> bool:
    """Would acting on this routed decision have been right?"""
    return _correct_act(routed.intent.value if routed.intent else None, routed.slots,
                        t_addr, t_intent, t_slots)


def _correct_act(intent: str | None, slots: dict, t_addr, t_intent, t_slots) -> bool:
    if t_addr != ASSISTANT or intent is None or intent != t_intent:
        return False
    for name, f in slots.items():
        if f.value == OTHER or t_slots.get(name) is None or int(f.value) != int(t_slots[name]):
            return False
    return True


# -- threshold selection -------------------------------------------------------

def choose_thresholds(model: StudentModel, X: np.ndarray, rows: Sequence[dict], schema: Schema,
                      *, max_false_ignore: float, max_wrong_act: float,
                      max_false_activation: float, truth_mode: str = "auto",
                      margin_z: float = 1.0) -> tuple[Thresholds, dict]:
    """Thresholds for the router, from the select split (see the module notes).

    The act budgets are applied to an UPPER CONFIDENCE BOUND (Wilson, ``margin_z``
    standard errors), not to the rate seen on select. The select split is a
    handful of speakers; thresholds that only just fit its point estimate
    overshoot on new voices -- measured here, a 1% budget met on select came
    out at 2-3% on holdout. ``margin_z=0`` is the plain point estimate.

    The search is vectorised, but it mirrors Router exactly -- and the report
    afterwards is produced by Router itself, so a mismatch would show.
    """
    preds = predict_all(model, X)
    truths = [truth(r, truth_mode) for r in rows]
    n = len(rows)
    p_assist = np.asarray([p.addressed.get(ASSISTANT, 0.0) for p in preds])
    is_assist = np.asarray([t[0] == ASSISTANT for t in truths])
    info: dict = {}
    ignore_below = None
    if is_assist.any():
        ignore_below = round(choose_threshold(p_assist, np.ones(n, bool), is_assist,
                                              max_false_ignore, ceiling=0.5), 6)
        info["false_ignores_on_select"] = int(((p_assist < ignore_below) & is_assist).sum())

    # What does not depend on the thresholds: the top intent, whether it is
    # actable at all in a cold state, its slots' confidence, and correctness.
    cold = ConversationState()
    actable = np.zeros(n, bool)
    p_intent = np.zeros(n)
    p_slot = np.ones(n)
    floor = np.zeros(n)
    correct = np.zeros(n, bool)
    router = Router(schema, Thresholds(act_addressed=0.0, act_intent=0.0, act_slot=0.0))
    for k, p in enumerate(preds):
        name, pi = p.top(p.intent)
        spec = schema.intent(name)
        p_intent[k] = pi
        if spec is None or spec.name == OPEN_REQUEST or spec.needs_transcript:
            continue
        if any(c not in cold.contexts() for c in spec.when):
            continue
        slots = router._slots(name, p)
        if any(s.required and s.name not in slots for s in spec.slots):
            continue
        if any(f.value == OTHER for f in slots.values()):
            continue
        actable[k] = True
        p_slot[k] = min([f.p for f in slots.values()] or [1.0])
        floor[k] = spec.min_confidence or 0.0
        correct[k] = _correct_act(name, slots, *truths[k])

    n_non = int((~is_assist).sum())
    best = None
    for ta, ti, ts in itertools.product(GRID, GRID, GRID):
        acts = actable & (p_assist >= ta) & (p_intent >= np.maximum(ti, floor)) & (p_slot >= ts)
        if ignore_below is not None:
            acts &= p_assist >= ignore_below
        n_acts = int(acts.sum())
        if not n_acts:
            continue
        wrong = int((acts & ~correct).sum())
        false_act = int((acts & ~is_assist).sum())
        if _bound(wrong, n_acts, margin_z) > max_wrong_act + 1e-12:
            continue
        if _bound(false_act, max(n_non, 1), margin_z) > max_false_activation + 1e-12:
            continue
        key = (int((acts & correct).sum()), ta + ti + ts)
        if best is None or key > best[0]:
            best = (key, (ta, ti, ts), wrong, false_act, n_acts)
    if best is None:
        info["act"] = "no threshold combination met the act budgets on select: Reflex will not act"
        return Thresholds(ignore_below=ignore_below), info
    _, (ta, ti, ts), wrong, false_act, n_acts = best
    info["act_on_select"] = {"acts": n_acts, "wrong": wrong, "false_activations": false_act}
    return Thresholds(ignore_below=ignore_below, act_addressed=ta, act_intent=ti, act_slot=ts), info


def _bound(k: int, n: int, z: float) -> float:
    """k/n, or its Wilson upper bound at z standard errors."""
    if n == 0:
        return 0.0
    return k / n if z <= 0 else wilson_upper(k, n, z)


# -- the report ----------------------------------------------------------------

def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos, neg = scores[labels], scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def _rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
            "upper95": round(wilson_upper(k, n), 4) if n else None}


def evaluate(model: StudentModel, X: np.ndarray, rows: Sequence[dict], *,
             encode_ms: np.ndarray | None = None, schema: Schema | None = None,
             thresholds: Thresholds | None = None, truth_mode: str = "auto",
             state: ConversationState | None = None) -> dict:
    """The full report for these rows, judged by the runtime Router."""
    schema = schema or model.schema
    thresholds = thresholds or Thresholds.from_dict(model.thresholds)
    preds = predict_all(model, X)
    routed = route_all(model, preds, thresholds, schema, state)
    truths = [truth(r, truth_mode) for r in rows]
    n = len(rows)
    routes = np.asarray([r.route for r in routed])
    is_assist = np.asarray([t[0] == ASSISTANT for t in truths])
    has_truth = np.asarray([t[0] is not None for t in truths])
    p_assist = np.asarray([p.addressed.get(ASSISTANT, 0.0) for p in preds])

    out: dict = {"rows": n, "truth": truth_mode, "thresholds": thresholds.to_dict()}
    out["routes"] = {k: int((routes == k).sum()) for k in ("act", "ignore", "escalate")}
    out["stt_avoided"] = round(float(np.mean(routes != "escalate")), 4) if n else None

    out["false_ignore"] = _rate(int(((routes == "ignore") & is_assist).sum()), int(is_assist.sum()))
    non = has_truth & ~is_assist
    out["false_activation"] = _rate(int(((routes == "act") & non).sum()), int(non.sum()))
    out["correct_ignore"] = _rate(int(((routes == "ignore") & non).sum()), int(non.sum()))
    acts = routes == "act"
    correct_act = np.asarray([r.route == "act" and _act_correct(r, *t) for r, t in zip(routed, truths)])
    out["wrong_act"] = _rate(int((acts & ~correct_act).sum()), int(acts.sum()))
    typed_assist = np.asarray([t[0] == ASSISTANT and t[1] not in (None, OPEN_REQUEST) for t in truths])
    out["typed_commands_acted"] = _rate(int((acts & correct_act & typed_assist).sum()),
                                        int(typed_assist.sum()))

    # Raw classifier quality, before any routing.
    top_addr = [max(p.addressed, key=p.addressed.get) for p in preds]
    addr_ok = np.asarray([a == t[0] for a, t in zip(top_addr, truths)])
    out["addressed_accuracy"] = round(float(addr_ok[has_truth].mean()), 4) if has_truth.any() else None
    auc = _auc(p_assist[has_truth], is_assist[has_truth])
    out["addressed_auc"] = round(auc, 4) if auc is not None else None
    p_top_addr = np.asarray([max(p.addressed.values()) for p in preds])
    out["addressed_ece"] = round(expected_calibration_error(p_top_addr[has_truth], addr_ok[has_truth]), 4)
    intent_rows = [k for k, t in enumerate(truths) if t[0] == ASSISTANT and t[1]]
    if intent_rows:
        top_i = [max(preds[k].intent, key=preds[k].intent.get) for k in intent_rows]
        ok = np.asarray([ti == truths[k][1] for ti, k in zip(top_i, intent_rows)])
        p_i = np.asarray([max(preds[k].intent.values()) for k in intent_rows])
        out["intent_accuracy"] = round(float(ok.mean()), 4)
        out["intent_ece"] = round(expected_calibration_error(p_i, ok), 4)
        confusion = Counter((truths[k][1], ti) for ti, k in zip(top_i, intent_rows) if ti != truths[k][1])
        out["intent_confusions"] = [{"true": a, "predicted": b, "n": c}
                                    for (a, b), c in confusion.most_common(10)]
    slot_hits = []
    for k, t in enumerate(truths):
        if t[0] != ASSISTANT or not t[2]:
            continue
        for slot, value in t[2].items():
            dist = preds[k].slots.get(f"{t[1]}:{slot}")
            if not dist:
                continue
            guess = max(dist, key=dist.get)
            classes = [c for c in dist if c != OTHER]
            want = int(value) if int(value) in [int(c) for c in classes] else OTHER
            slot_hits.append(guess == want or (guess != OTHER and want != OTHER and int(guess) == int(want)))
    if slot_hits:
        out["slot_accuracy"] = round(float(np.mean(slot_hits)), 4)

    # Teacher: how good is it against gold, and how closely does the student follow it?
    teach = [r for r in rows if r.get("t_addressed") and r.get("g_addressed")]
    if teach:
        t_ok = [(ASSISTANT if r["t_accept"] else "other") == (ASSISTANT if r["g_addressed"] == ASSISTANT else "other")
                for r in teach]
        out["teacher_vs_gold"] = {
            "rows": len(teach),
            "addressed_binary_accuracy": round(float(np.mean(t_ok)), 4),
            "false_activation": _rate(sum(1 for r in teach if r["t_accept"] and r["g_addressed"] != ASSISTANT),
                                      sum(1 for r in teach if r["g_addressed"] != ASSISTANT)),
            "missed": _rate(sum(1 for r in teach if not r["t_accept"] and r["g_addressed"] == ASSISTANT),
                            sum(1 for r in teach if r["g_addressed"] == ASSISTANT)),
            "intent_accuracy_on_accepted": _safe_mean(
                [r.get("t_intent") == r.get("g_intent") for r in teach
                 if r["t_accept"] and r["g_addressed"] == ASSISTANT]),
        }

    # End to end: Reflex where it decided, the cascade (the teacher's recorded
    # verdict) where it escalated, against the teacher alone on every row.
    hybrid_ok, teacher_ok, stt_calls = [], [], 0
    for r_, row, t in zip(routed, rows, truths):
        if t[0] is None or not row.get("t_addressed"):
            continue
        teacher_outcome = _outcome_teacher(row)
        if r_.route == "escalate":
            stt_calls += 1
            outcome = teacher_outcome
        elif r_.route == "ignore":
            outcome = ("ignore", None, {})
        else:
            outcome = ("act", r_.intent.value, {k: v.value for k, v in r_.slots.items()})
        hybrid_ok.append(_outcome_correct(outcome, t))
        teacher_ok.append(_outcome_correct(teacher_outcome, t))
    if hybrid_ok:
        out["end_to_end"] = {
            "rows": len(hybrid_ok),
            "hybrid_correct": round(float(np.mean(hybrid_ok)), 4),
            "teacher_only_correct": round(float(np.mean(teacher_ok)), 4),
            "stt_calls_hybrid": stt_calls, "stt_calls_teacher_only": len(hybrid_ok),
        }
        stt_ms = [float(r["stt_ms"]) for r in rows if r.get("stt_ms") is not None]
        if stt_ms and encode_ms is not None:
            mean_stt = float(np.mean(stt_ms))
            mean_enc = float(np.mean(encode_ms))
            out["end_to_end"]["compute_ms_per_utterance"] = {
                "teacher_only": round(mean_stt, 2),
                "hybrid": round(mean_enc + mean_stt * stt_calls / len(hybrid_ok), 2),
                "note": "teacher STT time as recorded at labelling; Reflex encode time as "
                        "recorded at embedding. Text-side time is excluded from both.",
            }

    out["by_category"] = _breakdown(rows, routed, truths, "category")
    out["by_source"] = _breakdown(rows, routed, truths, "source")
    if encode_ms is not None and len(encode_ms):
        out["encode_ms"] = {"p50": round(float(np.percentile(encode_ms, 50)), 3),
                            "p95": round(float(np.percentile(encode_ms, 95)), 3)}
    out["worst_false_ignores"] = [
        {"id": row.get("id"), "text": row.get("text") or row.get("transcript"),
         "p_assistant": round(float(p_assist[k]), 4)}
        for k, row in sorted(enumerate(rows), key=lambda kv: p_assist[kv[0]])
        if routes[k] == "ignore" and is_assist[k]][:10]
    out["false_activations"] = [
        {"id": row.get("id"), "text": row.get("text") or row.get("transcript"),
         "acted_as": routed[k].intent.value if routed[k].intent else None,
         "truth": truths[k][0]}
        for k, row in enumerate(rows) if routes[k] == "act" and non[k]][:10]
    return out


def _safe_mean(values):
    return round(float(np.mean(values)), 4) if values else None


def _outcome_teacher(row: dict):
    if not row.get("t_accept"):
        return ("ignore", None, {})
    return ("act", row.get("t_intent"), row.get("t_slots") or {})


def _outcome_correct(outcome, t) -> bool:
    kind, intent, slots = outcome
    addressed, t_intent, t_slots = t
    if addressed != ASSISTANT:
        return kind == "ignore"
    if kind != "act" or intent != t_intent:
        return False
    if t_intent == OPEN_REQUEST:
        return True
    for name, value in (t_slots or {}).items():
        got = (slots or {}).get(name)
        if got is None or got == OTHER or int(got) != int(value):
            return False
    return True


def _breakdown(rows, routed, truths, key: str) -> dict:
    groups: dict = defaultdict(lambda: {"n": 0, "act": 0, "ignore": 0, "escalate": 0,
                                        "addressed_correct": 0, "wrong": 0})
    for row, r, t in zip(rows, routed, truths):
        g = groups[row.get(key) or "?"]
        g["n"] += 1
        g[r.route] += 1
        if r.addressed.value == t[0]:
            g["addressed_correct"] += 1
        if r.route == "ignore" and t[0] == ASSISTANT:
            g["wrong"] += 1
        elif r.route == "act" and not _act_correct(r, *t):
            g["wrong"] += 1
    out = {}
    for name, g in sorted(groups.items()):
        n = g["n"]
        out[name] = {"n": n, "act": g["act"], "ignore": g["ignore"], "escalate": g["escalate"],
                     "addressed_accuracy": round(g["addressed_correct"] / n, 4),
                     "wrong_decisions": g["wrong"]}
    return out


# -- commands --------------------------------------------------------------------

def _cmd_sources(args) -> int:
    for key, info in D.SLU_SOURCES.items():
        dist = {True: "distributable", False: "NOT distributable", None: "depends on row"}[info["distributable"]]
        print(f"{key:16s} {str(info['license']):20s} {dist:18s} {info['name']}")
    return 0


def _cmd_schema(args) -> int:
    schema = CORE_SCHEMA
    if args.extend:
        schema = schema.extend(Schema.load(args.extend))
    text = json.dumps(schema.to_dict(), indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        _log(f"wrote {args.out}")
    else:
        print(text)
    return 0


def _cmd_synth(args) -> int:
    from juno_core.slu.corpus import build_corpus

    schema = Schema.load(args.schema) if args.schema else CORE_SCHEMA
    mix = None
    if args.mix:
        mix = {k: float(v) for k, v in (part.split("=") for part in args.mix.split(","))}
    lines = build_corpus(schema, size=args.size, seed=args.seed, name=args.name, mix=mix)
    voices = [v.strip() for v in args.voices.split(",")] if args.voices else None
    _log(f"voicing {len(lines)} lines with {len(voices or D.say_voices())} voices")
    rows = D.synthesise(lines, args.out, voices=voices, augment_copies=args.augment,
                        seed=args.seed, workers=args.workers, noise_clips=args.noise,
                        keep_clean=not args.no_clean, log=_log)
    n = D.write_manifest(Path(args.out) / "manifest.jsonl", rows)
    _log(f"wrote {n} clips to {args.out}/manifest.jsonl (source synthetic_say: "
         f"NOT distributable -- see `sources`)")
    return 0


def _cmd_import_ami(args) -> int:
    meetings = args.meetings.split(",") if args.meetings else D.AMI_MEETINGS[:4]
    rows = D.import_ami(args.out, meetings, max_segments=args.max_segments, log=_log)
    n = D.write_manifest(Path(args.out) / "ami.jsonl", rows)
    _log(f"wrote {n} AMI segments to {args.out}/ami.jsonl (CC BY 4.0: keep the attribution)")
    return 0


def _cmd_import_sc(args) -> int:
    rows = D.import_speech_commands(args.out, per_word=args.per_word, log=_log)
    n = D.write_manifest(Path(args.out) / "speech_commands.jsonl", rows)
    _log(f"wrote {n} Speech Commands clips to {args.out}/speech_commands.jsonl")
    return 0


def build_teacher(args, stt=True):
    from juno_core.slu.teacher import Teacher
    from juno_core.stt import build_stt

    name = args.stt
    if not stt:
        stt = None
    elif name.startswith("parakeet"):
        stt = build_stt({"provider": "parakeet", "model": name})
    elif name.startswith("whisper:"):
        stt = build_stt({"provider": "mlx_whisper", "model": name.split(":", 1)[1]})
    else:
        raise SystemExit("--stt is parakeet-110m | parakeet-0.6b | whisper:<model>")
    stt_name = stt.name if stt is not None else "stored transcripts"
    model = None
    if args.llm:
        from juno_core.llm import build_model

        model = build_model({"provider": args.llm})
    schema = Schema.load(args.schema) if getattr(args, "schema", None) else CORE_SCHEMA
    judge = None
    if getattr(args, "judge", None):
        from juno_core.slu.judge import build_judge

        aliases = tuple(a.strip() for a in (getattr(args, "aliases", None) or "").split(",")
                        if a.strip())
        judge = build_judge(args.judge, schema, args.name, aliases)
    intent_config = {}
    if getattr(args, "config", None):
        from juno_core.config import load_config

        intent_config = (load_config(args.config).get("intent") or {}).to_dict()
    if getattr(args, "aliases", None):
        intent_config["assistant_aliases"] = [a.strip() for a in args.aliases.split(",") if a.strip()]
    teacher = Teacher(stt, schema=schema, model=model, assistant_name=args.name,
                      intent_config=intent_config, judge=judge,
                      judge_weight=getattr(args, "judge_weight", 0.8))
    return teacher, f"{stt_name} | {teacher.name}"


def _cmd_label(args) -> int:
    rows = D.read_manifest(args.manifest)
    teacher, stt_name = build_teacher(args)
    teacher.stt.warmup()
    out = []
    t0 = time.time()
    for k, row in enumerate(rows, 1):
        try:
            label = teacher.label(D.load_clip(row))
        except Exception as exc:          # one bad clip must not end the run
            _log(f"  {row.get('id')}: {type(exc).__name__}: {exc}")
            continue
        out.append({**row, **label.as_row(), "teacher": stt_name})
        if k % 200 == 0:
            _log(f"  labelled {k}/{len(rows)} ({(time.time() - t0) / k * 1000:.0f} ms/clip)")
    # Absolute paths: the labelled file usually lives somewhere else.
    n = D.write_manifest(args.out, [D.absolute_paths(row) for row in out])
    _log(f"wrote {n} labelled rows to {args.out}")
    return 0


def _cmd_relabel(args) -> int:
    """The teacher again, on transcripts already made: only the judging is redone."""
    rows = D.read_manifest(args.rows)
    teacher, name = build_teacher(args, stt=False)
    out, t0 = [], time.time()
    for k, row in enumerate(rows, 1):
        if "transcript" not in row:
            _log(f"  {row.get('id')}: no transcript (run `label` first); skipped")
            continue
        label = teacher.judge_text(row.get("transcript") or "",
                                   reliable=bool(row.get("t_reliable", True)))
        new = {key: v for key, v in row.items() if not key.startswith("t_")}
        new.update({**label.as_row(), "stt_ms": row.get("stt_ms"), "teacher": name})
        out.append(D.absolute_paths(new))
        if k % 250 == 0:
            _log(f"  judged {k}/{len(rows)} ({(time.time() - t0) / k * 1000:.0f} ms/row, "
                 f"repeats answered from memory)")
    n = D.write_manifest(args.out, out)
    _log(f"wrote {n} relabelled rows to {args.out} ({name})")
    return 0


def teacher_report(rows: Sequence[dict], weights: Sequence[float] = (0.0, 0.5, 0.8, 1.0)) -> dict:
    """Each teacher against the gold labels: the old engine (weight 0), the
    language-model judge (weight 1), and blends -- from the opinions stored
    at labelling time, so nothing is re-run."""
    rows = [r for r in rows if r.get("g_addressed") and r.get("t_judge_addressed")]
    out: dict = {"rows": len(rows), "teachers": {}}
    for w in weights:
        name = {0.0: "engine (old teacher)", 1.0: "judge alone"}.get(w, f"blend {w:g} judge")
        hits, fa, fa_n, miss, miss_n, int_ok, int_n = 0, 0, 0, 0, 0, 0, 0
        by_cat: dict = defaultdict(lambda: [0, 0])
        for r in rows:
            j, e = r["t_judge_addressed"], r["t_engine_addressed"]
            p = w * j[ASSISTANT] + (1 - w) * e[ASSISTANT]
            accept = p >= 0.5
            gold = r["g_addressed"] == ASSISTANT
            hits += accept == gold
            by_cat[r.get("category") or "?"][0] += 1
            by_cat[r.get("category") or "?"][1] += accept == gold
            if gold:
                miss_n += 1
                miss += not accept
                if accept and r.get("g_intent"):
                    int_n += 1
                    int_ok += r.get("t_intent") == r["g_intent"]
            else:
                fa_n += 1
                fa += accept
        out["teachers"][name] = {
            "who_for_accuracy": round(hits / len(rows), 4) if rows else None,
            "missed_requests": _rate(miss, miss_n), "false_activations": _rate(fa, fa_n),
            "intent_accuracy_when_accepted": round(int_ok / int_n, 4) if int_n else None,
            "by_category": {k: round(v[1] / v[0], 3) for k, v in sorted(by_cat.items())},
        }
    return out


def _cmd_teachers(args) -> int:
    _print(teacher_report(D.read_manifest(args.rows), [float(w) for w in args.weights]))
    return 0


def _cmd_embed(args) -> int:
    from juno_core.slu.encoder import build_encoder

    rows = D.read_manifest(args.rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.encoder:
        encoder = build_encoder(spec)
        encoder.warmup()
        ids, vecs, ms = [], [], []
        for k, row in enumerate(rows, 1):
            try:
                audio = D.load_clip(row)
            except Exception as exc:
                _log(f"  {row.get('id')}: {exc}")
                continue
            t0 = time.perf_counter()
            vecs.append(encoder.encode(audio))
            ms.append((time.perf_counter() - t0) * 1000.0)
            ids.append(row["id"])
            if k % 500 == 0:
                _log(f"  {encoder.spec}: {k}/{len(rows)}")
        path = out_dir / D.table_name(encoder.spec)
        np.savez(path, ids=np.asarray(ids), X=np.asarray(vecs, np.float32),
                 encode_ms=np.asarray(ms), spec=np.asarray(encoder.spec))
        _log(f"wrote {path}  ({len(ids)} vectors, dim {len(vecs[0])}, "
             f"median {np.median(ms):.1f} ms)")
    return 0


def _split(rows, args):
    keys = [k for k in args.group_by.split(",") if k]
    fractions = [float(x) for x in args.split.split(",")]
    assignment = group_split(rows, fractions, keys, seed=args.seed)
    return [[k for k, a in enumerate(assignment) if a == s] for s in range(3)]


def _trainable_rows(args, log=_log) -> list[dict]:
    rows = D.read_manifest(args.rows)
    refused = [r for r in rows if not D.row_trainable(r, args.allow_internal)]
    rows = [r for r in rows if D.row_trainable(r, args.allow_internal)]
    if refused and log:
        log(f"not training on {len(refused)} rows whose provenance forbids it "
            f"({dict(Counter(r.get('source') for r in refused))}); pass --allow-internal "
            f"for a non-distributable experimental model")
    return rows


def train_student(rows: list[dict], emb: dict, schema: Schema, args, log=_log):
    """Split, fit, threshold and report one student. Returns (model, holdout report)."""
    if args.targets in ("teacher", "mix"):
        missing = [r for r in rows if not r.get("t_addressed")]
        if missing:
            if log:
                log(f"{len(missing)} rows have no teacher labels (run `label`); dropping them")
            rows = [r for r in rows if r.get("t_addressed")]
    rows, X, enc_ms = join(rows, emb)
    train_idx, select_idx, hold_idx = _split(rows, args)
    for name, idx in (("train", train_idx), ("select", select_idx), ("holdout", hold_idx)):
        if log:
            counts = Counter(truth(rows[k], args.truth)[0] for k in idx)
            log(f"{name:8s} {len(idx):6d} rows  {dict(counts)}")
        if not idx:
            raise SystemExit(f"the {name} split is empty; add data or change --split/--group-by")
    sub = lambda idx: [rows[k] for k in idx]  # noqa: E731
    T_train = build_targets(sub(train_idx), schema, args.targets, args.gold_weight)
    T_select = build_targets(sub(select_idx), schema, args.targets, args.gold_weight)
    model = fit(X[train_idx], T_train, X[select_idx], T_select, schema=schema,
                encoder_spec=emb["spec"], hidden=args.hidden, epochs=args.epochs, lr=args.lr,
                weight_decay=args.weight_decay, dropout=args.dropout, seed=args.seed,
                log=log if getattr(args, "verbose", False) else None)
    thresholds, info = choose_thresholds(
        model, X[select_idx], sub(select_idx), schema, max_false_ignore=args.max_false_ignore,
        max_wrong_act=args.max_wrong_act, max_false_activation=args.max_false_activation,
        truth_mode=args.truth, margin_z=args.margin_z)
    model.meta["thresholds"] = thresholds.to_dict()
    model.meta["threshold_selection"] = {**info, "margin_z": args.margin_z,
                                         "max_false_ignore": args.max_false_ignore,
                                         "max_wrong_act": args.max_wrong_act,
                                         "max_false_activation": args.max_false_activation,
                                         "truth": args.truth}
    model.meta["targets"] = {"mode": args.targets, "gold_weight": args.gold_weight}
    model.meta["splits"] = {"group_by": args.group_by, "fractions": args.split, "seed": args.seed,
                            "rows": [len(train_idx), len(select_idx), len(hold_idx)],
                            "holdout_ids": [rows[k]["id"] for k in hold_idx]}
    model.meta["provenance"] = D.provenance(sub(train_idx) + sub(select_idx))
    model.meta["trained_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    model.meta["tag"] = getattr(args, "tag", None) or \
        f"slu-{args.targets}-{time.strftime('%Y%m%d%H%M')}"
    report = evaluate(model, X[hold_idx], sub(hold_idx), encode_ms=enc_ms[hold_idx],
                      schema=schema, truth_mode=args.truth)
    model.meta["metrics"] = {"holdout": report}
    return model, report


def _cmd_train(args) -> int:
    schema = Schema.load(args.schema) if args.schema else CORE_SCHEMA
    model, report = train_student(_trainable_rows(args), load_embeddings(args.embeddings),
                                  schema, args)
    model.save(args.out)
    info = model.meta["threshold_selection"]
    _print({"model": args.out, "tag": model.tag, "thresholds": model.meta["thresholds"],
            "threshold_selection": info, "holdout": _headline(report),
            "distributable": model.meta["provenance"]["distributable"]})
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        _log(f"full holdout report: {args.report}")
    fi = report["false_ignore"]["rate"] or 0.0
    fa = report["false_activation"]["rate"] or 0.0
    if fi > args.max_false_ignore or fa > args.max_false_activation:
        _log(f"\nholdout misses a budget (false ignore {fi:.2%}, false activation {fa:.2%}): "
             f"keep reflex.mode at shadow and get more (and more real) data.")
        return 2
    _log(f"\nwrote {args.out}. Run it with reflex.mode: shadow first.")
    return 0


SWEEP_COLUMNS = (
    "encoder", "targets", "hidden", "seed", "rows_holdout", "stt_avoided", "act", "ignore",
    "escalate", "false_ignore", "false_ignore_up95", "false_activation", "false_activation_up95",
    "wrong_act", "typed_acted", "addressed_acc", "addressed_auc", "addressed_ece",
    "intent_acc", "slot_acc", "hybrid_correct", "teacher_correct", "encode_ms_p50",
    "real_addressed_acc", "synthetic_addressed_acc",
)


def _sweep_row(spec, targets, hidden, seed, r: dict) -> dict:
    def g(*path):
        node = r
        for key in path:
            node = (node or {}).get(key) if isinstance(node, dict) else None
        return node

    def acc_for(sources):
        n = c = 0
        for name, b in (r.get("by_source") or {}).items():
            if name in sources:
                n += b["n"]
                c += b["addressed_accuracy"] * b["n"]
        return round(c / n, 4) if n else None

    return {
        "encoder": spec, "targets": targets, "hidden": hidden, "seed": seed,
        "rows_holdout": r.get("rows"), "stt_avoided": r.get("stt_avoided"),
        "act": g("routes", "act"), "ignore": g("routes", "ignore"),
        "escalate": g("routes", "escalate"),
        "false_ignore": g("false_ignore", "rate"), "false_ignore_up95": g("false_ignore", "upper95"),
        "false_activation": g("false_activation", "rate"),
        "false_activation_up95": g("false_activation", "upper95"),
        "wrong_act": g("wrong_act", "rate"), "typed_acted": g("typed_commands_acted", "rate"),
        "addressed_acc": r.get("addressed_accuracy"), "addressed_auc": r.get("addressed_auc"),
        "addressed_ece": r.get("addressed_ece"), "intent_acc": r.get("intent_accuracy"),
        "slot_acc": r.get("slot_accuracy"), "hybrid_correct": g("end_to_end", "hybrid_correct"),
        "teacher_correct": g("end_to_end", "teacher_only_correct"),
        "encode_ms_p50": g("encode_ms", "p50"),
        "real_addressed_acc": acc_for({"ami", "speech_commands", "consented"}),
        "synthetic_addressed_acc": acc_for({"synthetic_say", "noise"}),
    }


def _cmd_sweep(args) -> int:
    """Every (encoder, targets, hidden, seed) combination, one table."""
    import copy
    import csv

    schema = Schema.load(args.schema) if args.schema else CORE_SCHEMA
    rows = _trainable_rows(args)
    tables: dict[str, list[Path]] = defaultdict(list)
    for d in args.emb_dirs:
        for path in sorted(Path(d).glob("*.npz")):
            tables[path.name].append(path)
    wanted = set(args.encoders or [])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, paths in sorted(tables.items()):
        emb = load_embeddings(paths)
        if wanted and not any(w in emb["spec"] for w in wanted):
            continue
        for targets, hidden, seed in itertools.product(args.targets, args.hidden, args.seeds):
            a = copy.copy(args)
            a.targets, a.hidden, a.seed = targets, hidden, seed
            a.tag = f"{emb['spec']}|{targets}|h{hidden}|s{seed}"
            t0 = time.time()
            model, report = train_student(rows, emb, schema, a, log=None)
            row = _sweep_row(emb["spec"], targets, hidden, seed, report)
            results.append(row)
            stem = f"{name[:-4]}__{targets}__h{hidden}__s{seed}"
            if args.save_models:
                model.save(out_dir / f"{stem}.npz")
            (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=1, default=str),
                                                 encoding="utf-8")
            _log(f"{a.tag:70s} avoided {row['stt_avoided']}  false-ign {row['false_ignore']}  "
                 f"false-act {row['false_activation']}  hybrid {row['hybrid_correct']}  "
                 f"({time.time() - t0:.0f}s)")
    with open(out_dir / "sweep.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SWEEP_COLUMNS))
        writer.writeheader()
        writer.writerows(results)
    _write_sweep_markdown(out_dir / "sweep.md", results)
    _log(f"wrote {out_dir}/sweep.csv and sweep.md ({len(results)} runs)")
    return 0


def _write_sweep_markdown(path: Path, results: list[dict]) -> None:
    """Mean ± sd over seeds, one line per configuration."""
    groups: dict = defaultdict(list)
    for r in results:
        groups[(r["encoder"], r["targets"], r["hidden"])].append(r)
    cols = ("stt_avoided", "false_ignore", "false_activation", "wrong_act", "typed_acted",
            "addressed_auc", "intent_acc", "hybrid_correct", "teacher_correct",
            "real_addressed_acc", "encode_ms_p50")
    lines = ["| encoder | targets | hidden | n | " + " | ".join(cols) + " |",
             "|" + "---|" * (4 + len(cols))]
    for (enc, targets, hidden), rs in sorted(groups.items()):
        cells = []
        for c in cols:
            vals = [r[c] for r in rs if r[c] is not None]
            if not vals:
                cells.append("–")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.3f}")
            else:
                cells.append(f"{np.mean(vals):.3f} ± {np.std(vals):.3f}")
        short = enc.replace("mlx-community/", "")
        lines.append(f"| {short} | {targets} | {hidden} | {len(rs)} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _headline(report: dict) -> dict:
    keys = ("rows", "routes", "stt_avoided", "false_ignore", "false_activation", "wrong_act",
            "typed_commands_acted", "addressed_accuracy", "addressed_auc", "intent_accuracy",
            "slot_accuracy", "end_to_end", "encode_ms")
    return {k: report.get(k) for k in keys if k in report}


def _cmd_evaluate(args) -> int:
    model = StudentModel.load(args.model)
    rows = D.read_manifest(args.rows)
    if args.only_holdout:
        keep = set((model.meta.get("splits") or {}).get("holdout_ids") or [])
        rows = [r for r in rows if r.get("id") in keep]
    emb = load_embeddings(args.embeddings)
    if emb["spec"] != model.encoder_spec:
        raise SystemExit(f"embeddings are {emb['spec']}, model wants {model.encoder_spec}")
    rows, X, enc_ms = join(rows, emb)
    state = ConversationState(awaiting_answer=args.awaiting_answer, since_ai=float("inf"))
    report = evaluate(model, X, rows, encode_ms=enc_ms, truth_mode=args.truth, state=state)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        _log(f"wrote {args.out}")
    _print(report if args.full else _headline(report))
    return 0


def _cmd_collect(args) -> int:
    from juno_core.slu.collect import collect_main

    return collect_main(args)


def _cmd_studio(args) -> int:
    from juno_core.slu.studio.server import studio_main

    return studio_main(args)


def _cmd_bench(args) -> int:
    from juno_core.slu.bench import bench_main

    return bench_main(args)


def _cmd_shadow(args) -> int:
    from juno_core.slu.bench import shadow_report

    _print(shadow_report(args.log))
    return 0


def _judge_args(s) -> None:
    s.add_argument("--judge", help="a language-model teacher: mlx (local Qwen3.5-4B), "
                                   "mlx:<repo>, or an llm.provider name")
    s.add_argument("--judge-weight", type=float, default=0.8,
                   help="how much the judge counts against the engine's verdict (1 = judge only)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m juno_core.slu",
                                     description="Train and measure Reflex (audio -> typed decision).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sources", help="registered data sources and whether models trained on them ship")

    s = sub.add_parser("schema", help="print (or write) the core schema, optionally extended")
    s.add_argument("--out")
    s.add_argument("--extend", help="an agent schema JSON to merge onto the core one")

    s = sub.add_parser("synth", help="voice the scripted corpus with TTS -> clips + manifest")
    s.add_argument("--out", required=True)
    s.add_argument("--size", type=int, default=3000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--augment", type=int, default=1, help="augmented copies per clean clip")
    s.add_argument("--no-clean", action="store_true", help="keep only augmented copies")
    s.add_argument("--noise", type=int, default=150, help="non-speech clips to add")
    s.add_argument("--voices", help="comma-separated `say` voices (default: all English)")
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--schema", help="agent schema JSON (default: core)")
    s.add_argument("--name", default="Juno")
    s.add_argument("--mix", help="category shares, e.g. typed=0.5,open=0.15,human=0.15,"
                                 "hard_negative=0.1,background=0.1")

    s = sub.add_parser("import-ami", help="far-field AMI meetings -> human-directed clips")
    s.add_argument("--out", required=True)
    s.add_argument("--meetings", help=f"comma-separated (default: {','.join(D.AMI_MEETINGS[:4])})")
    s.add_argument("--max-segments", type=int, default=150)

    s = sub.add_parser("import-speech-commands", help="real 'stop'/'yes'/'no' clips")
    s.add_argument("--out", required=True)
    s.add_argument("--per-word", type=int, default=120)

    s = sub.add_parser("label", help="run the teacher (the cascade) over clips")
    s.add_argument("--manifest", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--stt", default="parakeet-0.6b",
                   help="parakeet-110m | parakeet-0.6b | whisper:<model> (e.g. whisper:large-v3-turbo)")
    s.add_argument("--llm", help="provider for the adjudicator (openai | anthropic | gemini | ollama)")
    s.add_argument("--schema")
    s.add_argument("--name", default="Juno")
    s.add_argument("--aliases", help="other spellings the recogniser produces for the name, "
                                     "e.g. Jumo,June -- same as intent.assistant_aliases")
    s.add_argument("--config", help="config.yaml whose intent: section the teacher uses")
    _judge_args(s)

    s = sub.add_parser("relabel", help="re-judge rows that already have transcripts "
                                       "(no speech-to-text): e.g. with a better teacher")
    s.add_argument("--rows", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--stt", default="parakeet-0.6b", help=argparse.SUPPRESS)
    s.add_argument("--llm", help="adjudicator provider for the engine half")
    s.add_argument("--schema")
    s.add_argument("--name", default="Juno")
    s.add_argument("--aliases")
    s.add_argument("--config")
    _judge_args(s)

    s = sub.add_parser("teachers", help="score the old teacher, the judge, and blends "
                                        "against gold labels (rows from relabel --judge)")
    s.add_argument("--rows", nargs="+", required=True)
    s.add_argument("--weights", nargs="+", default=["0", "0.5", "0.8", "1"])

    s = sub.add_parser("embed", help="pooled encoder vectors for every clip")
    s.add_argument("--rows", nargs="+", required=True)
    s.add_argument("--encoder", nargs="+", default=["parakeet"],
                   help="parakeet | whisper | logmel | gate | full spec, e.g. "
                        "parakeet:mlx-community/parakeet-tdt_ctc-110m@L8:stats")
    s.add_argument("--out", required=True)

    s = sub.add_parser("train", help="fit, calibrate and threshold a student")
    s.add_argument("--rows", nargs="+", required=True)
    s.add_argument("--embeddings", nargs="+", required=True,
                   help="vector tables from `embed` (and `collect`), all from one encoder")
    s.add_argument("--out", required=True)
    s.add_argument("--schema")
    s.add_argument("--targets", choices=("teacher", "gold", "mix"), default="teacher")
    s.add_argument("--gold-weight", type=float, default=0.5)
    s.add_argument("--truth", choices=("auto", "gold", "teacher"), default="auto")
    s.add_argument("--max-false-ignore", type=float, default=0.01)
    s.add_argument("--max-wrong-act", type=float, default=0.01)
    s.add_argument("--max-false-activation", type=float, default=0.01)
    s.add_argument("--margin-z", type=float, default=1.0,
                   help="apply act budgets to a Wilson upper bound at z std errors (0 = point)")
    s.add_argument("--hidden", type=int, default=256, help="0 = linear heads")
    s.add_argument("--epochs", type=int, default=80)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--weight-decay", type=float, default=1e-4)
    s.add_argument("--dropout", type=float, default=0.2)
    s.add_argument("--split", default="0.6,0.2,0.2")
    s.add_argument("--group-by", default="speaker")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--allow-internal", action="store_true",
                   help="train on non-distributable rows (synthetic_say, consent=internal)")
    s.add_argument("--tag")
    s.add_argument("--report", help="write the full holdout report here")
    s.add_argument("--verbose", action="store_true")

    s = sub.add_parser("sweep", help="train every encoder x targets x hidden x seed; one table")
    s.add_argument("--rows", nargs="+", required=True)
    s.add_argument("--emb-dirs", nargs="+", required=True,
                   help="directories of vector tables; same-named files are joined")
    s.add_argument("--encoders", nargs="*", help="substrings of encoder specs to include")
    s.add_argument("--targets", nargs="+", default=["teacher", "mix", "gold"])
    s.add_argument("--hidden", nargs="+", type=int, default=[256])
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    s.add_argument("--out", required=True, help="directory for sweep.csv, sweep.md, reports")
    s.add_argument("--save-models", action="store_true")
    s.add_argument("--schema")
    s.add_argument("--gold-weight", type=float, default=0.5)
    s.add_argument("--truth", choices=("auto", "gold", "teacher"), default="auto")
    s.add_argument("--max-false-ignore", type=float, default=0.01)
    s.add_argument("--max-wrong-act", type=float, default=0.01)
    s.add_argument("--max-false-activation", type=float, default=0.01)
    s.add_argument("--margin-z", type=float, default=1.0)
    s.add_argument("--epochs", type=int, default=80)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--weight-decay", type=float, default=1e-4)
    s.add_argument("--dropout", type=float, default=0.2)
    s.add_argument("--split", default="0.6,0.2,0.2")
    s.add_argument("--group-by", default="speaker")
    s.add_argument("--allow-internal", action="store_true")

    s = sub.add_parser("evaluate", help="report a trained student on rows")
    s.add_argument("--rows", nargs="+", required=True)
    s.add_argument("--embeddings", nargs="+", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--truth", choices=("auto", "gold", "teacher"), default="auto")
    s.add_argument("--only-holdout", action="store_true", help="only the model's own holdout ids")
    s.add_argument("--awaiting-answer", action="store_true",
                   help="judge as if the assistant had just asked a question")
    s.add_argument("--full", action="store_true")
    s.add_argument("--out")

    s = sub.add_parser("collect", help="guided, consented live session -> vectors + teacher "
                                       "labels (no audio is saved)")
    s.add_argument("--out", required=True, help="a new directory for this session")
    s.add_argument("--speakers", required=True, help="pseudonyms, e.g. p1,p2")
    s.add_argument("--room", required=True)
    s.add_argument("--consent", required=True, choices=("release", "internal"))
    s.add_argument("--encoder", nargs="+", default=["parakeet", "logmel"])
    s.add_argument("--stt", default="parakeet-0.6b")
    s.add_argument("--llm")
    s.add_argument("--aliases")
    s.add_argument("--name", default="Juno")
    s.add_argument("--session")
    s.add_argument("--mic")
    s.add_argument("--config")
    s.add_argument("--scale", type=float, default=1.0)
    _judge_args(s)

    s = sub.add_parser("studio", help="a local web page to try Reflex and collect data")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--stt", default="parakeet-0.6b", help="the cascade's recogniser")
    s.add_argument("--model", help="student to load (default: the newest in models/)")
    s.add_argument("--encoder", nargs="+", help="encoders to keep vectors from "
                                               "(default: parakeet logmel gate)")
    s.add_argument("--simulate", nargs="+", help="WAV files or folders to play instead of the mic")
    s.add_argument("--judge", help="give the cascade the language-model judge: mlx, mlx:<repo>, "
                                   "or an llm.provider")
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--config")
    s.add_argument("--aliases")
    s.add_argument("--name", default="Juno")

    s = sub.add_parser("bench", help="live benchmark: Reflex vs always-transcribe baselines")
    from juno_core.slu.bench import add_bench_args

    add_bench_args(s)

    s = sub.add_parser("shadow", help="compare Reflex with the cascade in a shadow log")
    s.add_argument("--log", required=True)

    args = parser.parse_args(argv)
    return {"sources": _cmd_sources, "schema": _cmd_schema, "synth": _cmd_synth,
            "import-ami": _cmd_import_ami, "import-speech-commands": _cmd_import_sc,
            "label": _cmd_label, "relabel": _cmd_relabel, "teachers": _cmd_teachers,
            "embed": _cmd_embed, "train": _cmd_train,
            "evaluate": _cmd_evaluate, "sweep": _cmd_sweep, "collect": _cmd_collect,
            "studio": _cmd_studio, "bench": _cmd_bench,
            "shadow": _cmd_shadow}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
