"""Teaching the pre-transcription gate, offline, from data it may learn from.

    python -m juno_core.intelligence.gate_training sources
    python -m juno_core.intelligence.gate_training collect  --out s1.csv --speakers p1,p2 \
                                                            --room kitchen --consent release
    python -m juno_core.intelligence.gate_training features --manifest m.csv --out rows.csv
    python -m juno_core.intelligence.gate_training train    --rows rows.csv --out model.json
    python -m juno_core.intelligence.gate_training evaluate --rows unseen.csv --model model.json
    python -m juno_core.intelligence.gate_training shadow   --log logs/juno.jsonl

WHAT IT MAY LEARN FROM
----------------------
This project is open source, and so is anything trained with this file that
ships with it. So every row must come from a source whose licence allows
that, and that is enforced, not suggested:

  - public corpora with open licences (``sources`` lists the registered ones
    -- AMI is CC BY 4.0, Common Voice CC0, Speech Commands CC BY 4.0);
  - ``custom`` sources, only with an explicit open licence in the manifest;
  - ``consented`` recordings -- from ``collect`` (a guided live session that
    keeps only feature rows; see gate_collect.py) or from a manifest of
    clips -- only with a consent scope: ``release``
    (the participants agreed to derived models and feature tables being
    published) or ``internal`` (evaluation only -- a model trained on it is
    marked non-distributable, and only with --allow-internal).

Non-commercial or share-alike licences are refused. The shadow log -- numbers
from ordinary use, which includes bystanders who agreed to nothing -- can be
evaluated against but never trained on.

The model file carries its provenance: every source, its licence and the
attribution it requires. Anyone who redistributes a model must keep that.

WHAT IT KEEPS
-------------
Feature rows: the 20 numbers features.py produces, a label, and provenance
(source, licence, speaker, session, room, mic). Never audio. Clips are read
into memory, reduced to features, and released; ``--delete-source`` removes
a consented clip once its row is written. Use pseudonymous speaker IDs.

HOW IT DECIDES
--------------
The target is p(assistant_directed). Splits are by speaker, session and room
together (connected components, so no speaker, session or room appears in
two splits) into train / select / holdout. A logistic model is fitted on
train, calibrated (Platt) on select, and its skip threshold is the largest
one whose false-skip rate on select -- over assistant_directed and, unless
--ignore-uncertain, uncertain rows -- stays within --max-false-skip. Holdout is
touched once, to report. Every number comes from the runtime Gate itself, fed
the stored feature vector -- so the evaluation is of the code that runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from juno_core.intelligence.features import FEATURE_NAMES, extract_gate_features
from juno_core.intelligence.gate import NEVER_SPOKEN, Acoustics, Gate, Snapshot

LABELS = ("assistant_directed", "human_directed", "background_or_media", "uncertain")
POSITIVE = "assistant_directed"
SKIPPABLE = ("human_directed", "background_or_media")
META = ("label", "source", "license", "consent", "speaker", "session", "room",
        "mic", "feature_ms", "intent_verdict")
GROUP_KEYS = ("speaker", "session", "room", "mic")

# Licences a model can be trained on and still be published with a
# permissively licensed project, with attribution. Deliberately short: no NC
# (the project's licence allows commercial use),
# no SA (whether weights are a "derivative" is unsettled -- not worth the risk).
OPEN_LICENSES = frozenset({
    "CC0-1.0", "CC-BY-4.0", "CC-BY-3.0", "PDDL-1.0", "ODC-By-1.0",
    "CDLA-Permissive-2.0", "MIT", "Apache-2.0",
})

# Licences as published by each project at the time of writing. Check the
# dataset's own page when you download it; terms can change between releases.
SOURCES = {
    "ami": {
        "name": "AMI Meeting Corpus",
        "license": "CC-BY-4.0",
        "url": "https://groups.inf.ed.ac.uk/ami/corpus/",
        "attribution": "AMI Meeting Corpus, AMI Consortium, CC BY 4.0",
    },
    "common_voice": {
        "name": "Mozilla Common Voice",
        "license": "CC0-1.0",
        "url": "https://commonvoice.mozilla.org/",
        "attribution": "Mozilla Common Voice, CC0 1.0 (do not attempt to "
                       "identify speakers)",
    },
    "speech_commands": {
        "name": "Google Speech Commands v0.02",
        "license": "CC-BY-4.0",
        "url": "https://arxiv.org/abs/1804.03209",
        "attribution": "Speech Commands v0.02, P. Warden (Google), CC BY 4.0",
    },
    "consented": {
        "name": "Consented Juno recordings",
        "license": "consent",
        "url": "",
        "attribution": "",
    },
    "custom": {
        "name": "Other openly licensed data (licence column required)",
        "license": None,
        "url": "",
        "attribution": "",
    },
    "shadow_log": {
        "name": "Juno shadow-mode event log (evaluation only)",
        "license": "none",
        "url": "",
        "attribution": "",
    },
}
CONSENT_SCOPES = ("release", "internal")


class DataPolicyError(ValueError):
    """A row whose provenance does not allow what is being done with it."""


# -- provenance -------------------------------------------------------------

def resolve_provenance(row: dict) -> tuple[str, str, str]:
    """(source, licence, consent) for a manifest row, or DataPolicyError."""
    source = (row.get("source") or "").strip().lower()
    if source not in SOURCES:
        raise DataPolicyError(
            f"unknown source {source!r}; registered: {', '.join(sorted(SOURCES))}")
    if source == "shadow_log":
        raise DataPolicyError("shadow-log rows come from `shadow --export`, not a manifest")
    consent = (row.get("consent") or "").strip().lower()
    if source == "consented":
        if consent not in CONSENT_SCOPES:
            raise DataPolicyError(
                "consented rows need consent=release (participants agreed to "
                "published derived models) or consent=internal (evaluation only)")
        return source, "consent", consent
    registered = SOURCES[source]["license"]
    license_ = (row.get("license") or "").strip() or registered
    if registered and license_ != registered:
        raise DataPolicyError(
            f"{source} is {registered}; the manifest says {license_!r}")
    if license_ not in OPEN_LICENSES:
        raise DataPolicyError(
            f"licence {license_!r} is not one an open-source model can be "
            f"trained on (allowed: {', '.join(sorted(OPEN_LICENSES))})")
    return source, license_, ""


def trainable(row: dict, allow_internal: bool = False) -> bool:
    source = row.get("source", "")
    if source == "shadow_log":
        return False
    if source == "consented":
        return row.get("consent") == "release" or (
            allow_internal and row.get("consent") == "internal")
    return row.get("license") in OPEN_LICENSES


def provenance_summary(rows: Sequence[dict]) -> dict:
    counts = Counter((r["source"], r.get("license", ""), r.get("consent", "")) for r in rows)
    sources = []
    for (source, license_, consent), n in sorted(counts.items()):
        info = SOURCES.get(source, {})
        sources.append({
            "source": source, "name": info.get("name", source),
            "license": license_, "consent": consent or None, "rows": n,
            "url": info.get("url", ""), "attribution": info.get("attribution", ""),
        })
    distributable = all(
        s["license"] in OPEN_LICENSES or s["consent"] == "release" for s in sources)
    return {"sources": sources, "distributable": distributable}


# -- audio in (memory only) -------------------------------------------------

def read_wav(path: str | Path, start: float | None = None,
             end: float | None = None) -> tuple[np.ndarray, int]:
    """Mono float32 PCM from an integer-PCM WAV, optionally a slice of it."""
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        total = handle.getnframes()
        first = max(0, int(round(start * rate))) if start else 0
        last = min(total, int(round(end * rate))) if end else total
        handle.setpos(min(first, total))
        raw = handle.readframes(max(0, last - first))
    if width == 2:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        pcm = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"{path}: {8 * width}-bit WAV is not supported")
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, rate


def vad_confidence(audio: np.ndarray, backend, threshold: float = 0.5,
                   frame: int = 512) -> float:
    """Mean probability over voiced frames -- what SegmentDetector reports."""
    backend.reset()
    probs = [backend.probability(audio[i : i + frame])
             for i in range(0, audio.size - frame + 1, frame)]
    voiced = [p for p in probs if p >= threshold]
    return float(np.mean(voiced)) if voiced else 0.0


# -- manifest rows <-> the runtime's inputs ---------------------------------

def _flag(value, default: bool = False) -> bool:
    text = str(value).strip().lower() if value is not None else ""
    if not text:
        return default
    return text in ("1", "true", "yes", "y")


def _num(value, default: float | None = None) -> float | None:
    text = str(value).strip() if value is not None else ""
    return float(text) if text else default


def context_from_manifest(row: dict, seconds: float,
                          confidence: float) -> tuple[Acoustics, Snapshot]:
    """Build the runtime's inputs from manifest columns.

    Missing context means "unknown", modelled as the runtime would see a cold
    start: the assistant has not spoken, nothing is pending, no voiceprint.
    Fill these in for consented sessions -- a model only learns context it
    has been shown.
    """
    voice_confident = _flag(row.get("voice_confident"))
    wearer_text = str(row.get("is_wearer") or "").strip()
    is_wearer = _flag(wearer_text) if wearer_text and voice_confident else None
    run = int(_num(row.get("ignored_run"), 0) or 0)
    if run:
        verdicts: tuple = (False,) * run
    elif _flag(row.get("accepted_recently")):
        verdicts = (True,)
    else:
        verdicts = ()
    acoustics = Acoustics(
        seconds=seconds, confidence=confidence,
        p_own=_num(row.get("p_own")) if voice_confident else None,
        voice_confident=voice_confident, is_wearer=is_wearer,
    )
    snapshot = Snapshot(
        since_ai=_num(row.get("since_ai"), NEVER_SPOKEN),
        awaiting_answer=_flag(row.get("awaiting_answer")),
        confirming=_flag(row.get("confirming")),
        offer_pending=_flag(row.get("offer_pending")),
        recent_verdicts=verdicts,
    )
    return acoustics, snapshot


def context_from_features(vec: np.ndarray) -> tuple[Acoustics, Snapshot]:
    """Invert extract_gate_features' context columns, for evaluation."""
    f = dict(zip(FEATURE_NAMES, (float(v) for v in vec)))
    voice_confident = f["voice_confident"] >= 0.5
    is_wearer = None if abs(f["is_wearer"] - 0.5) < 1e-6 else f["is_wearer"] >= 0.5
    run = int(round(f["ignored_run"]))
    if run:
        verdicts: tuple = (False,) * run
    elif f["accepted_recently"] >= 0.5:
        verdicts = (True,)
    else:
        verdicts = ()
    acoustics = Acoustics(
        seconds=f["duration"], confidence=f["vad_confidence"],
        p_own=f["p_own"] if voice_confident else None,
        voice_confident=voice_confident,
        is_wearer=is_wearer if voice_confident else None,
    )
    snapshot = Snapshot(
        since_ai=math.expm1(f["since_ai_log"]),
        awaiting_answer=f["awaiting_answer"] >= 0.5,
        confirming=f["confirming"] >= 0.5,
        offer_pending=f["offer_pending"] >= 0.5,
        recent_verdicts=verdicts,
    )
    return acoustics, snapshot


def build_rows(manifest: Iterable[dict], *, vad_backend=None, rate: int = 16000,
               base_dir: Path | None = None, delete_source: bool = False,
               on_error: Callable[[int, str], None] | None = None) -> list[dict]:
    """Manifest rows in, feature rows out. Audio never leaves memory."""
    rows: list[dict] = []
    manifest = list(manifest)
    pending_delete: dict[Path, int] = Counter()
    for entry in manifest:
        if (entry.get("source") or "").strip().lower() == "consented":
            pending_delete[_resolve(entry["path"], base_dir)] += 1
    failed: set[Path] = set()

    for i, entry in enumerate(manifest, start=1):
        path = _resolve(entry.get("path", ""), base_dir)
        try:
            label = (entry.get("label") or "").strip()
            if label not in LABELS:
                raise ValueError(f"label must be one of {LABELS}, not {label!r}")
            source, license_, consent = resolve_provenance(entry)
            audio, file_rate = read_wav(path, _num(entry.get("start")), _num(entry.get("end")))
            if file_rate != rate:
                raise ValueError(
                    f"{file_rate} Hz; convert first: ffmpeg -i IN -ac 1 -ar {rate} OUT.wav")
            if audio.size == 0:
                raise ValueError("empty clip")
            confidence = _num(entry.get("vad_confidence"))
            if confidence is None:
                confidence = vad_confidence(audio, vad_backend) if vad_backend else 1.0
            acoustics, snapshot = context_from_manifest(
                entry, audio.size / float(rate), confidence)
            t0 = time.perf_counter()
            vec = extract_gate_features(audio, rate, acoustics, snapshot)
            feature_ms = (time.perf_counter() - t0) * 1000.0
            del audio
        except (DataPolicyError, ValueError, OSError, wave.Error, EOFError) as exc:
            failed.add(path)
            if on_error is not None:
                on_error(i, f"{entry.get('path', '')}: {exc}")
            continue
        row = {
            "label": label, "source": source, "license": license_, "consent": consent,
            "feature_ms": round(feature_ms, 3), "intent_verdict": "",
            **{k: (entry.get(k) or "").strip() for k in ("speaker", "session", "room", "mic")},
        }
        row.update({name: float(v) for name, v in zip(FEATURE_NAMES, vec)})
        rows.append(row)

        if delete_source and source == "consented":
            pending_delete[path] -= 1
            if pending_delete[path] == 0 and path not in failed:
                path.unlink(missing_ok=True)
    return rows


def _resolve(path: str, base_dir: Path | None) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() or base_dir is None else base_dir / p


# -- feature tables ---------------------------------------------------------

def write_rows(path: str | Path, rows: Sequence[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(META) + list(FEATURE_NAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})


def read_rows(paths: Sequence[str | Path]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                row = {k: (raw.get(k) or "").strip() for k in META}
                missing = [n for n in FEATURE_NAMES if not (raw.get(n) or "").strip()]
                if missing:
                    raise ValueError(f"{path}: row without {missing[0]}")
                row.update({n: float(raw[n]) for n in FEATURE_NAMES})
                rows.append(row)
    return rows


def matrix(rows: Sequence[dict]) -> np.ndarray:
    return np.asarray([[r[n] for n in FEATURE_NAMES] for r in rows], dtype=np.float64)


# -- splitting without leakage ----------------------------------------------

def group_split(rows: Sequence[dict], fractions: Sequence[float] = (0.6, 0.2, 0.2),
                keys: Sequence[str] = ("speaker", "session", "room"),
                seed: int = 0) -> list[int]:
    """Assign each row a split index so no key value spans two splits.

    Rows sharing any of ``keys`` (blank values excepted) are joined into one
    component; components are dealt, largest first, to whichever split is
    furthest below its share.
    """
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_seen: dict[tuple[str, str], int] = {}
    for i, row in enumerate(rows):
        for key, value in _key_values(row, keys):
            j = first_seen.setdefault((key, value), i)
            parent[find(i)] = find(j)

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        components[find(i)].append(i)
    groups = list(components.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    total = float(len(rows))
    targets = [f * total for f in fractions]
    filled = [0] * len(fractions)
    assignment = [0] * len(rows)
    for group in groups:
        k = max(range(len(fractions)),
                key=lambda s: (targets[s] - filled[s]) / max(targets[s], 1e-9))
        filled[k] += len(group)
        for i in group:
            assignment[i] = k
    return assignment


def _key_values(row: dict, keys: Sequence[str]):
    """(key, value) pairs a row belongs to. A conversation's speaker is
    written "p1+p2" and belongs to both p1 and p2."""
    for key in keys:
        value = row.get(key, "")
        parts = value.split("+") if key == "speaker" else [value]
        for part in parts:
            if part:
                yield key, part


def check_disjoint(rows: Sequence[dict], assignment: Sequence[int],
                   keys: Sequence[str]) -> list[str]:
    """Key values that appear in more than one split (should be none)."""
    seen: dict[tuple[str, str], set] = defaultdict(set)
    for row, split in zip(rows, assignment):
        for pair in _key_values(row, keys):
            seen[pair].add(split)
    return [f"{k}={v}" for (k, v), s in seen.items() if len(s) > 1]


# -- the model --------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                 iters: int = 100) -> tuple[np.ndarray, float]:
    """L2-regularised logistic regression by Newton's method (intercept free)."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    reg = np.full(d + 1, float(l2))
    reg[-1] = 0.0
    for _ in range(iters):
        p = _sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) + reg * w
        hess = (Xb * (p * (1.0 - p))[:, None]).T @ Xb + np.diag(reg + 1e-9)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w[:-1], float(w[-1])


def train_model(train_rows: Sequence[dict], calib_rows: Sequence[dict],
                l2: float = 1.0) -> dict:
    """Fit on train, Platt-calibrate on calib, folded into one linear model."""
    fit_rows = [r for r in train_rows if r["label"] != "uncertain"]
    X = matrix(fit_rows)
    y = np.asarray([r["label"] == POSITIVE for r in fit_rows], dtype=np.float64)
    if y.sum() == 0 or y.sum() == y.size:
        raise ValueError("training split needs both assistant_directed and other rows")
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    weights, bias = fit_logistic((X - mean) / scale, y, l2=l2)

    calibration = {"a": 1.0, "c": 0.0}
    cal = [r for r in calib_rows if r["label"] != "uncertain"]
    if cal:
        yc = np.asarray([r["label"] == POSITIVE for r in cal], dtype=np.float64)
        if 0 < yc.sum() < yc.size:
            z = ((matrix(cal) - mean) / scale) @ weights + bias
            (a,), c = fit_logistic(z[:, None], yc, l2=1e-3)
            if a > 0:                      # never let calibration invert the ranking
                weights, bias = weights * a, a * bias + c
                calibration = {"a": round(float(a), 6), "c": round(float(c), 6)}
    return {
        "format": "juno-gate-logistic/1",
        "feature_names": list(FEATURE_NAMES),
        "weights": [round(float(v), 8) for v in weights],
        "bias": round(float(bias), 8),
        "scaler_mean": [round(float(v), 8) for v in mean],
        "scaler_scale": [round(float(v), 8) for v in scale],
        "calibration": calibration,
        "threshold": None,
    }


# -- judging, with the runtime gate ------------------------------------------

def make_gate(model: dict | None = None, *, learned: bool = True,
              protect_wearer: bool = True) -> Gate:
    gate = Gate({"mode": "shadow", "learned": learned and model is not None,
                 "always_transcribe_wearer": protect_wearer,
                 "model_path": "/dev/null/no-model"})
    if model is not None:
        gate.load_model(model)
    return gate


def score_rows(rows: Sequence[dict], gate: Gate) -> list:
    """The runtime Decision for each row, from its stored feature vector."""
    out = []
    for row in rows:
        vec = np.asarray([row[n] for n in FEATURE_NAMES], dtype=np.float64)
        acoustics, snapshot = context_from_features(vec)
        out.append(gate.score(acoustics, snapshot, features=vec))
    return out


def eligibility(rows: Sequence[dict], model: dict,
                protect_wearer: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(p, may_skip) per row: the model's score and whether any runtime rule
    forbids skipping regardless of it. Found by asking the gate with a
    threshold nothing sits above."""
    probe = dict(model, threshold=2.0)
    decisions = score_rows(rows, make_gate(probe, protect_wearer=protect_wearer))
    p = np.asarray([d.learned_confidence if d.learned_confidence is not None else 1.0
                    for d in decisions])
    may_skip = np.asarray([d.learned_would_skip for d in decisions])
    return p, may_skip


def choose_threshold(p: np.ndarray, may_skip: np.ndarray, positive: np.ndarray,
                     max_false_skip: float, ceiling: float = 0.5) -> float:
    """Largest threshold whose false-skip rate on these rows is within budget.

    Skips are ``p < threshold``, so setting it AT the (k+1)-th lowest skippable
    positive skips at most k of them. Capped at ``ceiling``: never skip what
    the model thinks is more likely for the assistant than not.
    """
    n_pos = int(positive.sum())
    if n_pos == 0:
        raise ValueError("no assistant_directed rows to choose a threshold on")
    budget = int(math.floor(max_false_skip * n_pos + 1e-9))
    exposed = np.sort(p[positive & may_skip])
    if exposed.size <= budget:
        return float(ceiling)
    return float(min(exposed[budget], ceiling))


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    """95% upper bound on a rate seen as k of n (0 of n is not 'zero')."""
    if n == 0:
        return 1.0
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return min(1.0, (centre + margin) / denom)


def report(rows: Sequence[dict], decisions: Sequence, model: dict | None = None) -> dict:
    """False-skip rate first; per-class skip rates; worst group; latency."""
    skipped = np.asarray([d.would_skip for d in decisions])
    labels = np.asarray([r["label"] for r in rows])
    out: dict = {"rows": len(rows), "classes": {}}
    for label in LABELS:
        mask = labels == label
        n = int(mask.sum())
        k = int(skipped[mask].sum())
        out["classes"][label] = {"n": n, "skipped": k,
                                 "skip_rate": round(k / n, 4) if n else None}
    pos = labels == POSITIVE
    n_pos, k_pos = int(pos.sum()), int(skipped[pos].sum())
    out["false_skip_rate"] = round(k_pos / n_pos, 4) if n_pos else None
    out["false_skip_upper95"] = round(wilson_upper(k_pos, n_pos), 4)
    out["asked_a_question_rows"] = int(sum(
        1 for r in rows if r["label"] == POSITIVE and r["awaiting_answer"] >= 0.5))

    groups: dict = {}
    for key in GROUP_KEYS:
        per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row, s in zip(rows, skipped):
            if row["label"] == POSITIVE and row.get(key):
                per[row[key]][0] += 1
                per[row[key]][1] += int(s)
        if per:
            worst = max(per.items(), key=lambda kv: (kv[1][1] / kv[1][0], kv[1][0]))
            groups[key] = {"groups": len(per), "worst": worst[0],
                           "worst_false_skip_rate": round(worst[1][1] / worst[1][0], 4),
                           "worst_n": worst[1][0]}
    out["by_group"] = groups

    feature_ms = [float(r["feature_ms"]) for r in rows if r.get("feature_ms")]
    gate_ms = [d.latency * 1000.0 for d in decisions]
    out["latency_ms"] = {
        "feature_extraction_median": round(float(np.median(feature_ms)), 3) if feature_ms else None,
        "decision_median": round(float(np.median(gate_ms)), 3) if gate_ms else None,
    }

    misses = []
    for row, d in zip(rows, decisions):
        if row["label"] == POSITIVE and d.would_skip:
            misses.append({
                "source": row["source"], "speaker": row.get("speaker", ""),
                "session": row.get("session", ""),
                "p": d.learned_confidence,
                "why": explain(model, row) if model else
                       [s.as_json() for s in d.signals],
            })
    out["skipped_assistant_directed"] = misses
    return out


def explain(model: dict, row: dict, top: int = 3) -> list[dict]:
    """The features that pushed hardest towards skipping this row."""
    vec = np.asarray([row[n] for n in FEATURE_NAMES], dtype=np.float64)
    scaled = (vec - np.asarray(model["scaler_mean"])) / np.maximum(
        np.asarray(model["scaler_scale"]), 1e-8)
    contrib = scaled * np.asarray(model["weights"])
    order = np.argsort(contrib)[:top]
    return [{"feature": FEATURE_NAMES[i], "value": round(float(vec[i]), 4),
             "logit": round(float(contrib[i]), 3)} for i in order]


# -- the shadow log ----------------------------------------------------------

def read_shadow_log(path: str | Path) -> list[dict]:
    """One record per scored turn: the gate's verdict next to the engine's."""
    turns: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            turn = event.get("turn")
            if not turn:
                continue
            name = event.get("event")
            if name == "gate_scored":
                turns.setdefault(turn, {})["gate"] = event
            elif name in ("intent_accepted", "intent_ignored"):
                turns.setdefault(turn, {})["verdict"] = name.split("_", 1)[1]
            elif name == "transcript_rejected":
                turns.setdefault(turn, {}).setdefault("verdict", "no_transcript")
            elif name == "segment_dropped" and event.get("reason") == "gate":
                turns.setdefault(turn, {})["verdict"] = "skipped"
    return [dict(turn=t, **v) for t, v in turns.items() if "gate" in v]


def shadow_report(turns: Sequence[dict]) -> dict:
    by = Counter()
    danger = []
    latency, feature = [], []
    for t in turns:
        gate = t["gate"]
        verdict = t.get("verdict", "unknown")
        would = bool(gate.get("would_skip"))
        by[(would, verdict)] += 1
        if gate.get("latency_ms") is not None:
            latency.append(float(gate["latency_ms"]))
        if gate.get("feature_ms"):
            feature.append(float(gate["feature_ms"]))
        if would and verdict == "accepted":
            danger.append({"turn": t["turn"], "confidence": gate.get("confidence"),
                           "learned_confidence": gate.get("learned_confidence"),
                           "signals": gate.get("signals")})

    def pct(values, q):
        return round(float(np.percentile(values, q)), 3) if values else None

    return {
        "scored": len(turns),
        "would_skip": sum(v for (w, _), v in by.items() if w),
        "would_skip_by_verdict": {verdict: n for (w, verdict), n in sorted(by.items()) if w},
        "would_skip_but_accepted": danger,
        "latency_ms": {"p50": pct(latency, 50), "p95": pct(latency, 95)},
        "feature_ms": {"p50": pct(feature, 50), "p95": pct(feature, 95)},
    }


def shadow_rows(turns: Sequence[dict]) -> list[dict]:
    """Feature rows from a shadow log (needs intent.gate.log_features: true).

    Labelled by nobody: ``label`` is blank until someone labels it, and the
    source is shadow_log, which `train` refuses -- this is ordinary use, and
    the people in the room did not agree to become training data.
    """
    rows = []
    for t in turns:
        feats = t["gate"].get("features")
        if not feats or any(n not in feats for n in FEATURE_NAMES):
            continue
        row = {k: "" for k in META}
        row.update(source="shadow_log", license="none",
                   intent_verdict=t.get("verdict", ""),
                   feature_ms=t["gate"].get("feature_ms", ""))
        row.update({n: float(feats[n]) for n in FEATURE_NAMES})
        rows.append(row)
    return rows


# -- command line -------------------------------------------------------------

def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _cmd_sources(args) -> int:
    for key, info in SOURCES.items():
        print(f"{key:16s} {str(info['license']):12s} {info['name']}  {info['url']}")
    print("\nopen licences accepted for training:", ", ".join(sorted(OPEN_LICENSES)))
    return 0


def _cmd_collect(args) -> int:
    from juno_core.intelligence.gate_collect import collect_main

    return collect_main(args)


def _cmd_features(args) -> int:
    manifest_path = Path(args.manifest)
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    backend = None
    if args.vad != "none":
        from juno_core.audio.vad import EnergyVAD, SileroVAD

        backend = SileroVAD(sample_rate=args.rate) if args.vad == "silero" else EnergyVAD()
    errors: list[str] = []
    rows = build_rows(manifest, vad_backend=backend, rate=args.rate,
                      base_dir=manifest_path.parent, delete_source=args.delete_source,
                      on_error=lambda i, msg: errors.append(f"row {i}: {msg}"))
    write_rows(args.out, rows)
    for line in errors:
        print("skipped", line, file=sys.stderr)
    print(f"wrote {len(rows)} feature rows to {args.out} "
          f"({len(errors)} skipped); no audio was written")
    return 1 if errors and args.strict else 0


def _cmd_train(args) -> int:
    rows = read_rows(args.rows)
    refused = [r for r in rows if not trainable(r, args.allow_internal)]
    rows = [r for r in rows if trainable(r, args.allow_internal)]
    if refused:
        print(f"not training on {len(refused)} rows whose provenance forbids it "
              f"({Counter(r['source'] for r in refused)})", file=sys.stderr)
    rows = [r for r in rows if r["label"] in LABELS]
    fractions = [float(x) for x in args.split.split(",")]
    keys = [k for k in args.group_by.split(",") if k]
    assignment = group_split(rows, fractions, keys, seed=args.seed)
    leaks = check_disjoint(rows, assignment, keys)
    if leaks:                                   # cannot happen; checked anyway
        raise SystemExit(f"leakage across splits: {leaks[:5]}")
    parts = [[r for r, a in zip(rows, assignment) if a == k] for k in range(3)]
    train, select, holdout = parts
    for name, part in zip(("train", "select", "holdout"), parts):
        counts = Counter(r["label"] for r in part)
        print(f"{name:8s} {len(part):6d} rows  {dict(counts)}", file=sys.stderr)
        if not any(r["label"] == POSITIVE for r in part):
            raise SystemExit(f"the {name} split has no assistant_directed rows; "
                             f"add data or loosen --group-by")

    model = train_model(train, select, l2=args.l2)
    protect = not args.no_wearer_protection
    p, may_skip = eligibility(select, model, protect)
    # "uncertain" is transcribed too, so by default it spends the same
    # false-skip budget as a missed request.
    protected = (POSITIVE,) if args.ignore_uncertain else (POSITIVE, "uncertain")
    positive = np.asarray([r["label"] in protected for r in select])
    model["threshold"] = round(choose_threshold(
        p, may_skip, positive, args.max_false_skip, args.max_threshold), 6)
    model["target_false_skip"] = args.max_false_skip
    model["always_transcribe_wearer"] = protect

    gate = make_gate(model, protect_wearer=protect)
    baseline = make_gate(None, learned=False, protect_wearer=protect)
    model["metrics"] = {
        "select": report(select, score_rows(select, gate), model),
        "holdout": report(holdout, score_rows(holdout, gate), model),
        "holdout_rules_baseline": report(holdout, score_rows(holdout, baseline)),
    }
    model["splits"] = {"group_by": keys, "fractions": fractions, "seed": args.seed,
                       "rows": [len(train), len(select), len(holdout)]}
    model["provenance"] = provenance_summary(train + select)
    model["trained_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(model, indent=1, allow_nan=False), encoding="utf-8")
    held = model["metrics"]["holdout"]
    _print({"threshold": model["threshold"], "holdout": held,
            "rules_baseline_holdout": model["metrics"]["holdout_rules_baseline"],
            "distributable": model["provenance"]["distributable"]})
    fsr = held["false_skip_rate"] or 0.0
    if fsr > args.max_false_skip:
        print(f"\nholdout false-skip rate {fsr:.2%} misses the {args.max_false_skip:.2%} "
              f"target: keep intent.gate.mode at shadow (or try more data / a "
              f"stricter target) before anything else.", file=sys.stderr)
        return 2
    print(f"\nwrote {args.out}. Load it with intent.gate.model_path and run in "
          f"shadow first; skip mode only after shadow agrees.", file=sys.stderr)
    return 0


def _cmd_evaluate(args) -> int:
    rows = [r for r in read_rows(args.rows) if r["label"] in LABELS]
    protect = not args.no_wearer_protection
    model = json.loads(Path(args.model).read_text(encoding="utf-8")) if args.model else None
    out = {"rules_baseline": report(rows, score_rows(
        rows, make_gate(None, learned=False, protect_wearer=protect)))}
    if model is not None:
        out["model"] = report(rows, score_rows(rows, make_gate(model, protect_wearer=protect)), model)
    _print(out)
    return 0


def _cmd_shadow(args) -> int:
    turns = read_shadow_log(args.log)
    _print(shadow_report(turns))
    if args.export:
        rows = shadow_rows(turns)
        write_rows(args.export, rows)
        print(f"exported {len(rows)} unlabelled rows (evaluation only)", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m juno_core.intelligence.gate_training",
                                     description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sources", help="list registered datasets and their licences")

    c = sub.add_parser("collect", help="guided, consented live session -> feature rows "
                                       "(no audio is saved)")
    c.add_argument("--out", required=True)
    c.add_argument("--speakers", required=True,
                   help="comma-separated pseudonyms of everyone speaking, e.g. p1,p2")
    c.add_argument("--room", required=True, help="e.g. kitchen, office")
    c.add_argument("--consent", required=True, choices=("release", "internal"))
    c.add_argument("--session", help="default: session-YYYYMMDD-HHMM")
    c.add_argument("--mic", help="default: the current input device's name")
    c.add_argument("--config", help="config.yaml for audio/vad/own_voice settings")
    c.add_argument("--scale", type=float, default=1.0,
                   help="multiply every prompt's duration (0.5 = a half-length session)")
    c.add_argument("--append", action="store_true", help="add to an existing --out file")

    f = sub.add_parser("features", help="manifest of labelled clips -> feature rows")
    f.add_argument("--manifest", required=True,
                   help="CSV: path,label,source[,license,consent,speaker,session,room,mic,"
                        "start,end,since_ai,awaiting_answer,confirming,offer_pending,"
                        "ignored_run,accepted_recently,p_own,voice_confident,is_wearer,"
                        "vad_confidence]")
    f.add_argument("--out", required=True)
    f.add_argument("--rate", type=int, default=16000)
    f.add_argument("--vad", choices=("silero", "energy", "none"), default="energy",
                   help="backend for vad_confidence when the manifest has none")
    f.add_argument("--delete-source", action="store_true",
                   help="delete each consented clip once its features are written")
    f.add_argument("--strict", action="store_true", help="exit non-zero if any row failed")

    t = sub.add_parser("train", help="fit, calibrate and threshold a logistic gate")
    t.add_argument("--rows", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--max-false-skip", type=float, default=0.01,
                   help="largest tolerated share of assistant_directed rows skipped")
    t.add_argument("--max-threshold", type=float, default=0.5)
    t.add_argument("--split", default="0.6,0.2,0.2", help="train,select,holdout")
    t.add_argument("--group-by", default="speaker,session,room")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--l2", type=float, default=1.0)
    t.add_argument("--allow-internal", action="store_true",
                   help="also train on consent=internal rows (model is not distributable)")
    t.add_argument("--no-wearer-protection", action="store_true",
                   help="evaluate as if always_transcribe_wearer were off")
    t.add_argument("--ignore-uncertain", action="store_true",
                   help="let 'uncertain' rows be skipped when choosing the threshold")

    e = sub.add_parser("evaluate", help="rules baseline (and a model) on unseen rows")
    e.add_argument("--rows", nargs="+", required=True)
    e.add_argument("--model")
    e.add_argument("--no-wearer-protection", action="store_true")

    s = sub.add_parser("shadow", help="compare shadow would-skips with the engine's verdicts")
    s.add_argument("--log", required=True)
    s.add_argument("--export", help="write unlabelled feature rows (needs log_features)")

    args = parser.parse_args(argv)
    return {"sources": _cmd_sources, "collect": _cmd_collect, "features": _cmd_features,
            "train": _cmd_train, "evaluate": _cmd_evaluate,
            "shadow": _cmd_shadow}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
