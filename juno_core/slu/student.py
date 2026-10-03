"""The student: a small network over one pooled encoder vector.

One shared hidden layer and one softmax head per question the schema asks:

    addressed   assistant_directed | human_directed | background_or_media
    intent      every intent in the schema, open_request included
    slot heads  one per (intent, slot), over that slot's declared values

It is trained by distillation: the targets are what System Two -- the
expensive cascade -- said about the same audio, as probabilities, not just
its top answer. Where a gold label exists (a prompted recording, a scripted
synthetic clip), training can mix it in; the training CLI says how much.

Everything here is numpy. Inference is a couple of matrix products --
microseconds next to the encoder -- and the model file is an .npz holding
the weights and a JSON header (schema, encoder spec, thresholds, metrics,
provenance), so a model cannot be loaded against the wrong encoder or the
wrong schema without that being an error.

Calibration: each head gets a temperature fitted on a held-out split, so a
0.9 means right about nine times in ten on data the weights never saw. The
router's thresholds are chosen on those calibrated numbers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from juno_core.slu.schema import ADDRESSEES, Schema

MODEL_FORMAT = "juno-slu-student/1"

# Every slot head has one class more than the slot declares: "a value, but not
# one of these". A timer for 13 minutes is a real request with an untyped
# length, and without somewhere to put it the head would round it to 12 or 15
# with confidence. The router escalates whenever this class wins.
OTHER = "__other__"


def slot_classes(schema: Schema) -> dict[str, list]:
    return {key: list(schema.intent(key.split(":")[0]).slot(key.split(":")[1]).values) + [OTHER]
            for key in schema.slot_keys()}


@dataclass
class Prediction:
    addressed: dict[str, float]
    intent: dict[str, float]
    slots: dict[str, dict[Any, float]]      # "intent:slot" -> {value: p}
    latency_ms: float = 0.0

    def top(self, dist: dict) -> tuple[Any, float]:
        key = max(dist, key=dist.get)
        return key, float(dist[key])


@dataclass
class Targets:
    """Training targets for n rows. Masks say which rows teach which head."""

    addressed: np.ndarray                    # (n, 3) probabilities
    intent: np.ndarray                       # (n, K) probabilities
    intent_mask: np.ndarray                  # (n,) 1 where the intent is known
    slots: dict[str, np.ndarray] = field(default_factory=dict)       # (n, V)
    slot_masks: dict[str, np.ndarray] = field(default_factory=dict)  # (n,)
    weight: np.ndarray | None = None         # (n,) per-row weight

    def subset(self, idx: np.ndarray) -> "Targets":
        return Targets(
            addressed=self.addressed[idx], intent=self.intent[idx],
            intent_mask=self.intent_mask[idx],
            slots={k: v[idx] for k, v in self.slots.items()},
            slot_masks={k: v[idx] for k, v in self.slot_masks.items()},
            weight=None if self.weight is None else self.weight[idx],
        )


def _softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = z / max(temperature, 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class StudentModel:
    """Weights plus the header that says what they mean."""

    def __init__(self, meta: dict, params: dict[str, np.ndarray]) -> None:
        self.meta = meta
        self.params = params
        self.addressees: list[str] = list(meta["addressees"])
        self.intents: list[str] = list(meta["intents"])
        self.slot_values: dict[str, list] = {k: list(v) for k, v in meta["slots"].items()}
        self.temperatures: dict[str, float] = dict(meta.get("temperatures", {}))
        self.mean = params["x_mean"]
        self.scale = params["x_scale"]

    # -- identity --------------------------------------------------------

    @property
    def encoder_spec(self) -> str:
        return self.meta["encoder"]

    @property
    def schema(self) -> Schema:
        return Schema.from_dict(self.meta["schema"])

    @property
    def thresholds(self) -> dict:
        return self.meta.get("thresholds") or {}

    @property
    def tag(self) -> str:
        return self.meta.get("tag") or f"student@{self.meta.get('trained_at', '?')}"

    @property
    def distributable(self) -> bool:
        return bool((self.meta.get("provenance") or {}).get("distributable", False))

    # -- inference -------------------------------------------------------

    def _hidden(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean) / self.scale
        if "W1" not in self.params:
            return Xs
        return np.maximum(0.0, Xs @ self.params["W1"] + self.params["b1"])

    def logits(self, X: np.ndarray) -> dict[str, np.ndarray]:
        H = self._hidden(np.atleast_2d(np.asarray(X, dtype=np.float32)))
        out = {"addressed": H @ self.params["Wa"] + self.params["ba"],
               "intent": H @ self.params["Wi"] + self.params["bi"]}
        for key in self.slot_values:
            out[key] = H @ self.params[f"Ws:{key}"] + self.params[f"bs:{key}"]
        return out

    def probabilities(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {head: _softmax(z, self.temperatures.get(head, 1.0))
                for head, z in self.logits(X).items()}

    def predict(self, vector: np.ndarray) -> Prediction:
        t0 = time.perf_counter()
        probs = self.probabilities(vector)
        pred = Prediction(
            addressed=dict(zip(self.addressees, map(float, probs["addressed"][0]))),
            intent=dict(zip(self.intents, map(float, probs["intent"][0]))),
            slots={key: dict(zip(values, map(float, probs[key][0])))
                   for key, values in self.slot_values.items()},
        )
        pred.latency_ms = (time.perf_counter() - t0) * 1000.0
        return pred

    # -- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = np.frombuffer(json.dumps(self.meta, allow_nan=False).encode("utf-8"),
                               dtype=np.uint8)
        np.savez(path, __meta__=header, **self.params)

    @classmethod
    def load(cls, path: str | Path) -> "StudentModel":
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(bytes(data["__meta__"]).decode("utf-8"))
            params = {k: data[k] for k in data.files if k != "__meta__"}
        if meta.get("format") != MODEL_FORMAT:
            raise ValueError(f"{path}: not a {MODEL_FORMAT} file")
        for key in ("x_mean", "x_scale", "Wa", "ba", "Wi", "bi"):
            if key not in params:
                raise ValueError(f"{path}: missing {key}")
            if not np.all(np.isfinite(params[key])):
                raise ValueError(f"{path}: non-finite values in {key}")
        return cls(meta, params)


# -- training ----------------------------------------------------------------

class _Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float, wd: float) -> None:
        self.lr, self.wd = lr, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads) -> None:
        self.t += 1
        b1, b2 = 0.9, 0.999
        for k, g in grads.items():
            if k.startswith("W"):
                g = g + self.wd * params[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)


def _init(d_in: int, hidden: int, n_intents: int, slot_values: dict, rng) -> dict:
    def glorot(a, b):
        return (rng.standard_normal((a, b)) * np.sqrt(2.0 / (a + b))).astype(np.float32)

    params: dict[str, np.ndarray] = {}
    width = d_in
    if hidden > 0:
        params["W1"], params["b1"] = glorot(d_in, hidden), np.zeros(hidden, np.float32)
        width = hidden
    params["Wa"], params["ba"] = glorot(width, len(ADDRESSEES)), np.zeros(len(ADDRESSEES), np.float32)
    params["Wi"], params["bi"] = glorot(width, n_intents), np.zeros(n_intents, np.float32)
    for key, values in slot_values.items():
        params[f"Ws:{key}"] = glorot(width, len(values))
        params[f"bs:{key}"] = np.zeros(len(values), np.float32)
    return params


def _loss_and_grads(params, Xs, T: Targets, slot_keys, dropout, rng, train=True):
    n = Xs.shape[0]
    w = np.ones(n, np.float32) if T.weight is None else T.weight.astype(np.float32)
    grads: dict[str, np.ndarray] = {}
    if "W1" in params:
        pre = Xs @ params["W1"] + params["b1"]
        H = np.maximum(0.0, pre)
        if train and dropout > 0:
            keep = (rng.random(H.shape) >= dropout).astype(np.float32) / (1.0 - dropout)
            H = H * keep
        else:
            keep = None
    else:
        H, pre, keep = Xs, None, None

    dH = np.zeros_like(H)
    total = 0.0

    def head(Wk, bk, target, mask):
        nonlocal total, dH
        z = H @ params[Wk] + params[bk]
        p = _softmax(z)
        rw = (w * mask)[:, None]
        denom = max(float(rw.sum()), 1.0)
        total += float(-(rw * target * np.log(p + 1e-9)).sum() / denom)
        dz = rw * (p - target) / denom
        grads[Wk] = H.T @ dz
        grads[bk] = dz.sum(axis=0)
        dH += dz @ params[Wk].T

    head("Wa", "ba", T.addressed, np.ones(n, np.float32))
    head("Wi", "bi", T.intent, T.intent_mask.astype(np.float32))
    for key in slot_keys:
        head(f"Ws:{key}", f"bs:{key}", T.slots[key], T.slot_masks[key].astype(np.float32))

    if "W1" in params:
        if keep is not None:
            dH = dH * keep
        dpre = dH * (pre > 0)
        grads["W1"] = Xs.T @ dpre
        grads["b1"] = dpre.sum(axis=0)
    return total, grads


def fit(X: np.ndarray, T: Targets, X_val: np.ndarray, T_val: Targets, *, schema: Schema,
        encoder_spec: str, hidden: int = 256, epochs: int = 80, lr: float = 1e-3,
        weight_decay: float = 1e-4, dropout: float = 0.2, batch: int = 64,
        patience: int = 12, seed: int = 0, log=None) -> StudentModel:
    """Train on (X, T), early-stop and calibrate on (X_val, T_val)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float32)
    X_val = np.asarray(X_val, np.float32)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-6] = 1.0
    Xs, Xv = (X - mean) / scale, (X_val - mean) / scale

    intents = list(schema.intent_names)
    slot_values = slot_classes(schema)
    slot_keys = list(slot_values)
    params = _init(X.shape[1], hidden, len(intents), slot_values, rng)
    opt = _Adam(params, lr, weight_decay)

    best, best_loss, stale = None, float("inf"), 0
    history = []
    for epoch in range(epochs):
        order = rng.permutation(X.shape[0])
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            _, grads = _loss_and_grads(params, Xs[idx], T.subset(idx), slot_keys, dropout, rng)
            opt.step(params, grads)
        val_loss, _ = _loss_and_grads(params, Xv, T_val, slot_keys, 0.0, rng, train=False)
        history.append(round(val_loss, 5))
        if log:
            log(f"epoch {epoch + 1:3d}  val loss {val_loss:.4f}")
        if val_loss < best_loss - 1e-4:
            best_loss, stale = val_loss, 0
            best = {k: v.copy() for k, v in params.items()}
        else:
            stale += 1
            if stale >= patience:
                break
    params = best or params
    params["x_mean"], params["x_scale"] = mean.astype(np.float32), scale.astype(np.float32)

    meta = {
        "format": MODEL_FORMAT,
        "encoder": encoder_spec,
        "schema": schema.to_dict(),
        "addressees": list(ADDRESSEES),
        "intents": intents,
        "slots": slot_values,
        "hidden": hidden,
        "training": {"epochs_run": len(history), "best_val_loss": round(best_loss, 5),
                     "val_loss_history": history, "lr": lr, "weight_decay": weight_decay,
                     "dropout": dropout, "batch": batch, "seed": seed},
        "thresholds": {},
        "temperatures": {},
    }
    model = StudentModel(meta, params)
    model.meta["temperatures"] = calibrate(model, X_val, T_val)
    model.temperatures = dict(model.meta["temperatures"])
    return model


def calibrate(model: StudentModel, X: np.ndarray, T: Targets) -> dict[str, float]:
    """One temperature per head, minimising held-out NLL over a grid."""
    logits = model.logits(X)
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 61))
    out = {}

    def best_t(z, target, mask):
        if mask.sum() < 5:
            return 1.0
        z, target = z[mask > 0], target[mask > 0]
        losses = [float(-(target * np.log(_softmax(z, t) + 1e-9)).sum(axis=1).mean()) for t in grid]
        return float(round(grid[int(np.argmin(losses))], 4))

    out["addressed"] = best_t(logits["addressed"], T.addressed, np.ones(len(X)))
    out["intent"] = best_t(logits["intent"], T.intent, T.intent_mask)
    for key in model.slot_values:
        out[key] = best_t(logits[key], T.slots[key], T.slot_masks[key])
    return out


def expected_calibration_error(p: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """How far stated confidence is from observed accuracy, averaged over bins."""
    p, correct = np.asarray(p, float), np.asarray(correct, float)
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - correct[m].mean())
    return float(ece)


def build_targets(rows: Sequence[dict], schema: Schema, mode: str = "teacher",
                  gold_weight: float = 0.5) -> Targets:
    """Training targets from labelled rows.

    Each row carries the teacher's soft answer (``t_addressed`` probabilities,
    ``t_intent``, ``t_intent_p``, ``t_slots``) and, when known, the gold one
    (``g_addressed``, ``g_intent``, ``g_slots``). ``mode``:

      teacher   distillation only -- what the cascade said
      gold      ground truth only (the upper bound when labels are free)
      mix       gold_weight * gold + (1 - gold_weight) * teacher
    """
    if mode not in ("teacher", "gold", "mix"):
        raise ValueError("mode must be teacher, gold or mix")
    n = len(rows)
    intents = list(schema.intent_names)
    keys = list(schema.slot_keys())
    classes = slot_classes(schema)
    A = np.zeros((n, len(ADDRESSEES)), np.float32)
    I = np.zeros((n, len(intents)), np.float32)
    Im = np.zeros(n, np.float32)
    S = {k: np.zeros((n, len(classes[k])), np.float32) for k in keys}
    Sm = {k: np.zeros(n, np.float32) for k in keys}

    def onehot_addr(label):
        v = np.zeros(len(ADDRESSEES), np.float32)
        if label in ADDRESSEES:
            v[ADDRESSEES.index(label)] = 1.0
        return v

    def intent_vec(name, p):
        v = np.zeros(len(intents), np.float32)
        if name not in intents:
            return None
        rest = (1.0 - p) / max(len(intents) - 1, 1)
        v[:] = rest
        v[intents.index(name)] = p
        return v

    for r, row in enumerate(rows):
        teacher_a = np.asarray([float(row.get("t_addressed", {}).get(a, 0.0)) for a in ADDRESSEES],
                               np.float32) if row.get("t_addressed") else None
        gold_a = onehot_addr(row.get("g_addressed")) if row.get("g_addressed") else None
        a = _blend(teacher_a, gold_a, mode, gold_weight)
        if a is None or a.sum() <= 0:
            a = np.full(len(ADDRESSEES), 1.0 / len(ADDRESSEES), np.float32)
        A[r] = a / a.sum()

        teacher_i = intent_vec(row.get("t_intent"), float(row.get("t_intent_p", 1.0))) \
            if row.get("t_intent") else None
        gold_i = intent_vec(row.get("g_intent"), 1.0) if row.get("g_intent") else None
        i = _blend(teacher_i, gold_i, mode, gold_weight)
        # The intent head learns only from speech that was for the assistant:
        # "what does the user want" has no answer for a remark to a person.
        if i is not None and A[r, 0] >= 0.5:
            I[r], Im[r] = i, 1.0

        t_slots = row.get("t_slots") or {}
        g_slots = row.get("g_slots") or {}
        chosen_intent = intents[int(np.argmax(I[r]))] if Im[r] else None
        for key in keys:
            name, slot = key.split(":")
            if chosen_intent != name:
                continue
            values = classes[key]
            tv = t_slots.get(slot)
            gv = g_slots.get(slot)
            t_vec = _onehot(values, tv)
            g_vec = _onehot(values, gv)
            s = _blend(t_vec, g_vec, mode, gold_weight)
            if s is not None and s.sum() > 0:
                S[key][r], Sm[key][r] = s / s.sum(), 1.0
    return Targets(A, I, Im, S, Sm)


def _onehot(values: list, value) -> np.ndarray | None:
    """One-hot over a slot's classes; a value off the list is OTHER."""
    if value is None or not values:
        return None
    try:
        idx = values.index(type(values[0])(value))
    except (ValueError, TypeError):
        idx = values.index(OTHER) if OTHER in values else -1
    if idx < 0:
        return None
    v = np.zeros(len(values), np.float32)
    v[idx] = 1.0
    return v


def _blend(teacher, gold, mode, w):
    if mode == "teacher":
        return teacher if teacher is not None else None
    if mode == "gold":
        return gold
    if teacher is None:
        return gold
    if gold is None:
        return teacher
    return w * gold + (1.0 - w) * teacher
