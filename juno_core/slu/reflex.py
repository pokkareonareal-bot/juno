"""Reflex SLU: hear an utterance, answer in the schema, without words.

    audio --> encoder (frames) --> pool --> student heads --> router --> Decision

The fast path. It runs on every segment the voiceprint lets through and
returns a typed Decision (``source`` is ``SOURCE_REFLEX``) and one of three
routes. What the pipeline does with that route depends on the mode:

  off      not loaded, not run. The pipeline is exactly what it was.
  shadow   runs, logs its Decision, and changes nothing: the cascade still
           runs on every segment, so the two can be compared turn by turn
           for free (``python -m juno_core.slu shadow`` reads the log).
  on       ``ignore`` drops the segment; ``act`` delivers the Decision to the
           agent with no transcription at all; ``escalate`` runs the cascade.

``act`` is only honoured when something can receive a typed Decision -- an
``on_decision`` handler (see pipeline.py). An agent that only takes text has
nothing to act with, so for it ``act`` becomes ``escalate``.

What can never happen here, whatever the model says:
  - a missing or mismatched model (wrong encoder, unreadable file) means
    the mode falls back to off, logged, and the pipeline runs as before;
  - a model without measured thresholds never ignores or acts (router.py);
  - an exception on one segment escalates that segment.
Audio is held in memory only: encoded, pooled, released.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from juno_core.events import REFLEX_UNAVAILABLE
from juno_core.slu.encoder import AudioEncoder, build_encoder
from juno_core.slu.router import ConversationState, Router, Thresholds
from juno_core.slu.schema import CORE_SCHEMA, SOURCE_REFLEX, Decision, Field, Schema
from juno_core.slu.student import StudentModel

MODES = ("off", "shadow", "on")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "slu_student.npz"


class Reflex:
    def __init__(self, config=None, observer=None, *, model: StudentModel | None = None,
                 encoder: AudioEncoder | None = None, schema: Schema | None = None) -> None:
        get = config.get if config is not None else (lambda k, d=None: d)
        mode = get("mode", "off")
        if mode is None or isinstance(mode, bool):   # YAML reads a bare off / on as a boolean
            mode = "on" if mode else "off"
        mode = str(mode).lower()
        if mode not in MODES:
            raise ValueError(f"reflex.mode must be one of {MODES}, not {mode!r}")
        self.mode = mode
        self.observer = observer
        self.error = ""
        self.model = model
        self.encoder = encoder
        self.router: Router | None = None
        self.followup_window = float(get("followup_window", 20.0))
        path = get("model_path")
        self.model_path = Path(path) if path else DEFAULT_MODEL_PATH
        self.counts = {"act": 0, "ignore": 0, "escalate": 0, "error": 0}
        if self.mode == "off":
            return
        try:
            if self.model is None:
                self.model = StudentModel.load(self.model_path)
            if self.encoder is None:
                self.encoder = build_encoder(self.model.encoder_spec)
            if self.encoder.spec != self.model.encoder_spec:
                raise ValueError(f"model was trained on {self.model.encoder_spec}, "
                                 f"encoder is {self.encoder.spec}")
            model_schema = self.model.schema
            if schema is not None and set(schema.intent_names) - set(model_schema.intent_names):
                missing = sorted(set(schema.intent_names) - set(model_schema.intent_names))
                raise ValueError(f"the agent's schema has intents the model was not trained "
                                 f"on: {', '.join(missing)} -- train with that schema")
            self.schema = schema or model_schema
            thresholds = Thresholds.from_dict(self.model.thresholds)
            overrides = get("thresholds")
            if overrides:
                # Config may only tighten what the model file measured.
                thresholds = _tighten(thresholds, Thresholds.from_dict(dict(overrides)))
            self.router = Router(self.schema, thresholds, self.followup_window)
        except Exception as exc:          # a bad model must never stop the loop
            self.error = f"{type(exc).__name__}: {exc}"
            self.mode = "off"
            self._emit(REFLEX_UNAVAILABLE, error=self.error, path=str(self.model_path))

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def warmup(self) -> None:
        if self.enabled and self.encoder is not None:
            self.encoder.warmup()

    def decide(self, audio: np.ndarray, sample_rate: int = 16000,
               state: ConversationState | None = None, turn: str | None = None) -> Decision:
        started = time.perf_counter()
        try:
            vector = self.encoder.encode(audio, sample_rate)
        except Exception as exc:
            return self._failed(exc, started, turn)
        return self.decide_vector(vector, state, turn,
                                  encode_ms=(time.perf_counter() - started) * 1000.0,
                                  started=started)

    def decide_vector(self, vector: np.ndarray, state: ConversationState | None = None,
                      turn: str | None = None, *, encode_ms: float = 0.0,
                      started: float | None = None) -> Decision:
        """The same decision from an already-encoded vector (from ``self.encoder``).

        ``started`` is when encoding began, if the caller timed it as part of
        this decision; otherwise ``encode_ms`` is added to the total."""
        prior_ms = 0.0
        if started is None:
            started, prior_ms = time.perf_counter(), encode_ms
        try:
            pred = self.model.predict(vector)
            routed = self.router.route(pred, state)
        except Exception as exc:
            return self._failed(exc, started, turn)
        self.counts[routed.route] += 1
        return Decision(
            route=routed.route, source=SOURCE_REFLEX, addressed=routed.addressed,
            intent=routed.intent, slots=routed.slots, reason=routed.reason, turn=turn,
            latency_ms={SOURCE_REFLEX: prior_ms + (time.perf_counter() - started) * 1000.0,
                        "encoder": encode_ms, "heads": pred.latency_ms},
            model=self.model.tag,
        )

    def _failed(self, exc: Exception, started: float, turn: str | None) -> Decision:
        self.counts["error"] += 1
        return Decision(route="escalate", source=SOURCE_REFLEX,
                        addressed=Field("assistant_directed", 0.0), turn=turn,
                        reason=f"reflex failed: {type(exc).__name__}: {exc}",
                        latency_ms={SOURCE_REFLEX: (time.perf_counter() - started) * 1000.0},
                        model=self.model.tag if self.model else None)

    def snapshot(self) -> dict:
        return {"mode": self.mode, "error": self.error,
                "model": self.model.tag if self.model else None,
                "encoder": self.encoder.spec if self.encoder else None, **self.counts}

    def _emit(self, name: str, **fields) -> None:
        if self.observer is not None:
            from juno_core.events import Stage

            self.observer.emit(Stage.SYSTEM, name, None, **fields)


def _tighten(model: Thresholds, config: Thresholds) -> Thresholds:
    """Config can raise act thresholds and lower ignore_below, never the reverse."""
    def lower(a, b):
        return b if a is None else (a if b is None else min(a, b))

    def higher(a, b):
        return None if a is None else (a if b is None else max(a, b))

    per = dict(model.per_intent)
    for k, v in config.per_intent.items():
        per[k] = max(per.get(k, 0.0), v)
    return Thresholds(
        ignore_below=lower(model.ignore_below, config.ignore_below) if model.ignore_below is not None else None,
        act_addressed=higher(model.act_addressed, config.act_addressed),
        act_intent=higher(model.act_intent, config.act_intent),
        act_slot=higher(model.act_slot, config.act_slot),
        per_intent=per,
    )


__all__ = ["Reflex", "MODES", "CORE_SCHEMA"]
