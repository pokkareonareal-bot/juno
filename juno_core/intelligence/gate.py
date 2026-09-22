"""Is this worth transcribing at all?

Asked before Whisper runs, from what is known before Whisper runs: how the
utterance sounds, who it sounds like, and what the conversation was doing when
it started. Whisper is the most expensive thing in the pipeline by a factor of
forty, and over the six days of logs this was designed against, 63% of what it
transcribed was then thrown away by the intent engine. The existing gate --
"drop it if the voice is confidently not the wearer's" -- caught 3%.

WHAT THE LOGS SAID THE WASTE WAS MADE OF
----------------------------------------
Not strangers. Every one of the 245 ignored utterances was the wearer's own
voice, or a voice that could not be told apart from it. The waste is the wearer
talking to other people, and no voiceprint can catch that -- p(the wearer
spoke) says nothing about p(the wearer meant us).

What separates them is WHEN. Seconds since the assistant last spoke: accepted
utterances at a median of 2 s, ignored ones at a median of 65 s. Whether the
previous utterance was accepted: 69% of accepted ones follow an accepted one,
82% of ignored ones follow an ignored one. Both are timestamps and a flag,
available before a single sample is decoded. Duration and the VAD's confidence
separate only at the tails: very short, very long and low-confidence segments
skew ignored, the middle is indistinguishable.

THE ASYMMETRY, WHICH IS THE OPPOSITE OF THE INTENT ENGINE'S
-----------------------------------------------------------
For the engine, a false activation -- speaking into somebody else's
conversation -- is three times worse than a miss. Here a false NEGATIVE is a
missed activation the engine never gets to see, while a false positive costs a
second of GPU. So this is built to be conservative in one direction only:
it skips when several independent things agree, and it never skips when any
one of a short list of things holds. The acceptance criterion is zero false
negatives on the shadow set; the skip rate is whatever that leaves.

WHAT IT DECIDES, AND WHAT IT DOES NOT
-------------------------------------
Whether to LOOK. The intent engine still decides whether to ANSWER and the
broker still decides whether to ACT. Nothing here grants an interaction or
any authority; it can only decline to spend the expensive path.

It runs in three modes. ``off`` scores nothing. ``shadow`` scores every
segment, logs what it would have done, and skips nothing -- so its verdict
can be compared with the engine's, per segment, for free, before it is
trusted. ``skip`` is what shadow becomes once that comparison has been read.

Like the intent engine, every decision decomposes into named, weighted
signals that are logged. A gate that cannot explain a skip is a gate that
cannot be debugged at three in the morning by the person it just ignored.

THE LEARNED LAYER, AND WHAT IT MAY NOT DO
-----------------------------------------
An optional logistic model (trained offline by gate_training.py on public or
explicitly consented data -- never on audio captured by this runtime) scores
p(assistant_directed) from the 20 features in features.py. Its skip
threshold comes from the model file, chosen on held-out speakers and sessions
for a target false-skip rate; there is no built-in default. Whatever it says,
it cannot skip when:

  - the assistant is waiting for an answer, a confirmation or an offer, or
    the follow-up window is open;
  - the enrolled voice was confidently detected (``always_transcribe_wearer``,
    on unless evaluation shows a narrower exception is safe);
  - feature extraction failed or produced something it cannot trust
    (no audio, nothing voiced, a non-finite value);
  - the model failed to load, or does not carry a threshold.

Audio is only ever held in memory here: scored, handed to STT, released.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from juno_core.intelligence.features import FEATURE_NAMES, extract_gate_features

MODES = ("off", "shadow", "skip")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "gate_model.json"

# Log-odds contributions. Positive means "worth transcribing". The bias is a
# prior in FAVOUR of looking, which is the conservative direction: with no
# evidence at all the answer is to transcribe.
DEFAULT_WEIGHTS = {
    "bias": 1.2,
    # Conversational. The strongest evidence there is, and free.
    "cold_conversation": -1.1,      # nothing said by the assistant for > cold_after
    "very_cold_conversation": -0.7,  # ...and for > very_cold_after (adds to the above)
    "ignored_run": -0.55,           # per consecutive ignored verdict, capped
    "accepted_recently": 0.9,       # the previous utterance was answered
    # Acoustic. Tails only; the middle carries no information.
    "not_wearer": -3.0,             # confident, and below the enrolled line
    "very_long": -0.8,
    "very_short": -0.5,
    "low_vad_confidence": -0.6,
}

# A skip needs one of THESE as well as a cold conversation. Found by judging
# the conversational rules alone against 1,177 logged utterances with the
# voice assumed to be the wearer's: 55 to 72 of the 337 the engine accepted
# would have been skipped, and the list was unambiguous -- "What is the
# capital of Mongolia?", "Juno, open a Safari tab", the first thing said after
# minutes of silence. A conversation OPENER is cold by definition, with a run
# of ignored speech behind it by definition, so coldness cannot license a skip
# on its own. It can only argue; something about the SOUND has to agree.
#
# very_short is deliberately not here: "Juno?" is short, and it is an opener.
LICENSING = frozenset({"not_wearer", "very_long", "low_vad_confidence"})

DEFAULT_THRESHOLDS = {
    "skip_below": 0.25,     # p(worth transcribing) under which skip mode skips
    "cold_after": 20.0,     # seconds; the intent engine's followup_window
    "very_cold_after": 60.0,
    "ignored_run_cap": 3,
    "long_seconds": 10.0,
    "short_seconds": 1.2,
    "low_confidence": 0.90,
}


@dataclass(frozen=True)
class Acoustics:
    """What the segment sounds like, before anyone decodes it."""

    seconds: float
    confidence: float           # mean VAD probability over voiced frames
    truncated: bool = False
    p_own: float | None = None   # the voiceprint's estimate, when it has one
    voice_confident: bool = False
    is_wearer: bool | None = None


# "The assistant has not spoken at all yet." Infinity is the right value in
# code -- every comparison in this file works with it -- and the wrong one the
# moment it is written down. JSON encoders write a bare Infinity, which many
# parsers reject. The conversion happens once, here, at the edge where a
# number becomes data.
NEVER_SPOKEN = 1e9


def loggable(seconds: float) -> float:
    """A seconds value safe to write to a log or a JSON sidecar."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return NEVER_SPOKEN
    if value != value or value > NEVER_SPOKEN:      # NaN or infinity
        return NEVER_SPOKEN
    return round(value, 1)


@dataclass(frozen=True)
class Snapshot:
    """What the conversation was doing when the utterance started."""

    since_ai: float                 # seconds since the assistant last spoke
    awaiting_answer: bool = False   # its last line ended in a question mark
    confirming: bool = False        # a spoken confirmation is open
    offer_pending: bool = False     # "that'll take a bit longer, shall I?"
    busy: bool = False
    # Newest last. True = accepted, False = ignored, None = no verdict.
    recent_verdicts: tuple = ()


@dataclass(frozen=True)
class Signal:
    name: str
    weight: float
    detail: str = ""

    def as_json(self) -> dict:
        out = {"signal": self.name, "w": round(self.weight, 3)}
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass
class Decision:
    """What the gate thinks, what it would do, and what it will do."""

    confidence: float               # p(worth transcribing)
    signals: list[Signal] = field(default_factory=list)
    would_skip: bool = False        # the verdict, whatever the mode
    skip: bool = False              # the action: only ever true in skip mode
    reason: str = ""                # why it must transcribe, when it must
    mode: str = "off"
    latency: float = 0.0
    rule_would_skip: bool = False
    learned_confidence: float | None = None   # p(assistant_directed)
    learned_would_skip: bool = False
    learned_reason: str = ""        # why the learned layer could not skip
    features: np.ndarray | None = None
    feature_latency: float = 0.0
    model_latency: float = 0.0

    def as_log(self, include_features: bool = False) -> dict:
        """The numbers worth writing down. Never audio."""
        out = {
            "confidence": round(self.confidence, 3),
            "would_skip": self.would_skip,
            "skip": self.skip,
            "mode": self.mode,
            "reason": self.reason,
            "rule_would_skip": self.rule_would_skip,
            "signals": [s.as_json() for s in self.signals],
            "latency_ms": round(self.latency * 1000.0, 3),
        }
        if self.learned_confidence is not None or self.learned_reason:
            out.update(
                learned_confidence=self.learned_confidence,
                learned_would_skip=self.learned_would_skip,
                learned_reason=self.learned_reason,
                feature_ms=round(self.feature_latency * 1000.0, 3),
                model_ms=round(self.model_latency * 1000.0, 3),
            )
        if include_features and self.features is not None:
            out["features"] = {name: round(float(v), 5)
                               for name, v in zip(FEATURE_NAMES, self.features)}
        return out


class Gate:
    def __init__(self, config=None, observer=None) -> None:
        get = (config.get if config is not None else (lambda k, d=None: d))
        mode = str(get("mode", "off")).lower()
        if mode not in MODES:
            raise ValueError(f"intent.gate.mode must be one of {MODES}, not {mode!r}")
        self.mode = mode
        self.weights = dict(DEFAULT_WEIGHTS)
        self.weights.update({k: float(v) for k, v in (get("weights", {}) or {}).items()
                             if hasattr(get("weights", {}) or {}, "items")})
        self.limits = dict(DEFAULT_THRESHOLDS)
        for key in DEFAULT_THRESHOLDS:
            if get(key) is not None:
                self.limits[key] = float(get(key))
        self.veto_acknowledgement = bool(get("veto_acknowledgement", True))
        self.use_learned = bool(get("learned", False))
        self.always_transcribe_wearer = bool(get("always_transcribe_wearer", True))
        # Compute and log the feature vector for every scored segment, even
        # with no model loaded -- the numbers an offline evaluation needs.
        self.log_features = bool(get("log_features", False))
        self._observer = observer

        model_path_cfg = get("model_path")
        self.model_path = Path(model_path_cfg) if model_path_cfg else DEFAULT_MODEL_PATH
        self.model_data: dict | None = None
        self.model_error: str = ""
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0
        self._scaler_mean: np.ndarray | None = None
        self._scaler_scale: np.ndarray | None = None
        self._threshold: float | None = None
        self._load_learned_model()
        if self.mode != "off" and (self.model_data is not None or self.log_features):
            self._warm_up()

        # What it has done, for /status. Shadow counts what it would have done.
        self.scored = 0
        self.would_skip = 0
        self.skipped = 0
        self.never_skip = 0

    def _load_learned_model(self) -> None:
        if not self.model_path.exists():
            if self.use_learned:
                self.model_error = f"no model at {self.model_path}"
            return
        try:
            self.load_model(json.loads(self.model_path.read_text(encoding="utf-8")))
        except Exception as exc:          # a bad model must never stop the loop
            self.model_data = None
            self._weights = None
            self.model_error = f"{type(exc).__name__}: {exc}"
            if self._observer is not None:
                from juno_core.events import Stage

                self._observer.emit(Stage.SYSTEM, "gate_model_failed", None,
                                    path=str(self.model_path), error=self.model_error)

    def load_model(self, data: dict) -> None:
        """Install a model dict (the format gate_training.py writes)."""
        n = len(FEATURE_NAMES)
        names = data.get("feature_names")
        if names is not None and tuple(names) != FEATURE_NAMES:
            raise ValueError("model was trained on a different feature set")
        weights = np.asarray(data["weights"], dtype=np.float64)
        mean = np.asarray(data["scaler_mean"], dtype=np.float64)
        scale = np.asarray(data["scaler_scale"], dtype=np.float64)
        if weights.shape != (n,) or mean.shape != (n,) or scale.shape != (n,):
            raise ValueError(f"model vectors must have {n} entries")
        if not (np.all(np.isfinite(weights)) and np.all(np.isfinite(mean))
                and np.all(np.isfinite(scale))):
            raise ValueError("model contains non-finite values")
        threshold = data.get("threshold")
        # No threshold, no skipping: it has to be chosen on held-out data,
        # not assumed.
        self._threshold = None if threshold is None else float(threshold)
        self._weights = weights
        self._bias = float(data["bias"])
        self._scaler_mean = mean
        self._scaler_scale = np.maximum(scale, 1e-8)
        self.model_data = data
        self.model_error = ""

    @staticmethod
    def _warm_up() -> None:
        """Pay the first-call cost (FFT plans, lazy imports) at start-up,
        not on the first utterance -- measured at ~50 ms against ~5 ms."""
        noise = np.random.default_rng(0).standard_normal(8000).astype(np.float32)
        try:
            extract_gate_features(noise * 0.01, 16000, Acoustics(0.5, 1.0),
                                  Snapshot(since_ai=NEVER_SPOKEN))
        except Exception:
            pass

    def predict(self, features: np.ndarray) -> float:
        """p(assistant_directed) for one feature vector."""
        scaled = (np.asarray(features, dtype=np.float64) - self._scaler_mean) / self._scaler_scale
        logit = float(np.dot(scaled, self._weights) + self._bias)
        return _sigmoid(logit)

    # -- the decision ------------------------------------------------------

    def score(self, acoustics: Acoustics, snapshot: Snapshot,
              audio: np.ndarray | None = None, sample_rate: int = 16000,
              features: np.ndarray | None = None) -> Decision:
        """Score one segment. ``features`` short-circuits extraction (offline
        evaluation passes the stored vector, so it is judged by this exact
        code rather than a re-implementation of it)."""
        started = time.perf_counter()
        if self.mode == "off":
            return Decision(confidence=1.0, mode=self.mode)

        context_must = self._context_must(snapshot, acoustics)
        must = context_must or self._voice_must(acoustics)
        signals = self._signals(acoustics, snapshot)
        logit = sum(s.weight for s in signals)
        confidence = _sigmoid(logit)
        licensed = any(s.name in LICENSING for s in signals)
        rule_would_skip = (must is None and licensed
                           and confidence < self.limits["skip_below"])

        learned_conf: float | None = None
        learned_would_skip = False
        learned_reason = ""
        feature_latency = model_latency = 0.0
        have_model = self.model_data is not None and self._weights is not None

        # The fast path: when the answer is already "transcribe" and nothing
        # will read the score, do not spend the extraction on the critical
        # path. Shadow mode always scores -- measuring is its whole job.
        needed = (have_model and (self.mode == "shadow" or context_must is None)) \
            or self.log_features
        if needed and features is None:
            t0 = time.perf_counter()
            try:
                features = extract_gate_features(audio, sample_rate, acoustics, snapshot)
            except Exception as exc:
                features = None
                learned_reason = f"feature extraction failed: {type(exc).__name__}"
            feature_latency = time.perf_counter() - t0

        if not have_model:
            if self.use_learned:
                learned_reason = learned_reason or f"model unavailable ({self.model_error or 'not loaded'})"
        elif features is not None:
            t0 = time.perf_counter()
            try:
                learned_conf = round(self.predict(features), 4)
            except Exception as exc:
                learned_reason = f"model failed: {type(exc).__name__}"
            model_latency = time.perf_counter() - t0
            if learned_conf is not None:
                learned_reason = (context_must
                                  or self._feature_problem(features, audio)
                                  or ("" if self._threshold is not None
                                      else "model has no calibrated threshold"))
                learned_would_skip = (not learned_reason
                                      and learned_conf < self._threshold)
        elif not learned_reason and context_must:
            learned_reason = context_must

        if self.use_learned:
            # Model failure, doubtful features, a missing threshold: transcribe.
            would_skip = learned_would_skip
        else:
            would_skip = rule_would_skip

        decision = Decision(
            confidence=round(confidence, 4),
            signals=signals,
            would_skip=would_skip,
            skip=would_skip and self.mode == "skip",
            reason="" if would_skip else ((learned_reason if self.use_learned else must) or ""),
            mode=self.mode,
            latency=time.perf_counter() - started,
            rule_would_skip=rule_would_skip,
            learned_confidence=learned_conf,
            learned_would_skip=learned_would_skip,
            learned_reason=learned_reason,
            features=features,
            feature_latency=feature_latency,
            model_latency=model_latency,
        )
        self.scored += 1
        if must is not None:
            self.never_skip += 1
        if would_skip:
            self.would_skip += 1
        if decision.skip:
            self.skipped += 1
        return decision

    def _must_transcribe(self, acoustics: Acoustics, snapshot: Snapshot) -> str | None:
        """The short list. Any one of these and the answer is to look.

        Each is a situation where the next thing said is very likely for the
        assistant, or where not looking would lose something that cannot be
        recovered: a "yes" to a question nobody else will ask again.
        """
        return self._context_must(snapshot, acoustics) or self._voice_must(acoustics)

    def _context_must(self, snapshot: Snapshot, acoustics: Acoustics) -> str | None:
        """Reasons that bind the rules AND the learned model."""
        if snapshot.confirming:
            return "a confirmation is open"
        if snapshot.offer_pending:
            return "an offer is waiting for an answer"
        if snapshot.awaiting_answer:
            return "the assistant asked a question"
        if snapshot.since_ai < self.limits["cold_after"]:
            return "inside the follow-up window"
        if (self.always_transcribe_wearer and acoustics.voice_confident
                and acoustics.is_wearer):
            return "the enrolled voice was confidently detected"
        return None

    def _voice_must(self, acoustics: Acoustics) -> str | None:
        """The rules' own caution. The learned model sees voice_confident as
        a feature and was evaluated with it, so this does not bind it."""
        if not acoustics.voice_confident:
            # Cannot tell who spoke. Skipping on a guess about identity is
            # how the wearer's own "yeah" gets ignored.
            return "the voice could not be checked"
        return None

    @staticmethod
    def _feature_problem(features: np.ndarray, audio: np.ndarray | None) -> str:
        """Non-empty when the vector is not something to skip on."""
        if not np.all(np.isfinite(features)):
            return "non-finite features"
        if audio is not None and len(audio) == 0:
            return "no audio"
        if features[FEATURE_NAMES.index("voiced_ratio")] <= 0.0:
            return "nothing voiced in the segment"
        if features[FEATURE_NAMES.index("duration")] <= 0.0:
            return "zero-length segment"
        return ""

    def _signals(self, acoustics: Acoustics, snapshot: Snapshot) -> list[Signal]:
        w, lim = self.weights, self.limits
        out = [Signal("bias", w["bias"])]

        if snapshot.since_ai >= lim["cold_after"]:
            since = ("never spoken" if snapshot.since_ai >= NEVER_SPOKEN
                     else f"{snapshot.since_ai:.0f}s since the assistant spoke")
            out.append(Signal("cold_conversation", w["cold_conversation"], since))
            if snapshot.since_ai >= lim["very_cold_after"]:
                out.append(Signal("very_cold_conversation", w["very_cold_conversation"]))

        run = 0
        for verdict in reversed(snapshot.recent_verdicts):
            if verdict is False:
                run += 1
            else:
                break
        if run:
            capped = min(run, int(lim["ignored_run_cap"]))
            out.append(Signal("ignored_run", w["ignored_run"] * capped,
                              f"{run} ignored in a row"))
        elif snapshot.recent_verdicts and snapshot.recent_verdicts[-1] is True:
            out.append(Signal("accepted_recently", w["accepted_recently"]))

        if acoustics.voice_confident and acoustics.is_wearer is False:
            out.append(Signal("not_wearer", w["not_wearer"],
                              f"p_own={acoustics.p_own:.2f}" if acoustics.p_own is not None else ""))
        if acoustics.seconds >= lim["long_seconds"]:
            out.append(Signal("very_long", w["very_long"], f"{acoustics.seconds:.1f}s"))
        elif acoustics.seconds <= lim["short_seconds"]:
            out.append(Signal("very_short", w["very_short"], f"{acoustics.seconds:.2f}s"))
        if acoustics.confidence < lim["low_confidence"]:
            out.append(Signal("low_vad_confidence", w["low_vad_confidence"],
                              f"{acoustics.confidence:.2f}"))
        return out

    # -- the cheap half, usable at speech START ----------------------------

    def worth_acknowledging(self, snapshot: Snapshot) -> bool:
        """Whether the "Mm?" check is worth a look, from conversation alone.

        The acknowledgement runs 600 ms into an utterance, before there is a
        segment to judge, so only the conversational signals exist. Measured
        on this machine it costs about 850 ms of the main recogniser on every
        utterance -- including the 63% that are then ignored -- because the
        small model it was meant to use cannot coexist with the big one. In a
        cold conversation with a run of ignored speech behind it, that is a
        look nobody asked for.
        """
        if self.mode == "off":
            return True
        if self._must_transcribe(Acoustics(seconds=1.0, confidence=1.0,
                                           voice_confident=True), snapshot) is not None:
            return True
        run = 0
        for verdict in reversed(snapshot.recent_verdicts):
            if verdict is False:
                run += 1
            else:
                break
        cold = snapshot.since_ai >= self.limits["very_cold_after"]
        return not (cold and run >= 2)

    def worth_head_start(self, snapshot: Snapshot, seconds: float) -> bool:
        """Whether to start transcribing during the hangover.

        Skip mode only. The head start begins at 250 ms of silence, before the
        segment closes and before the voiceprint has run, so it is judged on
        conversation and duration alone with the voice assumed to be the
        wearer's. Losing a head start in the rare disagreement costs 300 ms;
        losing an utterance costs the utterance -- so this is vetoed only
        where the final gate would skip anyway.
        """
        if self.mode != "skip":
            return True
        guess = Acoustics(seconds=seconds, confidence=1.0,
                          voice_confident=True, is_wearer=True)
        must = self._must_transcribe(guess, snapshot)
        if must is not None:
            return True
        signals = self._signals(guess, snapshot)
        if not any(s.name in LICENSING for s in signals):
            return True                  # nothing about the sound argues yet
        logit = sum(s.weight for s in signals)
        return _sigmoid(logit) >= self.limits["skip_below"]

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "mode": self.mode,
            "learned": self.use_learned,
            "model_loaded": self.model_data is not None,
            "model_error": self.model_error,
            "scored": self.scored,
            "would_skip": self.would_skip,
            "skipped": self.skipped,
            "never_skip": self.never_skip,
        }


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    z = math.exp(logit)
    return z / (1.0 + z)


def verdicts_from(utterances: Sequence, limit: int = 4) -> tuple:
    """The recent verdicts, newest last, from ConversationContext.utterances()."""
    tail = list(utterances)[-limit:]
    return tuple(getattr(u, "ai_directed", None) for u in tail)
