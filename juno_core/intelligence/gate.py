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
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from juno_core.intelligence.features import extract_gate_features

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
# moment it is written down. json.dumps writes a bare Infinity, which Python
# reads back and no other parser will; the observer's encoder writes null,
# which made float() raise. Both happened: the retention sidecars were invalid
# JSON, and scripts/gate_eval.py crashed on the FIRST utterance of every
# session, which is the one measurement the whole shadow mode exists to
# collect. So the conversion happens once, here, at the edge where a number
# becomes data.
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
    learned_confidence: float | None = None
    learned_would_skip: bool = False


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
        self._observer = observer

        # Phase 3 learned layer
        model_path_cfg = get("model_path")
        self.model_path = Path(model_path_cfg) if model_path_cfg else DEFAULT_MODEL_PATH
        self.model_data: dict | None = None
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0
        self._scaler_mean: np.ndarray | None = None
        self._scaler_scale: np.ndarray | None = None
        self._threshold: float = 0.25
        self._load_learned_model()

        # What it has done, for /status. Shadow counts what it would have done.
        self.scored = 0
        self.would_skip = 0
        self.skipped = 0
        self.never_skip = 0

    def _load_learned_model(self) -> None:
        if not self.model_path.exists():
            return
        try:
            data = json.loads(self.model_path.read_text(encoding="utf-8"))
            self._weights = np.asarray(data["weights"], dtype=np.float32)
            self._bias = float(data["bias"])
            self._scaler_mean = np.asarray(data["scaler_mean"], dtype=np.float32)
            self._scaler_scale = np.asarray(data["scaler_scale"], dtype=np.float32)
            self._threshold = float(data.get("threshold", 0.25))
            self.model_data = data
        except Exception:
            self.model_data = None

    # -- the decision ------------------------------------------------------

    def score(self, acoustics: Acoustics, snapshot: Snapshot,
              audio: np.ndarray | None = None, sample_rate: int = 16000) -> Decision:
        started = time.perf_counter()
        if self.mode == "off":
            return Decision(confidence=1.0, mode=self.mode)

        must = self._must_transcribe(acoustics, snapshot)
        signals = self._signals(acoustics, snapshot)
        logit = sum(s.weight for s in signals)
        confidence = 1.0 / (1.0 + math.exp(-logit))
        licensed = any(s.name in LICENSING for s in signals)
        rule_would_skip = (must is None and licensed
                           and confidence < self.limits["skip_below"])

        # Learned model inference if available
        learned_conf: float | None = None
        learned_would_skip = False
        if self.model_data is not None and self._weights is not None:
            features = extract_gate_features(audio, sample_rate, acoustics, snapshot)
            scaled = (features - self._scaler_mean) / np.maximum(self._scaler_scale, 1e-8)
            l_logit = float(np.dot(scaled, self._weights) + self._bias)
            learned_conf = round(1.0 / (1.0 + math.exp(-l_logit)), 4)
            learned_would_skip = (must is None and learned_conf < self._threshold)

        # If learned mode is enabled, it governs would_skip; otherwise rules gate governs
        if self.use_learned and learned_conf is not None:
            would_skip = (must is None and learned_would_skip)
        else:
            would_skip = rule_would_skip

        decision = Decision(
            confidence=round(confidence, 4),
            signals=signals,
            would_skip=would_skip,
            skip=would_skip and self.mode == "skip",
            reason=must or "",
            mode=self.mode,
            latency=time.perf_counter() - started,
            learned_confidence=learned_conf,
            learned_would_skip=learned_would_skip,
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
        if snapshot.confirming:
            return "a confirmation is open"
        if snapshot.offer_pending:
            return "an offer is waiting for an answer"
        if snapshot.awaiting_answer:
            return "the assistant asked a question"
        if snapshot.since_ai < self.limits["cold_after"]:
            return "inside the follow-up window"
        if not acoustics.voice_confident:
            # Cannot tell who spoke. Skipping on a guess about identity is
            # how the wearer's own "yeah" gets ignored.
            return "the voice could not be checked"
        return None

    def _signals(self, acoustics: Acoustics, snapshot: Snapshot) -> list[Signal]:
        w, lim = self.weights, self.limits
        out = [Signal("bias", w["bias"])]

        if snapshot.since_ai >= lim["cold_after"]:
            out.append(Signal("cold_conversation", w["cold_conversation"],
                              f"{snapshot.since_ai:.0f}s since the assistant spoke"))
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
        return 1.0 / (1.0 + math.exp(-logit)) >= self.limits["skip_below"]

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "mode": self.mode,
            "scored": self.scored,
            "would_skip": self.would_skip,
            "skipped": self.skipped,
            "never_skip": self.never_skip,
        }


def verdicts_from(utterances: Sequence, limit: int = 4) -> tuple:
    """The recent verdicts, newest last, from ConversationContext.utterances()."""
    tail = list(utterances)[-limit:]
    return tuple(getattr(u, "ai_directed", None) for u in tail)
