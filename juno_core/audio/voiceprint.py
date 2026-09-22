"""Was that *you*, wherever you were standing?

This replaces the distance estimator that used to live in audio/ownvoice.py.
That one asked how near the microphone somebody was, which is a good proxy for
"the wearer" right up until the wearer walks across the room -- and then it is
exactly backwards. Measured on the shipping detector with real speech and a
simulated room:

    you, close to the mic     0.51 - 0.82
    you, across the room      0.36 - 0.48
    somebody else, close      0.39 - 0.81

The middle row sits entirely below the bottom one. No threshold admits a
distant wearer and blocks a nearby stranger, so it was never a matter of
tuning: the ordering itself was wrong for anyone who wants to be able to speak
from the other side of the room.

WHAT THIS DOES INSTEAD
----------------------
A speaker-verification embedding -- WeSpeaker's ECAPA-TDNN, trained on
VoxCeleb -- turns a few seconds of speech into 192 numbers that describe the
voice rather than the acoustics. Enrol once, then score the cosine similarity
of everything after. Same speech and same simulated room as above:

    you, near                +0.800 .. +0.905
    you, across the room     +0.569 .. +0.786
    you, very far            +0.419 .. +0.739
    three other voices, near +0.059 .. +0.192

Separated by +0.227 at the worst point, because reverberation changes how a
voice arrives and not whose voice it is. The model was trained with reverb and
noise augmentation, which is the difference between this and the two
hand-rolled spectral features that were tried first: both of those were
destroyed by the same reverb this shrugs off.

(Those are the numbers from the filterbank below rather than from librosa's,
which scored the same clips slightly differently -- +0.144 at the worst point.
Both separate; this one is what ships, so this one is quoted.)

Costs 13 ms on a median utterance and 23 ms to load, against roughly 976 ms
for the transcription it sits in front of. The model is 24 MB of ONNX, run on
the onnxruntime that Silero VAD already requires, so this adds a file and not
a dependency -- which matters, because the half of Juno worth publishing on
its own should stay installable as numpy and one runtime.

THIS ONE DOES KEEP SOMETHING
----------------------------
The old detector's honest boast was that it stored no model of anybody's
voice, because it did not need one -- it measured level and reverberation and
kept two numbers. That is no longer true and should not be quietly dropped
from the copy: enrolment writes 192 floats derived from recordings of you.
That is a biometric template. It never leaves the machine, no audio is kept,
it is a plain JSON file you can read, and deleting it switches the gate off.
But it exists, and anything the interface says about this should say so.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# WeSpeaker's ECAPA-TDNN, 512 channels, VoxCeleb + large-margin finetune
# (CC BY 4.0). Pinned and downloaded on first use by juno_core/assets.py.
DEFAULT_PROFILE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "voiceprint.json"
)

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400          # 25 ms
HOP = 160            # 10 ms
FMIN, FMAX = 20.0, 7600.0
# Below this there is not enough voiced speech for an embedding to mean
# anything, and scoring it produces confident nonsense.
#
# 0.35 rather than the 0.6 this started at, because the answers to a spoken
# question are the shortest things anybody says: "yeah" is 0.43 s and "sure"
# is 0.53 s, and refusing to score them meant the two most natural ways of
# agreeing came back as "that was not your voice". Measured on seven
# single-word answers, all of them under the old floor, the wearer scored
# +0.324 to +0.572 and two other speakers -0.074 to +0.178 -- separated every
# time.
MIN_SECONDS = 0.35

# What a short utterance does to the scale, and why one threshold will not do.
# The same voice scores +0.876 on a sentence and +0.324 on the word "sure":
# there is simply less evidence in half a second, and it pulls every score
# towards the middle. A threshold picked on ordinary speech therefore silences
# short answers, which is exactly backwards -- a short answer is usually
# somebody replying to a direct question.
#
# So the line is scaled by how much was heard, down to half of it. The floor
# is not lower than that because the gap between the wearer and somebody else
# stays wide even at 0.4 s -- the narrowest measured was 0.165 -- so there is
# no need to go further, and going further would start admitting the room.
SCALE_FULL_SECONDS = 1.2
SCALE_FLOOR = 0.5


def current_microphone() -> str:
    """The input device's name, or "" when it cannot be read.

    Empty is deliberately not a match for anything: a threshold is only
    meaningful on the microphone it was measured on, and "I cannot tell"
    must not read as "the right one".
    """
    try:
        from juno_core.audio.coreaudio import CoreAudio

        device = CoreAudio().describe_input()
    except Exception:
        return ""
    return (device.name if device else "") or ""


@dataclass(frozen=True)
class VoiceEstimate:
    """``p_own`` keeps the name and the range the intent engine already takes.

    Everything downstream reads a single number between 0 and 1 where low
    means "probably not the wearer", so replacing what produces it changes
    nothing above this line.
    """

    p_own: float
    similarity: float = 0.0
    confident: bool = False
    # The threshold this particular utterance should be judged against, scaled
    # for how long it was. Carried here so that no caller has to remember to
    # scale it, and none of them can disagree about how.
    threshold: float = 0.0
    seconds: float = 0.0

    @property
    def is_wearer(self) -> bool:
        return self.confident and self.similarity >= self.threshold

    @property
    def evidence_against(self) -> float:
        return 1.0 - self.p_own


# -- features, in numpy, because the dependency list is a design decision ----

def _mel_filterbank(sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT,
                    n_mels: int = N_MELS) -> np.ndarray:
    """Slaney-style triangular mel filters. Roughly thirty lines rather than
    a dependency: librosa would do this, and would also be the only reason
    this module needed librosa."""
    def to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    points = to_hz(np.linspace(to_mel(FMIN), to_mel(FMAX), n_mels + 2))
    bins = np.floor((n_fft + 1) * points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        left, centre, right = bins[m], bins[m + 1], bins[m + 2]
        if centre == left:
            centre = left + 1
        if right == centre:
            right = centre + 1
        if right > n_fft // 2:
            break
        filters[m, left:centre] = np.linspace(0, 1, centre - left, endpoint=False)
        filters[m, centre:right] = np.linspace(1, 0, right - centre, endpoint=False)
    return filters


_FILTERS = _mel_filterbank()
_WINDOW = np.hanning(N_FFT).astype(np.float32)


def fbank(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Log-mel filterbank, mean-normalised per coefficient.

    The mean subtraction is what makes this survive a change of microphone or
    a change of distance: a constant offset in the log domain is a gain or a
    channel colour, and removing it removes most of both.
    """
    audio = np.asarray(audio, dtype=np.float32).ravel()
    if audio.size < N_FFT:
        return np.zeros((0, N_MELS), dtype=np.float32)
    frames = 1 + (audio.size - N_FFT) // HOP
    strided = np.lib.stride_tricks.as_strided(
        audio, shape=(frames, N_FFT),
        strides=(audio.strides[0] * HOP, audio.strides[0]),
    ) * _WINDOW
    power = np.abs(np.fft.rfft(strided, n=N_FFT, axis=1)) ** 2
    mel = power.astype(np.float32) @ _FILTERS.T
    logmel = np.log(np.maximum(mel, 1e-10))
    return (logmel - logmel.mean(axis=0, keepdims=True)).astype(np.float32)


class VoicePrint:
    """Embeds speech, and says how like the enrolled voice it is.

    The model is loaded on first use rather than at startup: a session that
    never has the gate switched on should not pay 24 MB for it.
    """

    def __init__(self, config=None, observer=None) -> None:
        get = (config.get if config is not None else (lambda k, d=None: d))
        self.enabled = bool(get("enabled", False))
        # Cosine similarity below which a segment is treated as evidence the
        # wearer did not speak. Written by the calibration; the default is a
        # starting point, not a result.
        self.threshold = float(get("threshold", 0.55))
        from juno_core.assets import default_path

        configured = get("model", "") or ""
        # Only the default location is downloaded into; a path you set is yours.
        self._auto_download = not configured
        self.model_path = Path(configured).expanduser() if configured else default_path("ecapa")
        self.profile_path = Path(get("profile", "") or DEFAULT_PROFILE_PATH)
        self.calibrated_for = str(get("calibrated_for", "") or "").strip()
        self.device = ""
        self._observer = observer
        self._session = None
        self._lock = threading.Lock()
        self._profile: np.ndarray | None = None

        if self.enabled:
            self._profile = self.load_profile()
            if self._profile is None:
                self.enabled = False
                self._emit("voiceprint_not_enrolled", profile=str(self.profile_path))
            elif self.calibrated_for:
                # Voice survives a change of microphone far better than level
                # does, but not perfectly: a bone-conduction pickup and a
                # laptop across a desk are not the same signal. The stamp is
                # kept for the same reason as before, and unreadable still
                # counts as a mismatch.
                self.device = current_microphone()
                if self.device != self.calibrated_for:
                    self.enabled = False
                    self._emit("voiceprint_device_mismatch",
                               calibrated_for=self.calibrated_for,
                               now=self.device or "unknown")

    # -- the model ---------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._fetch()

    def _fetch(self) -> bool:
        """True once the model file is on disk, downloading it if it's ours to."""
        if self.model_path.exists():
            return True
        if not self._auto_download:
            return False
        from juno_core.assets import AssetError, ensure

        try:
            ensure("ecapa", self.model_path)
        except AssetError as exc:
            self._emit("voiceprint_download_failed", error=str(exc)[:200])
            return False
        return True

    def _session_or_none(self):
        if self._session is not None:
            return self._session
        with self._lock:
            if self._session is None:
                if not self._fetch():
                    return None
                try:
                    import onnxruntime as ort

                    options = ort.SessionOptions()
                    # It runs beside transcription; taking every core would
                    # make the thing it is trying not to delay slower.
                    options.intra_op_num_threads = 1
                    self._session = ort.InferenceSession(
                        str(self.model_path), options,
                        providers=["CPUExecutionProvider"],
                    )
                except Exception as exc:
                    self._emit("voiceprint_load_failed", error=str(exc)[:120])
                    self.enabled = False
                    return None
        return self._session

    def embed(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE
              ) -> np.ndarray | None:
        """A unit-length 192-d embedding, or None if it cannot be computed."""
        audio = np.asarray(audio, dtype=np.float32).ravel()
        if audio.size < int(MIN_SECONDS * sample_rate):
            return None
        session = self._session_or_none()
        if session is None:
            return None
        features = fbank(audio, sample_rate)
        if features.shape[0] < 20:
            return None
        try:
            embedding = session.run(None, {"feats": features[None]})[0][0]
        except Exception as exc:
            self._emit("voiceprint_failed", error=str(exc)[:120])
            return None
        norm = float(np.linalg.norm(embedding))
        if norm < 1e-6:
            return None
        return (embedding / norm).astype(np.float32)

    # -- enrolment ---------------------------------------------------------

    def enrol(self, clips: list[np.ndarray], sample_rate: int = SAMPLE_RATE
              ) -> np.ndarray | None:
        """Average the embeddings of several clips into one profile.

        Averaging unit vectors and renormalising is the standard way to build
        a speaker profile: it keeps what the clips agree on about the voice
        and cancels what they disagree on, which is mostly the words.
        """
        embeddings = [e for e in (self.embed(c, sample_rate) for c in clips)
                      if e is not None]
        if len(embeddings) < 2:
            return None
        profile = np.mean(embeddings, axis=0)
        norm = float(np.linalg.norm(profile))
        if norm < 1e-6:
            return None
        return (profile / norm).astype(np.float32)

    def save_profile(self, profile: np.ndarray, device: str = "") -> None:
        """Readable JSON on purpose: it is a template derived from somebody's
        voice, and they should be able to look at it and delete it."""
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(json.dumps({
            "note": ("A voice template, kept on this machine only. No audio "
                     "was stored. Delete this file to switch own-voice "
                     "gating off."),
            "dimensions": int(profile.size),
            "device": device,
            "embedding": [round(float(x), 6) for x in profile],
        }, indent=2), encoding="utf-8")
        self._profile = profile

    def load_profile(self) -> np.ndarray | None:
        try:
            data = json.loads(self.profile_path.read_text(encoding="utf-8"))
            vector = np.asarray(data["embedding"], dtype=np.float32)
        except Exception:
            return None
        norm = float(np.linalg.norm(vector))
        if vector.size < 64 or norm < 1e-6:
            return None
        return (vector / norm).astype(np.float32)

    # -- the question everything else asks ---------------------------------

    def similarity(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE
                   ) -> float | None:
        if self._profile is None:
            return None
        embedding = self.embed(audio, sample_rate)
        if embedding is None:
            return None
        return float(np.dot(embedding, self._profile))

    def threshold_for(self, seconds: float) -> float:
        """The line, scaled by how much speech there was to judge."""
        share = min(1.0, max(SCALE_FLOOR, seconds / SCALE_FULL_SECONDS))
        return self.threshold * max(SCALE_FLOOR, share)

    def estimate(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE
                 ) -> VoiceEstimate:
        """No-confidence is 1.0, not 0.5.

        The score is one-sided by construction -- it is only ever used as
        evidence against acting -- so "I could not tell" has to mean "no
        evidence", and no evidence against is the top of the range. Returning
        a middling number for silence would quietly argue that every unclear
        segment was somebody else.
        """
        if not self.enabled or self._profile is None:
            return VoiceEstimate(p_own=1.0)
        seconds = len(np.asarray(audio).ravel()) / float(sample_rate)
        score = self.similarity(audio, sample_rate)
        if score is None:
            return VoiceEstimate(p_own=1.0, seconds=round(seconds, 2))
        line = self.threshold_for(seconds)
        # A logistic centred on the measured threshold, so p_own crosses 0.5
        # exactly where the calibration put the line. The slope is fixed: it
        # decides how quickly doubt turns into certainty, and there is not
        # enough data from eight clips to fit it as well as to assert it.
        p_own = 1.0 / (1.0 + math.exp(-(score - line) / 0.08))
        return VoiceEstimate(p_own=round(p_own, 4), similarity=round(score, 4),
                             confident=True, threshold=round(line, 4),
                             seconds=round(seconds, 2))

    def _emit(self, name: str, **fields) -> None:
        if self._observer is None:
            return
        from juno_core.events import Stage

        self._observer.emit(Stage.AUDIO, name, None, **fields)
