"""What the studio does, independent of how it is shown.

One object owns the microphone, the models and the state of the three
activities the browser page offers:

  TRY       listen continuously; for every utterance show what Reflex
            decided from the audio and what the cascade made of the words,
            side by side. Optionally mark what was actually meant, and save
            those marks as a small real-world test set.
  COLLECT   a guided, consented session: prompt by prompt ("ask Juno the
            time, a few ways"; "talk to each other"), recorded for a fixed
            time, every utterance labelled by its prompt. Plus free
            recording of any one label -- for minimal pairs, say.
  RESULTS   evaluate a model on the sessions collected, or train a new one
            with them.

What leaves memory: nothing, unless a session is saved -- and then only what
collect.py keeps (encoder vectors, the teacher's transcript and verdict, the
label, provenance). Audio is encoded and released, exactly as at runtime.

Everything here runs on a few threads: the capture loop (frames -> VAD ->
segments), ONE worker that does all model work, and a training thread
(numpy only). The single worker is not a simplification: MLX keeps its
streams per thread, so a model loaded on one thread cannot run on another.
Loading, switching models and every segment therefore go through the same
queue. The page learns what happened through ``events``, a fan-out of
(kind, data) pairs the server streams.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
import time
import uuid
from argparse import Namespace
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from juno_core.slu import data as D
from juno_core.slu.collect import SLU_SCRIPT, consented_row, write_session
from juno_core.slu.router import ConversationState
from juno_core.slu.schema import ADDRESSEES, ASSISTANT, CORE_SCHEMA, OPEN_REQUEST, SOURCE_REFLEX

DEFAULT_ENCODERS = ("parakeet", "logmel", "gate")
RATE = 16000
FRAME = 512
KEEP_RECENT = 200


class Broadcaster:
    """Fan-out of events to every open page. A slow page loses events, never blocks."""

    def __init__(self) -> None:
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, kind: str, data) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait((kind, data))
            except queue.Full:
                pass


class SimulatedCapture:
    """WAV files played as if they were a microphone: a clip, then silence, in real time.

    For demos and for trying the studio without speaking -- and for its tests.
    """

    def __init__(self, paths: Sequence, gap: float = 1.5, realtime: bool = True,
                 loop: bool = True) -> None:
        clips = []
        for p in paths:
            if isinstance(p, np.ndarray):          # audio already in memory (tests)
                clips.append(p.astype(np.float32))
                continue
            p = Path(p)
            files = sorted(p.glob("*.wav")) if p.is_dir() else [p]
            for f in files:
                audio, rate = D.read_wav(f)
                if rate == RATE:
                    clips.append(audio)
        if not clips:
            raise ValueError("no 16 kHz WAV files to simulate a microphone with")
        silence = np.zeros(int(gap * RATE), np.float32)
        noise = np.random.default_rng(0)
        parts = []
        for clip in clips:
            parts += [silence + noise.standard_normal(silence.size).astype(np.float32) * 1e-4, clip]
        parts.append(silence)
        self.stream = np.concatenate(parts).astype(np.float32)
        self.realtime, self.loop = realtime, loop
        self._running = threading.Event()

    def start(self) -> None:
        self._running.set()

    def stop(self) -> None:
        self._running.clear()

    def frames(self):
        from juno_core.audio.capture import AudioFrame

        i, n = 0, self.stream.size
        t0 = time.monotonic()
        k = 0
        while self._running.is_set():
            if i + FRAME > n:
                if not self.loop:
                    return
                i = 0
            yield AudioFrame(samples=self.stream[i:i + FRAME], timestamp=time.monotonic(),
                             adc_time=0.0, index=k)
            i += FRAME
            k += 1
            if self.realtime:
                delay = t0 + k * FRAME / RATE - time.monotonic()
                if delay > 0:
                    time.sleep(delay)


@dataclass
class Step:
    index: int
    label: str
    intent: str | None
    text: str
    seconds: float
    speaker: str
    who: str


@dataclass
class Tag:
    """What the utterances being recorded right now are, by construction."""

    label: str
    intent: str | None
    speaker: str
    step: int | None = None


@dataclass
class Session:
    id: str
    speakers: list[str]
    room: str
    mic: str
    consent: str
    encoders: list[str]
    steps: list[Step]
    rows: list[dict] = field(default_factory=list)
    vectors: dict[str, list] = field(default_factory=dict)
    current: int = 0
    n: int = 0

    def summary(self) -> dict:
        counts: dict = {}
        for r in self.rows:
            counts[r["g_addressed"]] = counts.get(r["g_addressed"], 0) + 1
        per_step: dict = {}
        for r in self.rows:
            if r.get("step") is not None:
                per_step[r["step"]] = per_step.get(r["step"], 0) + 1
        return {"id": self.id, "speakers": self.speakers, "room": self.room, "mic": self.mic,
                "consent": self.consent, "encoders": self.encoders,
                "steps": [asdict(s) for s in self.steps], "current": self.current,
                "rows": len(self.rows), "counts": counts, "per_step": per_step,
                "segments": [_public_row(r) for r in self.rows[-60:]]}


def _public_row(r: dict) -> dict:
    return {k: r.get(k) for k in ("id", "g_addressed", "g_intent", "speaker", "step",
                                  "transcript", "t_accept", "t_intent", "duration")}


def build_steps(speakers: Sequence[str]) -> list[Step]:
    """The script, expanded per speaker, as the CLI collector does it."""
    from juno_core.intelligence.gate_collect import SOLO_TEXT

    solo = len(speakers) == 1
    steps: list[Step] = []
    for prompt in SLU_SCRIPT:
        text = SOLO_TEXT.get(prompt.label, prompt.text) if solo and prompt.who == "all" else prompt.text
        if prompt.who == "each":
            targets = list(speakers)
        elif prompt.who == "all":
            targets = ["+".join(speakers)]
        else:
            targets = [""]
        for speaker in targets:
            steps.append(Step(len(steps), prompt.label, getattr(prompt, "intent", None), text,
                              float(prompt.seconds), speaker, prompt.who))
    return steps


class Studio:
    def __init__(self, config=None, *, root: str | Path = ".", stt: str = "parakeet-0.6b",
                 aliases: Sequence[str] = (), name: str = "Juno",
                 encoders: Sequence[str] = DEFAULT_ENCODERS, model: str | None = None,
                 simulate: Sequence | None = None, realtime: bool = True,
                 teacher=None, judge: str | None = None) -> None:
        self.config = config
        self.root = Path(root).resolve()
        self.models_dir = self.root / "models"
        self.data_dir = self.root / "data" / "slu"
        self.out_dir = self.data_dir / "studio"
        self.stt_name = stt
        self.judge_spec = judge
        self.aliases = list(aliases)
        self.name = name
        self.encoder_specs = list(encoders)
        self.initial_model = model
        self.simulate = list(simulate) if simulate else None
        self.realtime = realtime
        self.events = Broadcaster()
        self._lock = threading.RLock()

        self.loading: dict[str, str] = {}
        self.teacher = teacher
        self.encoders: dict = {}           # spec -> AudioEncoder
        self.reflex = None
        self.model_info: dict | None = None
        self.voice = None

        self.listening = False
        self.purpose: str | None = None     # "try" | "collect"
        self._capture = None
        self._capture_thread: threading.Thread | None = None
        self._work: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._work_loop, daemon=True, name="studio-worker")
        self._worker.start()

        self.state = {"awaiting_answer": False, "timer_running": False}
        self.recent: OrderedDict[str, dict] = OrderedDict()
        self.labelled: list[tuple[dict, dict]] = []   # (row, {spec: vector})
        self.stats = {"utterances": 0, "decided_by_reflex": 0, "agree": 0, "compared": 0,
                      "reflex_ms": []}

        self.session: Session | None = None
        self.recording: dict | None = None  # {"tag": Tag, "deadline": float | None}
        self.job: dict | None = None

    # -- status ----------------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            reflex_ms = self.stats["reflex_ms"][-200:]
            return {
                "ready": self.teacher is not None and bool(self.encoders),
                "loading": dict(self.loading),
                "listening": self.listening, "purpose": self.purpose,
                "simulated": bool(self.simulate),
                "mic": self._mic_name(),
                "teacher": self.stt_name + (" + judge" if self.judge_spec else ""),
                "encoders": list(self.encoders),
                "model": self.model_info,
                "state": dict(self.state),
                "stats": {
                    "utterances": self.stats["utterances"],
                    "stt_avoided": round(self.stats["decided_by_reflex"] /
                                         max(1, self.stats["utterances"]), 3),
                    "agreement": round(self.stats["agree"] / self.stats["compared"], 3)
                    if self.stats["compared"] else None,
                    "reflex_ms_p50": round(float(np.median(reflex_ms)), 1) if reflex_ms else None,
                },
                "labelled": len(self.labelled),
                "session": self.session.summary() if self.session else None,
                "recording": self._recording_public(),
                "job": dict(self.job) if self.job else None,
                "voiceprint": bool(self.voice is not None and self.voice.enabled),
                "intents": list(CORE_SCHEMA.intent_names),
                "addressees": list(ADDRESSEES),
            }

    def _push_status(self) -> None:
        self.events.publish("status", self.status())

    def _mic_name(self) -> str:
        if self.simulate:
            names = [Path(p).name if not isinstance(p, np.ndarray) else "audio"
                     for p in self.simulate]
            return "simulated (" + ", ".join(names) + ")"
        try:
            from juno_core.audio.voiceprint import current_microphone

            return current_microphone() or "default input"
        except Exception:
            return "default input"

    # -- loading -----------------------------------------------------------------

    def load(self, background: bool = True) -> None:
        """Teacher, encoders, voiceprint and the newest model. Slow the first time."""
        future = self._submit(self._load)
        if not background:
            future.result()

    def _submit(self, fn) -> Future:
        """Run ``fn`` on the worker thread -- the only thread that touches models."""
        future: Future = Future()
        self._work.put(("call", fn, future))
        return future

    def _load(self) -> None:
        from juno_core.audio.voiceprint import VoicePrint
        from juno_core.slu.encoder import build_encoder
        from juno_core.slu.training import build_teacher

        def step(name, fn):
            self.loading[name] = "loading"
            self._push_status()
            try:
                fn()
                self.loading[name] = "ready"
            except Exception as exc:          # one missing piece must not sink the rest
                self.loading[name] = f"failed: {type(exc).__name__}: {exc}"
                self.events.publish("problem", {"message": f"{name}: {exc}"})
            self._push_status()

        def teacher():
            args = Namespace(stt=self.stt_name, llm=None, schema=None, name=self.name,
                             aliases=",".join(self.aliases), config=None,
                             judge=self.judge_spec, judge_weight=0.8)
            t, _ = build_teacher(args)
            t.stt.warmup()
            self.teacher = t

        def encoders():
            for spec in self.encoder_specs:
                enc = build_encoder(spec)
                enc.warmup()
                self.encoders[enc.spec] = enc

        def voiceprint():
            section = (self.config.get("own_voice") if self.config is not None else None) or {}
            self.voice = VoicePrint(section)

        step("voiceprint", voiceprint)
        step("encoders", encoders)
        if self.teacher is None:
            step("teacher", teacher)
        else:
            self.loading["teacher"] = "ready"
        model = self.initial_model or self._newest_model()
        if model:
            step("model", lambda: self._set_model(model))

    def _newest_model(self) -> str | None:
        models = sorted(self.models_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        return str(models[-1]) if models else None

    def set_model(self, path: str) -> dict:
        return self._submit(lambda: self._set_model(path)).result(timeout=600)

    def _set_model(self, path: str) -> dict:
        from juno_core.slu.encoder import build_encoder
        from juno_core.slu.student import StudentModel
        from juno_core.slu.reflex import Reflex

        model = StudentModel.load(self._resolve(path))
        encoder = self.encoders.get(model.encoder_spec)
        if encoder is None:
            encoder = build_encoder(model.encoder_spec)
            encoder.warmup()
            self.encoders[encoder.spec] = encoder
        reflex = Reflex({"mode": "on"}, model=model, encoder=encoder)
        if not reflex.enabled:
            raise ValueError(reflex.error)
        with self._lock:
            self.reflex = reflex
            self.model_info = _model_info(self._resolve(path), model)
        self._push_status()
        return self.model_info

    def _resolve(self, path: str | Path) -> Path:
        p = Path(path).expanduser()
        return p if p.is_absolute() else self.root / p

    # -- the microphone ------------------------------------------------------------

    def _start_capture(self, purpose: str) -> None:
        from juno_core.audio.capture import build_capture
        from juno_core.audio.vad import SegmentDetector, build_vad
        from juno_core.config import Section

        with self._lock:
            if self.listening:
                if self.purpose != purpose:
                    raise RuntimeError(f"already listening for {self.purpose}")
                return
            audio_cfg = (self.config.get("audio") if self.config is not None else None) or \
                Section({"sample_rate": RATE, "channels": 1, "frame_ms": 32})
            vad_cfg = (self.config.get("vad") if self.config is not None else None) or Section(
                {"backend": "silero", "threshold": 0.5, "min_speech_ms": 200,
                 "min_silence_ms": 550, "min_segment_ms": 350, "max_segment_ms": 20000})
            if self.simulate:
                capture = SimulatedCapture(self.simulate, realtime=self.realtime)
            else:
                capture = build_capture(audio_cfg)
            detector = SegmentDetector(build_vad(vad_cfg, RATE), vad_cfg, RATE, preroll=0.4)
            capture.start()
            self._capture = capture
            self.listening, self.purpose = True, purpose
            self._capture_thread = threading.Thread(
                target=self._capture_loop, args=(capture, detector), daemon=True,
                name="studio-capture")
            self._capture_thread.start()
        self._push_status()

    def _stop_capture(self) -> None:
        with self._lock:
            capture, self._capture = self._capture, None
            self.listening, self.purpose = False, None
        if capture is not None:
            capture.stop()
        self._push_status()

    def _capture_loop(self, capture, detector) -> None:
        last_level = 0.0
        peak = 0.0
        try:
            for frame in capture.frames():
                if self._capture is not capture:
                    break
                peak = max(peak, float(np.sqrt(np.mean(np.square(frame.samples)))))
                now = time.monotonic()
                if now - last_level >= 0.1:
                    db = 20 * np.log10(max(peak, 1e-6))
                    self.events.publish("level", {"db": round(float(db), 1)})
                    last_level, peak = now, 0.0
                segment = detector.push(frame)
                if segment is not None:
                    self._work.put(("segment", self.purpose, segment, self._current_tag()))
                rec = self.recording
                if rec and rec.get("deadline") and now >= rec["deadline"]:
                    self.stop_recording()
        except Exception as exc:
            self.events.publish("problem", {"message": f"microphone: {type(exc).__name__}: {exc}"})
            self._stop_capture()

    def _current_tag(self) -> Tag | None:
        rec = self.recording
        return rec["tag"] if rec else None

    # -- the worker ----------------------------------------------------------------

    def _work_loop(self) -> None:
        while True:
            item = self._work.get()
            if item[0] == "call":
                _, fn, future = item
                try:
                    future.set_result(fn())
                except BaseException as exc:
                    future.set_exception(exc)
                continue
            _, purpose, segment, tag = item
            try:
                if purpose == "try":
                    self._handle_try(segment)
                elif purpose == "collect" and tag is not None:
                    self._handle_collect(segment, tag)
            except Exception as exc:
                self.events.publish("problem", {"message": f"{type(exc).__name__}: {exc}"})

    def _not_wearer(self, audio) -> bool:
        if self.voice is None or not self.voice.enabled:
            return False
        estimate = self.voice.estimate(audio, RATE)
        return bool(estimate.confident and not estimate.is_wearer)

    def _encode_all(self, audio) -> tuple[dict, dict]:
        vectors, ms = {}, {}
        for spec, enc in list(self.encoders.items()):
            t0 = time.perf_counter()
            vectors[spec] = enc.encode(audio, RATE)
            ms[spec] = (time.perf_counter() - t0) * 1000.0
        return vectors, ms

    def _handle_try(self, segment) -> None:
        uid = uuid.uuid4().hex[:8]
        audio = segment.audio
        if self._not_wearer(audio):
            self.events.publish("utterance", {"id": uid, "dropped": "not your voice (voiceprint)",
                                              "duration": round(float(segment.duration), 2)})
            return
        vectors, ms = self._encode_all(audio)
        reflex = None
        if self.reflex is not None:
            spec = self.reflex.encoder.spec
            state = ConversationState(
                since_ai=2.0 if self.state["awaiting_answer"] else float("inf"),
                awaiting_answer=self.state["awaiting_answer"],
                active=frozenset({"timer_running"} if self.state["timer_running"] else ()))
            decision = self.reflex.decide_vector(vectors[spec], state, uid,
                                                     encode_ms=ms.get(spec, 0.0))
            reflex = decision.as_json(full=True)
            reflex["ms"] = reflex["latency_ms"][SOURCE_REFLEX]
            reflex["intent_top"] = _top_intents(self.reflex, vectors[spec])
        teacher = self.teacher.label(audio, RATE).as_row() if self.teacher else None
        agreement = _agreement(reflex, teacher)
        with self._lock:
            self.stats["utterances"] += 1
            if reflex and reflex["route"] in ("act", "ignore"):
                self.stats["decided_by_reflex"] += 1
            if reflex:
                self.stats["reflex_ms"].append(reflex["ms"])
            if agreement in ("agree", "reflex_would_miss", "reflex_would_act_on_ignored",
                             "different_intent"):
                self.stats["compared"] += 1
                self.stats["agree"] += agreement == "agree"
            self.recent[uid] = {"vectors": vectors, "teacher": teacher or {},
                                "duration": float(segment.duration)}
            while len(self.recent) > KEEP_RECENT:
                self.recent.popitem(last=False)
        self.events.publish("utterance", {
            "id": uid, "at": time.strftime("%H:%M:%S"), "duration": round(float(segment.duration), 2),
            "reflex": reflex, "cascade": _teacher_public(teacher), "agreement": agreement,
            "state": dict(self.state)})
        self._push_status()

    def _handle_collect(self, segment, tag: Tag) -> None:
        with self._lock:
            session = self.session
        if session is None:
            return
        audio = segment.audio
        if self._not_wearer(audio):
            self.events.publish("segment", {"dropped": "not the enrolled voice"})
            return
        teacher = self.teacher.label(audio, RATE).as_row() if self.teacher else {}
        vectors = {spec: self.encoders[spec].encode(audio, RATE)
                   for spec in session.encoders if spec in self.encoders}
        with self._lock:
            if self.session is not session:
                return
            clip_id = f"{session.id}-{session.n:04d}"
            session.n += 1
            row = consented_row(clip_id, tag.label, tag.intent, tag.speaker, session=session.id,
                                room=session.room, mic=session.mic, consent=session.consent,
                                duration=float(segment.duration), teacher=teacher)
            row["step"] = tag.step
            session.rows.append(row)
            for spec in session.encoders:
                session.vectors.setdefault(spec, []).append(vectors[spec])
        self.events.publish("segment", _public_row(row))
        self._push_status()

    # -- TRY: controls -------------------------------------------------------------

    def start_try(self) -> None:
        if self.session is not None:
            raise RuntimeError("a collection session is open -- finish or abandon it first")
        self._start_capture("try")

    def stop_listening(self) -> None:
        self.recording = None
        self._stop_capture()

    def set_state(self, awaiting_answer: bool | None = None, timer_running: bool | None = None) -> None:
        if awaiting_answer is not None:
            self.state["awaiting_answer"] = bool(awaiting_answer)
        if timer_running is not None:
            self.state["timer_running"] = bool(timer_running)
        self._push_status()

    def label(self, uid: str, addressed: str, intent: str | None) -> dict:
        """What an utterance in TRY really was -- kept in memory until saved."""
        if addressed not in ADDRESSEES:
            raise ValueError(f"addressed must be one of {ADDRESSEES}")
        if intent and intent not in CORE_SCHEMA.intent_names:
            raise ValueError(f"unknown intent {intent!r}")
        with self._lock:
            item = self.recent.get(uid)
            if item is None:
                raise KeyError("that utterance is no longer in memory")
            row = consented_row(uid, addressed, intent if addressed == ASSISTANT else None, "me",
                                session="try", room="", mic=self._mic_name(), consent="",
                                duration=item["duration"], teacher=item["teacher"],
                                category=f"try_{addressed}")
            # Re-labelling replaces the earlier label.
            self.labelled = [(r, v) for r, v in self.labelled if r["id"] != uid]
            self.labelled.append((row, dict(item["vectors"])))
        self._push_status()
        return {"labelled": len(self.labelled)}

    def save_labelled(self, consent: str, room: str, speaker: str) -> dict:
        if consent not in ("release", "internal"):
            raise ValueError("consent must be release or internal")
        with self._lock:
            if not self.labelled:
                raise ValueError("nothing labelled yet")
            sid = time.strftime("try-%Y%m%d-%H%M%S")
            rows = [{**row, "id": f"{sid}-{row['id']}", "session": sid, "room": room,
                     "consent": consent, "speaker": speaker or "me"} for row, _ in self.labelled]
            # Only encoders every labelled utterance was encoded with.
            specs = set.intersection(*(set(v) for _, v in self.labelled))
            vectors = {spec: [v[spec] for _, v in self.labelled] for spec in sorted(specs)}
            out = self.out_dir / sid
            meta = write_session(out, rows, vectors, {"session": sid, "room": room,
                                                      "mic": self._mic_name(), "consent": consent,
                                                      "speakers": speaker, "kind": "try"})
            self.labelled = []
        self._push_status()
        return {"path": str(out), **meta}

    # -- COLLECT: controls -----------------------------------------------------------

    def start_session(self, speakers: Sequence[str], room: str, consent: str,
                      confirmed: bool, encoders: Sequence[str] | None = None) -> dict:
        if not confirmed:
            raise ValueError("everyone who will be heard has to agree first")
        if consent not in ("release", "internal"):
            raise ValueError("consent must be release or internal")
        speakers = [s.strip() for s in speakers if s and s.strip()]
        if not speakers:
            raise ValueError("name at least one speaker -- pseudonyms like p1, p2")
        if not room.strip():
            raise ValueError("say which room this is")
        if self.listening and self.purpose == "try":
            self._stop_capture()
        specs = [s for s in (encoders or list(self.encoders)) if s in self.encoders]
        if not specs:
            raise RuntimeError("the encoders are still loading")
        with self._lock:
            self.session = Session(id=time.strftime("session-%Y%m%d-%H%M%S"), speakers=speakers,
                                   room=room.strip(), mic=self._mic_name(), consent=consent,
                                   encoders=specs, steps=build_steps(speakers))
        self._push_status()
        return self.session.summary()

    def record_step(self, index: int) -> None:
        with self._lock:
            session = self._need_session()
            step = session.steps[index]
            session.current = index
            self.recording = {"tag": Tag(step.label, step.intent, step.speaker, index),
                              "deadline": time.monotonic() + step.seconds,
                              "seconds": step.seconds, "started": time.monotonic()}
        self._start_capture("collect")
        self._push_status()

    def record_free(self, label: str, intent: str | None, speaker: str) -> None:
        if label not in ADDRESSEES:
            raise ValueError(f"label must be one of {ADDRESSEES}")
        with self._lock:
            self._need_session()
            self.recording = {"tag": Tag(label, intent if label == ASSISTANT else None,
                                         speaker or "", None),
                              "deadline": None, "seconds": None, "started": time.monotonic()}
        self._start_capture("collect")
        self._push_status()

    def stop_recording(self) -> None:
        with self._lock:
            was = self.recording
            self.recording = None
            if was and was["tag"].step is not None and self.session is not None:
                # Move on once a prompt has been done.
                self.session.current = min(was["tag"].step + 1, len(self.session.steps) - 1)
        self._stop_capture()

    def goto_step(self, index: int) -> None:
        with self._lock:
            session = self._need_session()
            session.current = max(0, min(int(index), len(session.steps) - 1))
        self._push_status()

    def discard(self, clip_id: str) -> None:
        with self._lock:
            session = self._need_session()
            for k, row in enumerate(session.rows):
                if row["id"] == clip_id:
                    del session.rows[k]
                    for vecs in session.vectors.values():
                        del vecs[k]
                    break
        self._push_status()

    def finish_session(self) -> dict:
        self.stop_recording()
        with self._lock:
            session = self._need_session()
            if not session.rows:
                raise ValueError("nothing recorded yet -- record a prompt, or abandon the session")
            out = self.out_dir / session.id
            meta = write_session(out, session.rows, session.vectors, {
                "session": session.id, "room": session.room, "mic": session.mic,
                "consent": session.consent, "speakers": ",".join(session.speakers),
                "kind": "collect"})
            self.session = None
        self._push_status()
        return {"path": str(out), **meta}

    def abandon_session(self) -> None:
        self.stop_recording()
        with self._lock:
            self.session = None
        self._push_status()

    def _need_session(self) -> Session:
        if self.session is None:
            raise RuntimeError("no collection session is open")
        return self.session

    def _recording_public(self) -> dict | None:
        rec = self.recording
        if not rec:
            return None
        elapsed = time.monotonic() - rec["started"]
        return {**asdict(rec["tag"]), "seconds": rec["seconds"], "elapsed": round(elapsed, 1),
                "remaining": round(max(0.0, rec["deadline"] - time.monotonic()), 1)
                if rec["deadline"] else None}

    # -- RESULTS ---------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        from juno_core.slu.student import StudentModel

        out = []
        for path in sorted(self.models_dir.glob("*.npz"), key=lambda p: -p.stat().st_mtime):
            try:
                out.append(_model_info(path, StudentModel.load(path)))
            except Exception as exc:
                out.append({"path": str(path), "name": path.stem, "error": str(exc)})
        return out

    def list_sessions(self) -> list[dict]:
        import json

        out = []
        for meta_path in sorted(self.data_dir.glob("**/session.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({"path": str(meta_path.parent.relative_to(self.root)),
                        "name": meta_path.parent.name, **{k: meta.get(k) for k in (
                            "rows", "counts", "consent", "room", "speakers", "encoders",
                            "collected_at", "kind")}})
        return sorted(out, key=lambda s: s.get("collected_at") or "", reverse=True)

    def evaluate(self, model_path: str, sessions: Sequence[str],
                 awaiting_answer: bool = False) -> dict:
        from juno_core.slu import training as TR
        from juno_core.slu.student import StudentModel

        model = StudentModel.load(self._resolve(model_path))
        rows, tables = self._session_data(sessions, model.encoder_spec)
        emb = TR.load_embeddings(tables)
        rows, X, ms = TR.join(rows, emb)
        report = TR.evaluate(model, X, rows, truth_mode="auto",
                             state=ConversationState(awaiting_answer=awaiting_answer,
                                                     since_ai=2.0 if awaiting_answer else float("inf")))
        return {"model": _model_info(self._resolve(model_path), model), "sessions": list(sessions),
                "report": report}

    def _session_data(self, sessions: Sequence[str], spec: str) -> tuple[list[dict], list[Path]]:
        if not sessions:
            raise ValueError("pick at least one session")
        rows, tables = [], []
        for s in sessions:
            d = self._resolve(s)
            table = d / D.table_name(spec)
            if not table.exists():
                raise ValueError(f"{d.name} was not recorded with {spec} -- re-record it, or "
                                 f"pick a model trained on one of its encoders")
            rows += D.read_manifest(d / "manifest.jsonl")
            tables.append(table)
        return rows, tables

    def train(self, name: str, encoder: str, targets: str, sessions: Sequence[str],
              include_base: bool, allow_internal: bool) -> None:
        """Train in the background; progress arrives as 'job' events."""
        with self._lock:
            if self.job and self.job.get("state") == "running":
                raise RuntimeError("a job is already running")
            self.job = {"kind": "train", "state": "running", "log": [], "name": name}
        threading.Thread(target=self._train, daemon=True, name="studio-train", args=(
            name, encoder, targets, list(sessions), include_base, allow_internal)).start()
        self._push_status()

    def _train(self, name, encoder, targets, sessions, include_base, allow_internal) -> None:
        from juno_core.slu import training as TR

        def log(line: str) -> None:
            self.job["log"].append(line)
            self.events.publish("job", dict(self.job))

        try:
            if not name.replace("-", "").replace("_", "").isalnum():
                raise ValueError("name: letters, digits, - and _ only")
            rows, tables = ([], [])
            if sessions:
                rows, tables = self._session_data(sessions, encoder)
            if include_base:
                base_rows = sorted(self.data_dir.glob("*labelled*.jsonl"))
                for path in base_rows:
                    rows += D.read_manifest(path)
                for d in sorted(self.data_dir.glob("emb*")):
                    t = d / D.table_name(encoder)
                    if t.exists():
                        tables.append(t)
                log(f"base data: {len(base_rows)} labelled files, "
                    f"{len(tables) - len(sessions)} vector tables for {encoder}")
            if not tables:
                raise ValueError("no vectors for that encoder -- pick sessions or base data "
                                 "that were embedded with it")
            args = Namespace(targets=targets, gold_weight=0.5, truth="auto", hidden=256,
                             epochs=80, lr=1e-3, weight_decay=1e-4, dropout=0.2,
                             split="0.6,0.2,0.2", group_by="speaker", seed=0,
                             max_false_ignore=0.01, max_wrong_act=0.01,
                             max_false_activation=0.01, margin_z=1.0, tag=name,
                             allow_internal=allow_internal, verbose=False)
            kept = [r for r in rows if D.row_trainable(r, allow_internal)]
            if len(kept) < len(rows):
                log(f"left out {len(rows) - len(kept)} rows whose provenance forbids training "
                    f"(tick 'allow non-distributable' for an experiment-only model)")
            log(f"training on {len(kept)} rows with {targets} targets...")
            emb = TR.load_embeddings(tables)
            model, report = TR.train_student(kept, emb, CORE_SCHEMA, args, log=log)
            out = self.models_dir / f"{name}.npz"
            model.save(out)
            log(f"saved {out.relative_to(self.root)}")
            self.job.update(state="done", model=str(out.relative_to(self.root)),
                            report=TR._headline(report))
        except BaseException as exc:          # SystemExit from a split that cannot be made
            self.job.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        self.events.publish("job", dict(self.job))
        self._push_status()


# -- helpers -------------------------------------------------------------------------

def _model_info(path: Path, model) -> dict:
    holdout = (model.meta.get("metrics") or {}).get("holdout") or {}
    return {
        "path": str(path), "name": path.stem, "tag": model.tag, "encoder": model.encoder_spec,
        "trained_at": model.meta.get("trained_at"), "distributable": model.distributable,
        "targets": (model.meta.get("targets") or {}).get("mode"),
        "thresholds": model.thresholds,
        "holdout": {"rows": holdout.get("rows"), "stt_avoided": holdout.get("stt_avoided"),
                    "false_ignore": (holdout.get("false_ignore") or {}).get("rate"),
                    "false_activation": (holdout.get("false_activation") or {}).get("rate")},
    }


def _top_intents(reflex, vector, k: int = 3) -> list[list]:
    pred = reflex.model.predict(vector)
    top = sorted(pred.intent.items(), key=lambda kv: -kv[1])[:k]
    return [[name, round(p, 3)] for name, p in top]


def _teacher_public(t: dict | None) -> dict | None:
    if not t:
        return None
    return {"transcript": t.get("transcript"), "accepted": t.get("t_accept"),
            "confidence": t.get("t_confidence"), "intent": t.get("t_intent") if t.get("t_accept") else None,
            "slots": t.get("t_slots") if t.get("t_accept") else {}, "stt_ms": t.get("stt_ms")}


def _agreement(reflex: dict | None, teacher: dict | None) -> str:
    """How Reflex's decision compares with what the cascade made of the words."""
    if reflex is None:
        return "no_model"
    if teacher is None:
        return "no_teacher"
    route = reflex["route"]
    accepted = bool(teacher.get("t_accept"))
    if route == "escalate":
        return "escalated"
    if route == "ignore":
        return "agree" if not accepted else "reflex_would_miss"
    if not accepted:
        return "reflex_would_act_on_ignored"
    same_intent = (reflex.get("intent") or {}).get("value") == teacher.get("t_intent")
    reflex_slots = {k: v.get("value") for k, v in (reflex.get("slots") or {}).items()}
    t_slots = teacher.get("t_slots") or {}
    same_slots = all(t_slots.get(k) == v for k, v in reflex_slots.items())
    return "agree" if same_intent and same_slots else "different_intent"


__all__ = ["Studio", "SimulatedCapture", "Broadcaster", "build_steps", "OPEN_REQUEST"]
