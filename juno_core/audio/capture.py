"""Microphone capture.

Produces a continuous stream of fixed-size float32 frames at the configured
sample rate. The rest of the pipeline knows nothing about where the audio came
from, so swapping the MacBook microphone for a USB throat-mic interface is a
config change (``audio.input_device``) rather than a code change.

The stream can be muted, which is how half-duplex operation is implemented:
while the assistant is speaking we drop incoming frames rather than transcribe
our own voice back into the pipeline.
"""

from __future__ import annotations

import json
import os
import queue
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from juno_core.events import Stage
from juno_core.observability import Observer


@dataclass
class AudioFrame:
    """One fixed-size block of mono audio."""

    samples: np.ndarray  # float32, shape (frame_samples,), nominally [-1, 1]
    # time.monotonic() sampled inside the audio callback. Taken here rather
    # than when a consumer gets around to reading the frame, so queue delay
    # does not leak into the latency numbers. Comparable with every other
    # timestamp in the system.
    timestamp: float
    # PortAudio's own ADC clock. A different time base to the above -- useful
    # for detecting dropped audio, never mix the two in a subtraction.
    adc_time: float
    index: int

    @property
    def rms(self) -> float:
        if self.samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples), dtype=np.float64)))

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.samples))) if self.samples.size else 0.0


class AudioCaptureError(RuntimeError):
    pass


def resolve_device(name_fragment: str | None):
    """Map a device-name substring to a sounddevice index (None = default)."""
    import sounddevice as sd

    if not name_fragment:
        return None
    needle = name_fragment.lower()
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and needle in dev["name"].lower():
            return index
    available = [
        d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0
    ]
    raise AudioCaptureError(
        f"no input device matching {name_fragment!r}; available: {available}"
    )


def list_input_devices() -> list[tuple[int, str, int]]:
    import sounddevice as sd

    return [
        (i, d["name"], int(d["default_samplerate"]))
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


class AudioCapture:
    """Continuous microphone capture into a bounded queue."""

    def __init__(self, config, observer: Observer | None = None) -> None:
        self.sample_rate = int(config.sample_rate)
        self.channels = int(config.channels)
        self.frame_samples = int(self.sample_rate * config.frame_ms / 1000)
        self.device = config.get("input_device")
        self._observer = observer
        # Roughly two seconds of slack. If a consumer stalls longer than that we
        # would rather drop audio than grow memory without bound.
        maxsize = max(8, int(2.0 * self.sample_rate / self.frame_samples))
        self._queue: queue.Queue[AudioFrame | None] = queue.Queue(maxsize=maxsize)
        self._stream = None
        self._muted = threading.Event()
        self._running = threading.Event()
        self._index = 0
        self._dropped = 0

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        import sounddevice as sd

        if self._stream is not None:
            return
        device = resolve_device(self.device)
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.frame_samples,
            device=device,
            callback=self._callback,
        )
        self._stream.start()
        self._running.set()
        if self._observer:
            name = sd.query_devices(device)["name"] if device is not None else "default"
            self._observer.emit(
                Stage.AUDIO,
                "capture_started",
                device=name,
                sample_rate=self.sample_rate,
                frame_ms=round(1000 * self.frame_samples / self.sample_rate),
            )

    def stop(self) -> None:
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Unblock any consumer parked on frames().
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._observer:
            self._observer.emit(
                Stage.AUDIO, "capture_stopped", frames_dropped=self._dropped
            )

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- muting -----------------------------------------------------------

    def mute(self) -> None:
        self._muted.set()

    def unmute(self) -> None:
        # Drain whatever arrived during the mute window so the VAD does not see
        # a discontinuity stitched onto fresh audio.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._muted.clear()

    @property
    def muted(self) -> bool:
        return self._muted.is_set()

    # -- stream -----------------------------------------------------------

    def _callback(self, indata, frames, time_info, status) -> None:
        if status and self._observer:
            self._observer.emit(Stage.AUDIO, "stream_status", detail=str(status))
        if self._muted.is_set():
            return
        # sounddevice hands us a view into its own buffer; copy before queueing.
        samples = np.array(indata[:, 0], dtype=np.float32, copy=True)
        frame = AudioFrame(
            samples=samples,
            timestamp=time.monotonic(),
            adc_time=float(getattr(time_info, "inputBufferAdcTime", 0.0) or 0.0),
            index=self._index,
        )
        self._index += 1
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self._dropped += 1

    def frames(self) -> Iterator[AudioFrame]:
        """Blocking iterator over captured frames. Ends when stop() is called."""
        while self._running.is_set():
            try:
                frame = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if frame is None:
                break
            yield frame

    @property
    def frames_dropped(self) -> int:
        return self._dropped

class VoiceProcessingCapture:
    """Capture through Apple's voice-processing path, via a helper process.

    Same surface as ``AudioCapture`` -- frames(), mute(), unmute(), stop() --
    so the caller does not need to know which one it has. What differs is what the
    samples have been through before arriving: echo cancellation, Apple's
    near-end talker processing, and whichever mic mode the user has chosen.

    It is a separate process rather than a binding because the API that
    enables all this is AVAudioEngine's, and driving that from Python means
    PyObjC plus a real-time render callback crossing the language boundary on
    every buffer. A pipe carrying float32 is a much smaller thing to get
    wrong. See audio/vpio/vpio-capture.swift for why the audio unit cannot
    simply be opened directly at 16 kHz.

    Cost worth knowing about: enabling voice processing on a Bluetooth device
    measured 1.38 s here, and about 1.9 s to the first sample. That is paid
    once, at startup, next to a warmup that already costs eleven seconds.
    """

    HELPER = Path(__file__).resolve().parent / "vpio" / "vpio-capture"

    def __init__(self, config, observer: Observer | None = None) -> None:
        self.sample_rate = int(config.sample_rate)
        self.channels = 1
        self.frame_samples = int(self.sample_rate * config.frame_ms / 1000)
        self.device = config.get("input_device")
        self.agc = bool(config.get("vpio_agc", False))
        self.ducking = str(config.get("vpio_ducking", "min") or "min")
        self._observer = observer
        self._process: subprocess.Popen | None = None
        self._muted = threading.Event()
        self._running = threading.Event()
        self._index = 0
        self._dropped = 0
        self._info: dict = {}
        self._stderr_thread: threading.Thread | None = None

    @classmethod
    def available(cls) -> bool:
        return sys.platform == "darwin" and cls.HELPER.exists() and os.access(cls.HELPER, os.X_OK)

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._process is not None:
            return
        if not self.available():
            raise AudioCaptureError(
                f"the voice-processing helper is missing or not executable at "
                f"{self.HELPER}. Build it with scripts/build_vpio.sh, or set "
                f"audio.backend: sounddevice."
            )
        command = [str(self.HELPER), "--rate", str(self.sample_rate)]
        if self.device:
            command += ["--device", str(self.device)]
        if self.agc:
            command.append("--agc")
        command += ["--ducking", self.ducking]
        # bufsize=0: stdout is read through its file descriptor with select(),
        # and a buffered reader would hide arrived bytes from it.
        self._process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        self._running.set()

        # The helper's first stderr line is a JSON description of what it
        # actually opened. Everything after it is a diagnostic worth logging
        # but not worth parsing.
        first = self._process.stderr.readline().decode("utf-8", "replace").strip()
        try:
            self._info = json.loads(first)
        except (ValueError, TypeError):
            self._info = {}
            if first:
                self._fail(first)
        if not self._info and self._process.poll() is not None:
            raise AudioCaptureError(f"voice-processing capture failed to start: {first}")

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name="vpio-stderr"
        )
        self._stderr_thread.start()

        if self._observer:
            self._observer.emit(
                Stage.AUDIO, "capture_started",
                device=self._info.get("device", "?"),
                sample_rate=self.sample_rate,
                frame_ms=round(1000 * self.frame_samples / self.sample_rate),
                unit=self._info.get("unit"),
                source_rate=self._info.get("source_rate"),
                echo_cancellation=self._info.get("echo_cancellation"),
                agc=self._info.get("agc"),
                ducking=self._info.get("ducking"),
            )

    def _fail(self, message: str) -> None:
        if self._observer:
            self._observer.emit(Stage.AUDIO, "capture_error", detail=message[:200])

    def _drain_stderr(self) -> None:
        stream = self._process.stderr if self._process else None
        if stream is None:
            return
        for line in iter(stream.readline, b""):
            text = line.decode("utf-8", "replace").strip()
            if text:
                self._fail(text)

    def stop(self) -> None:
        self._running.clear()
        process, self._process = self._process, None
        if process is None:
            return
        # Closing stdout is the helper's ordinary stop signal; terminate is
        # the backstop for one that is wedged before its first write.
        try:
            process.terminate()
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
        except OSError:
            pass
        for stream in (process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        if self._observer:
            self._observer.emit(
                Stage.AUDIO, "capture_stopped", frames_dropped=self._dropped
            )

    def __enter__(self) -> "VoiceProcessingCapture":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- muting -----------------------------------------------------------

    def mute(self) -> None:
        self._muted.set()

    def unmute(self) -> None:
        self._muted.clear()

    @property
    def muted(self) -> bool:
        return self._muted.is_set()

    @property
    def info(self) -> dict:
        """What the helper reported opening. Empty before start()."""
        return dict(self._info)

    # -- stream -----------------------------------------------------------

    def _read_exact(self, fd: int, count: int) -> bytes | None:
        """Read exactly ``count`` bytes, or None when the stream is finished.

        Two things this has to get right, both of which were bugs.

        A pipe read returns what happens to have arrived, not what was asked
        for, so a short read is the normal case rather than the end of the
        stream. Treating the two as the same thing ended capture a tenth of a
        second in and looked like a microphone that produced three frames and
        stopped.

        And it must not block forever. A helper that stops producing -- a
        Bluetooth device renegotiating its profile is enough -- would
        otherwise wedge the audio loop inside a read with no way to notice
        that ``stop()`` had been called. So the wait is bounded, and each time
        it expires the helper is checked for still being alive. The raw file
        descriptor is used rather than the buffered reader because ``select``
        reports on the kernel buffer and would not know about bytes already
        sitting in Python's.
        """
        chunks: list[bytes] = []
        remaining = count
        while remaining > 0:
            if not self._running.is_set():
                return None
            try:
                ready, _, _ = select.select([fd], [], [], 0.25)
            except (OSError, ValueError):
                return None
            if not ready:
                if self._process is None or self._process.poll() is not None:
                    return None      # the helper exited while we were waiting
                continue
            try:
                chunk = os.read(fd, remaining)
            except (OSError, ValueError):
                return None
            if not chunk:
                return None          # genuine EOF: the helper has closed stdout
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def frames(self) -> Iterator[AudioFrame]:
        """Blocking iterator over captured frames. Ends when stop() is called."""
        block = self.frame_samples * 4   # float32
        stream = self._process.stdout if self._process else None
        if stream is None:
            return
        fd = stream.fileno()
        while self._running.is_set():
            data = self._read_exact(fd, block)
            if data is None:
                if self._running.is_set() and self._observer:
                    # Falling silent is how this fails, so say so: an audio
                    # loop that simply ends is indistinguishable from a quiet
                    # room until somebody reads the log.
                    self._observer.emit(
                        Stage.AUDIO, "capture_ended",
                        reason="helper stopped producing audio",
                        frames=self._index,
                    )
                break
            if self._muted.is_set():
                # Dropped here rather than in the helper so that voice
                # processing keeps running: its echo canceller adapts to the
                # room continuously, and stopping the stream to mute would
                # throw that state away every time the assistant speaks.
                self._dropped += 1
                continue
            frame = AudioFrame(
                samples=np.frombuffer(data, dtype="<f4").copy(),
                timestamp=time.monotonic(),
                adc_time=0.0,
                index=self._index,
            )
            self._index += 1
            yield frame

    @property
    def frames_dropped(self) -> int:
        return self._dropped


def _builtin_speakers() -> str | None:
    """The output device's name when it is the built-in speakers, else None.

    None also covers "cannot tell" -- a machine that will not answer the
    question keeps the behaviour it had, rather than silently losing echo
    cancellation over a reading that never arrived.
    """
    try:
        from juno_core.audio.coreaudio import CoreAudio

        output = CoreAudio().describe_output()
    except Exception:
        return None
    if output is not None and output.is_builtin_speakers:
        return output.name or "built-in speakers"
    return None


def build_capture(config, observer: Observer | None = None):
    """Pick a capture backend from config, degrading to sounddevice.

    ``vpio`` is preferred where it is available because it is the only way to
    reach echo cancellation and Apple's near-end processing -- but it needs a
    compiled helper, so falling back has to be graceful rather than fatal.
    """
    backend = str(config.get("backend", "sounddevice")).lower()
    if backend in ("vpio", "voiceprocessing", "voice_processing"):
        if VoiceProcessingCapture.available():
            speakers = _builtin_speakers()
            if speakers is not None and not config.get("vpio_on_builtin_speakers", False):
                # Voice processing costs about 28 dB on the laptop's own
                # speakers: enabling it moves them onto the voice-call output
                # route. Nothing here can turn that off -- it is the route,
                # not the samples -- so the choice is echo cancellation or
                # being audible, and being audible wins. The microphone is
                # already closed while Juno speaks, so cancelling its own
                # voice was the smaller half of what this was buying.
                #
                # Anything worn has its own output path and pays nothing, so
                # this only fires on the built-in speakers.
                if observer:
                    from juno_core.events import Stage as _Stage

                    observer.emit(
                        _Stage.AUDIO, "backend_fallback", requested="vpio",
                        using="sounddevice", output=speakers,
                        reason="voice processing costs ~28 dB on built-in speakers",
                    )
                return AudioCapture(config, observer)
            return VoiceProcessingCapture(config, observer)
        if observer:
            from juno_core.events import Stage as _Stage

            observer.emit(
                _Stage.AUDIO, "backend_fallback", requested="vpio",
                using="sounddevice",
                reason=f"helper not built at {VoiceProcessingCapture.HELPER}",
            )
    elif backend not in ("sounddevice", "portaudio"):
        raise ValueError(f"unknown audio backend {backend!r}")
    return AudioCapture(config, observer)
