"""macOS CoreAudio and AVFoundation probes, through ctypes.

What the microphone *is* turns out to matter to this assistant. AirPods sit a
few centimetres from one mouth; the laptop microphone sits in the middle of a
room and hears everyone in it. The intent engine cannot tell those apart from
a transcript, so it is told here instead.

Deliberately ctypes and not PyObjC: this needs to answer a question at startup
and once per device change, and adding a compiled dependency to do that is a
poor trade. Everything degrades to "unknown" off macOS, or if a symbol is
missing, and callers must treat that as no evidence rather than as a negative.

WHAT IS NOT HERE, AND WHY
-------------------------
There is no own-voice, bone-conduction or jaw-detection signal in this module
because Apple does not expose one. AirPods do contain a speech-detecting
accelerometer separate from the motion accelerometer, and Apple's own DSP uses
it for call and Siri beamforming -- but its output never crosses the framework
boundary. A word-boundary search for own-voice, bone-conduction, wearer and
accelerometer-near-voice across every header of CoreMotion, CoreAudio,
AudioToolbox and AVFAudio returns nothing, and CoreMotion does not contain the
words "voice" or "speech" at all. ``CMHeadphoneMotionManager`` does exist on
macOS 14+ (contrary to a good deal of what is written about it) but it reports
head *orientation* -- attitude, rotation rate, gravity -- and would crash a
process without ``NSMotionUsageDescription`` in an Info.plist, which a bare
``python`` has no way to supply.

Distinguishing the wearer from a bystander is therefore a speaker-verification
problem and is solved in ``audio/voiceprint.py``, not here.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import struct
import sys
from dataclasses import dataclass

IS_MACOS = sys.platform == "darwin"

# AudioHardwareBase.h / AudioHardware.h
_SYSTEM_OBJECT = 1
_SCOPE_GLOBAL = "glob"
_SCOPE_INPUT = "inpt"
_DEFAULT_INPUT = "dIn "
_DEFAULT_OUTPUT = "dOut"
_TRANSPORT_TYPE = "tran"
_DEVICE_NAME = "lnam"
_STREAM_CONFIG = "slay"
_NOMINAL_RATE = "nsrt"
# AudioHardware.h:1346-1347. Present only on the *input* scope: asking for
# them on the global scope returns "no such property" on every device, which
# reads exactly like "this version of macOS does not support it". It is the
# easiest possible way to conclude, wrongly, that the feature is unavailable.
_VAD_ENABLE = "vAd+"
_VAD_STATE = "vAdS"

_TRANSPORT_NAMES = {
    "blue": "bluetooth", "bltn": "builtin", "usb ": "usb",
    "hdmi": "hdmi", "airp": "airplay", "virt": "virtual", "aggr": "aggregate",
}


def _fourcc(code: str) -> int:
    return struct.unpack(">I", code.encode())[0]


class _PropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class InputDevice:
    """What is known about the current microphone."""

    device_id: int
    name: str
    transport: str          # "bluetooth", "builtin", ... or "unknown"
    sample_rate: float
    is_airpods: bool

    @property
    def is_headset(self) -> bool:
        """Worn, so the mouth is a fixed few centimetres from the microphone."""
        return self.transport == "bluetooth"


@dataclass(frozen=True)
class OutputDevice:
    """What is known about where the answer comes out.

    Only one question is really being asked of this: does the sound Juno makes
    reach the microphone Juno listens with. Built-in speakers do; anything
    worn does not.
    """

    device_id: int
    name: str
    transport: str          # "bluetooth", "builtin", ... or "unknown"

    @property
    def is_builtin_speakers(self) -> bool:
        """The laptop's own speakers, firing into the laptop's own mic.

        Voice processing costs about 28 dB here -- it moves these speakers
        onto the voice-call output route, which is quieter by construction.
        Measured: a tone through a loopback device is untouched at +0.0 dB
        while the same tone through the speakers loses 27.8 dB, both recovering
        the moment the helper exits. Headphones have their own output path and
        pay nothing, which is why this only ever shows up on the laptop.
        """
        return self.transport == "builtin"


class CoreAudio:
    """Thin ctypes wrapper. Every method returns None when it cannot answer."""

    def __init__(self) -> None:
        self._lib = None
        self._cf = None
        if not IS_MACOS:
            return
        try:
            self._lib = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
            self._cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        except (OSError, TypeError):
            self._lib = None

    @property
    def available(self) -> bool:
        return self._lib is not None

    # -- raw property access ---------------------------------------------

    def _address(self, selector: str, scope: str) -> _PropertyAddress:
        return _PropertyAddress(_fourcc(selector), _fourcc(scope), 0)

    def _get_u32(self, obj: int, selector: str, scope: str) -> int | None:
        if not self.available:
            return None
        addr = self._address(selector, scope)
        size = ctypes.c_uint32(4)
        value = ctypes.c_uint32(0)
        status = self._lib.AudioObjectGetPropertyData(
            obj, ctypes.byref(addr), 0, None, ctypes.byref(size), ctypes.byref(value)
        )
        return value.value if status == 0 else None

    def _get_f64(self, obj: int, selector: str, scope: str) -> float | None:
        if not self.available:
            return None
        addr = self._address(selector, scope)
        size = ctypes.c_uint32(8)
        value = ctypes.c_double(0)
        status = self._lib.AudioObjectGetPropertyData(
            obj, ctypes.byref(addr), 0, None, ctypes.byref(size), ctypes.byref(value)
        )
        return value.value if status == 0 else None

    def _get_string(self, obj: int, selector: str, scope: str) -> str | None:
        if not self.available or self._cf is None:
            return None
        addr = self._address(selector, scope)
        size = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
        ref = ctypes.c_void_p()
        status = self._lib.AudioObjectGetPropertyData(
            obj, ctypes.byref(addr), 0, None, ctypes.byref(size), ctypes.byref(ref)
        )
        if status != 0 or not ref:
            return None
        buffer = ctypes.create_string_buffer(512)
        ok = self._cf.CFStringGetCString(ref, buffer, 512, 0x08000100)  # UTF-8
        self._cf.CFRelease(ref)
        return buffer.value.decode("utf-8", "replace") if ok else None

    def _has(self, obj: int, selector: str, scope: str) -> bool:
        if not self.available:
            return False
        addr = self._address(selector, scope)
        return bool(self._lib.AudioObjectHasProperty(obj, ctypes.byref(addr)))

    def _set_u32(self, obj: int, selector: str, scope: str, value: int) -> bool:
        if not self.available:
            return False
        addr = self._address(selector, scope)
        payload = ctypes.c_uint32(value)
        status = self._lib.AudioObjectSetPropertyData(
            obj, ctypes.byref(addr), 0, None, 4, ctypes.byref(payload)
        )
        return status == 0

    # -- questions worth asking -------------------------------------------

    def default_input_device(self) -> int | None:
        return self._get_u32(_SYSTEM_OBJECT, _DEFAULT_INPUT, _SCOPE_GLOBAL)

    def default_output_device(self) -> int | None:
        return self._get_u32(_SYSTEM_OBJECT, _DEFAULT_OUTPUT, _SCOPE_GLOBAL)

    def describe_output(self, device_id: int | None = None) -> OutputDevice | None:
        """Identify where sound is going, or None if it cannot be read."""
        if device_id is None:
            device_id = self.default_output_device()
        if device_id is None:
            return None
        raw = self._get_u32(device_id, _TRANSPORT_TYPE, _SCOPE_GLOBAL)
        transport = "unknown"
        if raw is not None:
            transport = _TRANSPORT_NAMES.get(
                struct.pack(">I", raw).decode("ascii", "replace"), "unknown"
            )
        return OutputDevice(
            device_id=device_id,
            name=self._get_string(device_id, _DEVICE_NAME, _SCOPE_GLOBAL) or "",
            transport=transport,
        )

    def describe_input(self, device_id: int | None = None) -> InputDevice | None:
        """Identify the current microphone, or None if it cannot be read."""
        if device_id is None:
            device_id = self.default_input_device()
        if device_id is None:
            return None
        name = self._get_string(device_id, _DEVICE_NAME, _SCOPE_GLOBAL) or ""
        raw = self._get_u32(device_id, _TRANSPORT_TYPE, _SCOPE_GLOBAL)
        transport = "unknown"
        if raw is not None:
            transport = _TRANSPORT_NAMES.get(
                struct.pack(">I", raw).decode("ascii", "replace"), "unknown"
            )
        rate = self._get_f64(device_id, _NOMINAL_RATE, _SCOPE_GLOBAL) or 0.0
        lowered = name.lower()
        return InputDevice(
            device_id=device_id,
            name=name,
            transport=transport,
            sample_rate=rate,
            # Transport alone only proves Bluetooth -- any headset matches.
            is_airpods=transport == "bluetooth" and "airpod" in lowered,
        )

    # -- Apple's own near-end voice activity detection ---------------------
    #
    # Reachable, free, and NOT a solution to "is this the wearer": the header
    # defines the state as voice / no voice with no wearer semantics. It is an
    # any-talker detector. Kept because it is a cheap corroborating signal and
    # because it comes with echo cancellation, not because it discriminates.

    def voice_activity_supported(self, device_id: int | None = None) -> bool:
        if device_id is None:
            device_id = self.default_input_device()
        if device_id is None:
            return False
        return self._has(device_id, _VAD_ENABLE, _SCOPE_INPUT)

    def enable_voice_activity_detection(self, device_id: int | None = None) -> bool:
        if device_id is None:
            device_id = self.default_input_device()
        if device_id is None:
            return False
        return self._set_u32(device_id, _VAD_ENABLE, _SCOPE_INPUT, 1)

    def voice_activity_state(self, device_id: int | None = None) -> bool | None:
        """1 = voice present. Requires the input stream to be running."""
        if device_id is None:
            device_id = self.default_input_device()
        if device_id is None:
            return None
        value = self._get_u32(device_id, _VAD_STATE, _SCOPE_INPUT)
        return None if value is None else bool(value)


# -- microphone modes ------------------------------------------------------
#
# Voice Isolation cannot be turned on by an application. Both
# +[AVCaptureDevice preferredMicrophoneMode] and +[AVCaptureDevice
# activeMicrophoneMode] are declared @property(class, readonly) and the
# corresponding setters do not exist in the runtime -- verified by asking
# class_getClassMethod for them and getting NULL. The mode is the user's
# choice, made in Control Center.
#
# There is a second reason this cannot simply be switched on: mic modes are
# implemented inside the VoiceProcessingIO audio unit, and this project
# captures through sounddevice, which uses HALOutput. HALOutput has no voice
# processing path -- it rejects the property outright -- so even a mode the
# user selects by hand does not reach this audio.

MICROPHONE_MODES = {0: "standard", 1: "wide_spectrum", 2: "voice_isolation"}


def microphone_mode() -> str | None:
    """The mode the user has selected, or None if it cannot be read."""
    if not IS_MACOS:
        return None
    try:
        objc = ctypes.CDLL(ctypes.util.find_library("objc"))
        ctypes.CDLL(ctypes.util.find_library("AVFoundation"))
    except (OSError, TypeError):
        return None
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.class_getClassMethod.restype = ctypes.c_void_p
    objc.class_getClassMethod.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cls = objc.objc_getClass(b"AVCaptureDevice")
    if not cls:
        return None
    selector = objc.sel_registerName(b"activeMicrophoneMode")
    if not objc.class_getClassMethod(cls, selector):
        return None
    objc.objc_msgSend.restype = ctypes.c_long
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return MICROPHONE_MODES.get(objc.objc_msgSend(ctypes.c_void_p(cls), selector))
