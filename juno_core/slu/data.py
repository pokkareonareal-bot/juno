"""Clips to learn from: synthetic speech, real corpora, and where each came from.

Three sources, each with a provenance tag the training step enforces:

  synthetic speech   The scripted corpus (corpus.py) voiced by text-to-speech,
                     one voice per pseudo-speaker, then augmented (noise,
                     reverberation, level, far-field filtering) so the student
                     does not learn the sound of a clean TTS file. With macOS
                     ``say`` the source is ``synthetic_say``: Apple's licence
                     covers personal, non-commercial use of the system voices,
                     so a model trained on it is marked NON-DISTRIBUTABLE --
                     fine for your own experiments, not for shipping weights
                     with an open-source release. Register an openly licensed
                     TTS (e.g. Kokoro, Apache-2.0) as a ``custom`` source for
                     that.
  AMI meetings       Real people, real rooms, far-field microphones, talking to
                     each other (CC BY 4.0). Every segment is human_directed.
                     The far-field array channel, not the headsets, because
                     that is what a room microphone hears.
  Speech Commands    Real voices saying single command words (CC BY 4.0):
                     "stop", "yes", "no" -- the only real-human
                     assistant-directed audio among the public corpora.

WHY REAL AUDIO ON BOTH SIDES MATTERS
------------------------------------
If every assistant-directed clip were synthetic and every human-directed one
real, the student would learn "synthetic means for me". The corpus therefore
voices human-directed and background lines with the same TTS voices, and the
evaluation reports accuracy per source so the confound is visible if it is
there.

Manifests are JSON Lines, one clip per line: the gold labels, provenance
(source, license, consent, speaker, session, room, mic), and how to get the
audio. Clips are 16 kHz mono.

NOTHING HERE WRITES AUDIO
-------------------------
juno_core writes no audio to disk, and this file is no exception (the test
suite checks). The only audio files are the ones the TTS engine itself
writes and the public corpora exactly as published. Every DERIVED clip is a
recipe in its manifest row, rebuilt in memory each time it is read:

  an augmented copy     the clean clip's path plus ``augment: {seed, babble}``
  an AMI utterance      the meeting recording's path plus ``start``/``end``
  a non-speech clip     ``generate: {kind: noise, seed}``

So a dataset is a few original files and a list of instructions, which is
also why it is cheap to regenerate with another seed.
"""

from __future__ import annotations

import concurrent.futures as futures
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import wave
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from juno_core.intelligence.gate_training import OPEN_LICENSES, read_wav

RATE = 16000

# Sources System One's training knows, beyond gate_training's registry.
SLU_SOURCES = {
    "synthetic_say": {
        "name": "Scripted lines voiced by macOS `say`",
        "license": "Apple-SLA-personal",
        "url": "",
        "attribution": "",
        "distributable": False,
    },
    "noise": {
        "name": "Generated noise (no speech)",
        "license": "CC0-1.0",
        "url": "",
        "attribution": "",
        "distributable": True,
    },
    "ami": {
        "name": "AMI Meeting Corpus (far-field array channel)",
        "license": "CC-BY-4.0",
        "url": "https://groups.inf.ed.ac.uk/ami/corpus/",
        "attribution": "AMI Meeting Corpus, AMI Consortium, CC BY 4.0",
        "distributable": True,
    },
    "speech_commands": {
        "name": "Google Speech Commands v0.02 (test set)",
        "license": "CC-BY-4.0",
        "url": "https://arxiv.org/abs/1804.03209",
        "attribution": "Speech Commands v0.02, P. Warden (Google), CC BY 4.0",
        "distributable": True,
    },
    "consented": {
        "name": "Consented Juno recordings",
        "license": "consent",
        "url": "",
        "attribution": "",
        "distributable": None,     # decided by the consent scope
    },
    "custom": {
        "name": "Other openly licensed data (licence column required)",
        "license": None,
        "url": "",
        "attribution": "",
        "distributable": None,     # decided by the licence
    },
}

# Novelty and effect voices sound like nobody and teach nothing.
NOVELTY_VOICES = {
    "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Wobble", "Good News",
    "Jester", "Organ", "Superstar", "Trinoids", "Whisper", "Zarvox", "Voice 1",
}

AMI_MEETINGS = ("ES2002a", "ES2003a", "IS1000a", "TS3003a", "ES2004a", "IS1001a")
AMI_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Array1-01.wav"
SPEECH_COMMANDS_URL = ("http://download.tensorflow.org/data/"
                       "speech_commands_test_set_v0.02.tar.gz")
SPEECH_COMMAND_INTENTS = {"stop": "stop", "yes": "confirm.yes", "no": "confirm.no"}


def row_distributable(row: dict) -> bool:
    source = row.get("source", "")
    if source == "consented":
        return row.get("consent") == "release"
    info = SLU_SOURCES.get(source)
    if info is not None and info["distributable"] is not None:
        return bool(info["distributable"])
    return row.get("license") in OPEN_LICENSES


def row_trainable(row: dict, allow_internal: bool = False) -> bool:
    """Open licence or released consent; anything else only with allow_internal."""
    if row.get("source") == "shadow_log":
        return False
    if row.get("source") == "consented" and row.get("consent") not in ("release", "internal"):
        return False
    return row_distributable(row) or allow_internal


def provenance(rows: Sequence[dict]) -> dict:
    from collections import Counter

    counts = Counter((r.get("source", ""), r.get("license", ""), r.get("consent", "")) for r in rows)
    sources = []
    for (source, license_, consent), n in sorted(counts.items()):
        info = SLU_SOURCES.get(source, {})
        sources.append({"source": source, "name": info.get("name", source), "license": license_,
                        "consent": consent or None, "rows": n, "url": info.get("url", ""),
                        "attribution": info.get("attribution", "")})
    return {"sources": sources, "distributable": all(row_distributable(r) for r in rows)}


# -- manifests ---------------------------------------------------------------

def write_manifest(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            n += 1
    return n


def read_manifest(paths: str | Path | Sequence[str | Path]) -> list[dict]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows = []
    for path in paths:
        base = Path(path).resolve().parent
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row.setdefault("_base", str(base))
                    rows.append(row)
    return rows


def _resolve(path: str, row: dict) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else Path(row.get("_base", ".")) / p


def clip_path(row: dict) -> Path:
    return _resolve(row["path"], row)


def absolute_paths(row: dict) -> dict:
    """The row with every path in it made absolute (for files written elsewhere)."""
    out = dict(row)
    if out.get("path"):
        out["path"] = str(clip_path(row))
    aug = out.get("augment")
    if isinstance(aug, dict) and aug.get("babble"):
        out["augment"] = {**aug, "babble": [str(_resolve(b, row)) for b in aug["babble"]]}
    out.pop("_base", None)
    return out


def recipe_seed(clip_id: str) -> int:
    return int(hashlib.sha1(clip_id.encode()).hexdigest()[:8], 16)


def load_clip(row: dict) -> np.ndarray:
    """The clip's audio, rebuilt from its recipe if it is a derived one."""
    generate = row.get("generate")
    if generate:
        if generate.get("kind") != "noise":
            raise ValueError(f"{row.get('id')}: unknown generator {generate.get('kind')!r}")
        return noise_clip(np.random.default_rng(int(generate["seed"])))[0]
    audio, rate = read_wav(clip_path(row), row.get("start"), row.get("end"))
    if rate != RATE:
        raise ValueError(f"{row['path']}: {rate} Hz; clips must be 16 kHz mono")
    aug = row.get("augment")
    if isinstance(aug, dict) and "seed" in aug:
        babble = None
        if aug.get("babble"):
            babble = np.concatenate([read_wav(_resolve(b, row))[0] for b in aug["babble"]])
        audio, _ = augment(audio, np.random.default_rng(int(aug["seed"])), babble,
                           strength=float(aug.get("strength", 1.0)))
    return audio


# -- text to speech ----------------------------------------------------------

def say_voices(language_prefix: str = "en_") -> list[str]:
    """English macOS voices, novelty voices removed, one entry per voice."""
    if shutil.which("say") is None:
        return []
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    voices, seen = [], set()
    for line in out.splitlines():
        head, _, _ = line.partition("#")
        parts = head.rsplit(None, 1)
        if len(parts) != 2 or not parts[1].startswith(language_prefix):
            continue
        name = parts[0].strip()
        base = name.split(" (")[0]
        if base in NOVELTY_VOICES or name in seen:
            continue
        seen.add(name)
        voices.append(name)
    return voices


def say_to_wav(text: str, voice: str, out: Path, rate_wpm: int | None = None) -> None:
    cmd = ["say", "-v", voice, "-o", str(out), "--file-format=WAVE",
           f"--data-format=LEI16@{RATE}"]
    if rate_wpm:
        cmd += ["-r", str(int(rate_wpm))]
    subprocess.run(cmd + [text], check=True, capture_output=True, timeout=30)


def _readable(path: Path) -> bool:
    """A complete WAV with some audio in it (a killed `say` leaves a stub)."""
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() > 800
    except (OSError, EOFError, wave.Error):
        return False


def speaker_id(voice: str) -> str:
    return "say-" + hashlib.sha1(voice.encode()).hexdigest()[:8]


# -- augmentation ------------------------------------------------------------

def _colored_noise(n: int, rng: np.random.Generator, color: str) -> np.ndarray:
    white = rng.standard_normal(n)
    if color == "white":
        return white
    spectrum = np.fft.rfft(white)
    f = np.maximum(np.fft.rfftfreq(n), 1.0 / n)
    spectrum /= np.sqrt(f) if color == "pink" else f
    out = np.fft.irfft(spectrum, n)
    return out / (np.std(out) + 1e-9)


def _fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.size + b.size - 1
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b, size), size)[:n]


def room_impulse(rng: np.random.Generator, rt60: float) -> np.ndarray:
    """A synthetic room: a direct path then exponentially decaying noise."""
    n = int(RATE * min(rt60 * 1.2, 1.0))
    t = np.arange(n) / RATE
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / max(rt60, 1e-3))
    tail[: int(0.003 * RATE)] = 0.0
    ir = 0.4 * tail / (np.sqrt(np.sum(tail ** 2)) + 1e-9)
    ir[0] = 1.0
    return ir


def augment(audio: np.ndarray, rng: np.random.Generator, babble: np.ndarray | None = None,
            strength: float = 1.0) -> tuple[np.ndarray, dict]:
    """A plausible microphone take of a clean clip, and what was done to it."""
    x = np.asarray(audio, np.float64)
    info: dict = {}
    # Silence either side, as the VAD's pre-roll and hangover would leave it.
    lead, tail = int(rng.uniform(0.1, 0.5) * RATE), int(rng.uniform(0.2, 0.6) * RATE)
    x = np.concatenate([np.zeros(lead), x, np.zeros(tail)])
    if rng.random() < 0.6 * strength:
        rt60 = float(rng.uniform(0.15, 0.8))
        x = _fft_convolve(x, room_impulse(rng, rt60))[: x.size]
        info["rt60"] = round(rt60, 2)
    if rng.random() < 0.3 * strength:
        # Far field or a cheap microphone: lose the top of the spectrum.
        cutoff = float(rng.uniform(2500, 6000))
        spec = np.fft.rfft(x)
        spec[np.fft.rfftfreq(x.size, 1.0 / RATE) > cutoff] *= 0.1
        x = np.fft.irfft(spec, x.size)
        info["lowpass_hz"] = int(cutoff)
    speech_rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    if rng.random() < 0.85 * strength:
        snr = float(rng.uniform(5, 30))
        color = str(rng.choice(["white", "pink", "brown"]))
        noise = _colored_noise(x.size, rng, color)
        if babble is not None and babble.size and rng.random() < 0.3:
            reps = int(np.ceil(x.size / babble.size))
            start = int(rng.integers(0, max(1, babble.size)))
            noise = np.tile(babble, reps + 1)[start:start + x.size]
            noise = noise / (np.std(noise) + 1e-9)
            color = "babble"
        x = x + noise * speech_rms / (10 ** (snr / 20.0))
        info.update(snr_db=round(snr, 1), noise=color)
    gain_db = float(rng.uniform(-24, -3))
    peak = np.max(np.abs(x)) + 1e-9
    x = x / peak * (10 ** (gain_db / 20.0))
    info["level_db"] = round(gain_db, 1)
    return x.astype(np.float32), info


def noise_clip(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Seconds of something that is not speech: room tone, hum, a chord."""
    seconds = float(rng.uniform(0.6, 4.0))
    n = int(seconds * RATE)
    kind = str(rng.choice(["pink", "brown", "hum", "chord", "clatter"]))
    t = np.arange(n) / RATE
    if kind in ("pink", "brown"):
        x = _colored_noise(n, rng, kind)
    elif kind == "hum":
        x = np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 150 * t) + 0.1 * rng.standard_normal(n)
    elif kind == "chord":
        notes = rng.choice([220.0, 261.6, 329.6, 392.0, 440.0, 523.3], size=3, replace=False)
        x = sum(np.sin(2 * np.pi * f * t) for f in notes) * np.exp(-t / max(seconds, 0.5))
    else:
        x = np.zeros(n)
        for _ in range(int(rng.integers(3, 12))):
            at = int(rng.integers(0, max(1, n - 800)))
            x[at:at + 800] += rng.standard_normal(800) * np.exp(-np.arange(800) / 120.0)
    x = x / (np.max(np.abs(x)) + 1e-9) * (10 ** (rng.uniform(-30, -6) / 20.0))
    return x.astype(np.float32), {"noise_kind": kind}


# -- the synthetic set ---------------------------------------------------------

def synthesise(lines, out_dir: str | Path, *, voices: Sequence[str] | None = None,
               augment_copies: int = 1, seed: int = 0, workers: int = 6,
               noise_clips: int = 0, keep_clean: bool = True, log=print) -> list[dict]:
    """Voice each line with a TTS voice, write clean (+ augmented) clips, return rows.

    Voices are pseudo-speakers: each line gets one voice, chosen round-robin
    over a seeded shuffle, so every voice says a mix of every category and a
    split by speaker holds out whole voices.
    """
    out_dir = Path(out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    voices = list(voices or say_voices())
    if not voices:
        raise RuntimeError("no TTS voices found (this uses macOS `say`)")
    rng = random.Random(seed)
    jobs = []
    for i, line in enumerate(lines):
        voice = voices[i % len(voices)] if i < len(voices) else rng.choice(voices)
        wpm = rng.choice([None, 150, 170, 190, 210, 230])
        clip_id = f"syn{seed:02d}-{i:05d}"
        jobs.append((clip_id, line, voice, wpm))

    def render(job):
        clip_id, line, voice, wpm = job
        path = out_dir / "audio" / f"{clip_id}.wav"
        if _readable(path):
            return job, path
        # The speech service occasionally wedges on one request; a retry
        # almost always goes through, and one lost line is not worth a run.
        for _ in range(2):
            path.unlink(missing_ok=True)
            try:
                say_to_wav(line.text, voice, path, wpm)
            except (subprocess.SubprocessError, OSError):
                continue
            if _readable(path):
                return job, path
        path.unlink(missing_ok=True)
        return job, None

    rows: list[dict] = []
    clean: list[tuple] = []
    done = failed = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for (clip_id, line, voice, wpm), path in pool.map(render, jobs):
            done += 1
            if done % 250 == 0:
                log(f"  voiced {done}/{len(jobs)}")
            if path is None:
                failed += 1
                continue
            clean.append((clip_id, line, voice, wpm, path))
    if failed:
        log(f"  {failed} lines could not be voiced and were left out")

    nrng = np.random.default_rng(seed)
    # Babble is other people talking: clean human-directed clips, by path.
    babble_paths = [f"audio/{p.name}" for (_, ln, _, _, p) in clean
                    if ln.addressed != "assistant_directed"][:40]
    for clip_id, line, voice, wpm, path in clean:
        base = {
            **line.as_row(), "source": "synthetic_say", "license": "Apple-SLA-personal",
            "consent": "", "speaker": speaker_id(voice), "voice": voice,
            "session": f"synthetic-{seed}", "room": "tts", "mic": "tts",
            "rate_wpm": wpm, "path": f"audio/{path.name}",
        }
        if keep_clean:
            rows.append({**base, "id": clip_id, "augment": None})
        for k in range(augment_copies):
            aug_id = f"{clip_id}-a{k}"
            picks = [str(b) for b in nrng.choice(babble_paths, size=min(3, len(babble_paths)),
                                                 replace=False)] if babble_paths else []
            recipe = {"seed": recipe_seed(aug_id), "babble": picks}
            # Realise it once, in memory, only to record what it did.
            _, info = augment(read_wav(path)[0], np.random.default_rng(recipe["seed"]),
                              np.concatenate([read_wav(out_dir / b)[0] for b in picks]) if picks else None)
            rows.append({**base, "id": aug_id, "augment": recipe, "augment_info": info,
                         "room": "simulated", "mic": "simulated"})
    for k in range(noise_clips):
        noise_id = f"noise{seed:02d}-{k:04d}"
        gen = {"kind": "noise", "seed": recipe_seed(noise_id)}
        _, info = noise_clip(np.random.default_rng(gen["seed"]))
        rows.append({"id": noise_id, "path": None, "generate": gen, "text": "",
                     "g_addressed": "background_or_media", "g_intent": None, "g_slots": {},
                     "category": "noise", "source": "noise", "license": "CC0-1.0",
                     "consent": "", "speaker": "noise", "session": f"noise-{seed}",
                     "room": "none", "mic": "none", "augment": None, "augment_info": info})
    return rows


# -- public corpora ------------------------------------------------------------

def _download(url: str, dest: Path, log=print) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  downloading {url}")
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed public URLs
            shutil.copyfileobj(resp, tmp)
    Path(tmp.name).rename(dest)
    return dest


def import_ami(out_dir: str | Path, meetings: Sequence[str] = AMI_MEETINGS[:4], *,
               max_segments: int = 150, vad: str = "silero", seed: int = 0,
               log=print) -> list[dict]:
    """Far-field AMI meeting audio, cut into utterances by Juno's own VAD.

    Labels are human_directed by construction: these are people in a meeting
    talking to each other. The meeting is the session; its speakers are not
    separated (the array channel mixes them), so the meeting is also the
    speaker group, which keeps a whole meeting inside one split.
    """
    from juno_core.audio.vad import EnergyVAD, SegmentDetector, SileroVAD
    from juno_core.config import Section

    out_dir = Path(out_dir)
    raw = out_dir / "raw" / "ami"
    rows = []
    rng = np.random.default_rng(seed)
    vad_cfg = Section({"threshold": 0.5, "min_speech_ms": 200, "min_silence_ms": 550,
                       "min_segment_ms": 600, "max_segment_ms": 12000})
    from juno_core.audio.capture import AudioFrame

    for meeting in meetings:
        wav = _download(AMI_URL.format(m=meeting), raw / f"{meeting}.Array1-01.wav", log)
        audio, rate = read_wav(wav)
        if rate != RATE:
            raise ValueError(f"{wav}: expected 16 kHz")
        backend = SileroVAD(sample_rate=RATE) if vad == "silero" else EnergyVAD()
        detector = SegmentDetector(backend, vad_cfg, RATE, preroll=0.3)
        spans = []
        frame = 512
        for i in range(0, audio.size - frame, frame):
            seg = detector.push(AudioFrame(samples=audio[i:i + frame], timestamp=i / RATE,
                                           adc_time=i / RATE, index=i // frame))
            if seg is not None:
                end = i + frame
                spans.append((max(0, end - seg.audio.size) / RATE, end / RATE))
        del audio
        if len(spans) > max_segments:
            keep = sorted(rng.choice(len(spans), max_segments, replace=False))
            spans = [spans[k] for k in keep]
        for k, (start, end) in enumerate(spans):
            rows.append({"id": f"ami-{meeting}-{k:04d}", "path": f"raw/ami/{wav.name}",
                         "start": round(start, 4), "end": round(end, 4), "text": "",
                         "g_addressed": "human_directed", "g_intent": None, "g_slots": {},
                         "category": "real_human", "source": "ami", "license": "CC-BY-4.0",
                         "consent": "", "speaker": f"ami-{meeting}", "session": f"ami-{meeting}",
                         "room": f"ami-{meeting[:2]}", "mic": "array1-01", "augment": None})
        log(f"  {meeting}: {len(spans)} segments")
    return rows


def import_speech_commands(out_dir: str | Path, *, per_word: int = 120, seed: int = 0,
                           log=print) -> list[dict]:
    """Real voices saying "stop", "yes", "no" (Speech Commands v0.02 test set)."""
    out_dir = Path(out_dir)
    raw = out_dir / "raw" / "speech_commands"
    archive = _download(SPEECH_COMMANDS_URL, raw / "speech_commands_test_set_v0.02.tar.gz", log)
    extracted = raw / "extracted"
    rng = random.Random(seed)
    rows = []
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".wav")]
        by_word: dict[str, list] = {}
        for m in members:
            parts = Path(m.name).parts
            if len(parts) >= 2 and parts[-2] in SPEECH_COMMAND_INTENTS:
                by_word.setdefault(parts[-2], []).append(m)
        for word, items in sorted(by_word.items()):
            rng.shuffle(items)
            for m in items[:per_word]:
                name = Path(m.name).name
                speaker = name.split("_")[0]
                target = extracted / m.name
                if not target.exists():
                    try:                                          # the published file, as is
                        tar.extract(m, extracted, filter="data")
                    except TypeError:                             # Python without tar filters
                        tar.extract(m, extracted)
                rows.append({"id": f"sc-{word}-{name[:-4]}",
                             "path": str(target.relative_to(out_dir)),
                             "text": word, "g_addressed": "assistant_directed",
                             "g_intent": SPEECH_COMMAND_INTENTS[word], "g_slots": {},
                             "category": "real_command", "source": "speech_commands",
                             "license": "CC-BY-4.0", "consent": "", "speaker": f"sc-{speaker}",
                             "session": f"sc-{speaker}", "room": "speech_commands",
                             "mic": "various", "augment": None})
        log(f"  speech commands: {len(rows)} clips")
    return rows


if __name__ == "__main__":  # pragma: no cover
    print("use: python -m juno_core.slu --help", file=sys.stderr)
