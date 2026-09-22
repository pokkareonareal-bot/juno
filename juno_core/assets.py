"""The two model files Juno downloads, and nothing else.

    python -m juno_core.assets            # fetch both now (e.g. before going offline)
    python -m juno_core.assets --list     # where they live, and their licences

Both are fetched on first use if they're missing, so this command is only
needed ahead of time. Each is pinned to one exact published version and
checked against its SHA-256 before it is used: a changed file upstream is an
error, never a silent swap.

Where they go: ``$JUNO_MODELS_DIR`` if set, else ``~/.cache/juno/models``. A
copy already sitting in ``juno_core/data/models/`` (the old location) is used
as-is.

What they are, and their terms:

  - Silero VAD v6.2.2 (MIT, Silero Team) -- voice activity detection.
  - WeSpeaker ECAPA-TDNN512-LM (CC BY 4.0, WeSpeaker team; trained on
    VoxCeleb) -- the speaker embedding behind "was that you". CC BY requires
    attribution: keep this notice (and the README's) if you redistribute it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

LEGACY_DIR = Path(__file__).resolve().parent / "data" / "models"


@dataclass(frozen=True)
class Asset:
    key: str
    filename: str
    url: str
    sha256: str
    size: int
    title: str
    license: str
    attribution: str


ASSETS = {
    "silero_vad": Asset(
        key="silero_vad",
        filename="silero_vad.onnx",
        url=("https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.2/"
             "src/silero_vad/data/silero_vad.onnx"),
        sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        size=2_327_524,
        title="Silero VAD v6.2.2",
        license="MIT",
        attribution="Silero VAD, Silero Team, MIT licence "
                    "(https://github.com/snakers4/silero-vad)",
    ),
    "ecapa": Asset(
        key="ecapa",
        filename="ecapa_tdnn512.onnx",
        url=("https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM/resolve/"
             "a2f3dcb1c8702caccc7a55ceb57f5e8d1842112b/voxceleb_ECAPA512_LM.onnx"),
        sha256="d71b85d9b48058ef68004f04f1b78acebefb9dfcf542e19b976a12a5ad1f10b0",
        size=24_861_931,
        title="WeSpeaker ECAPA-TDNN512-LM (VoxCeleb)",
        license="CC-BY-4.0",
        attribution="WeSpeaker ECAPA-TDNN512-LM, WeSpeaker team, trained on "
                    "VoxCeleb, CC BY 4.0 "
                    "(https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM)",
    ),
}


class AssetError(RuntimeError):
    pass


def models_dir() -> Path:
    override = os.environ.get("JUNO_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "juno" / "models"


def default_path(key: str) -> Path:
    """Where the asset is, or will be put. A legacy copy wins if present."""
    asset = ASSETS[key]
    legacy = LEGACY_DIR / asset.filename
    if legacy.exists():
        return legacy
    return models_dir() / asset.filename


def ensure(key: str, path: str | Path | None = None, *, quiet: bool = False) -> Path:
    """The asset's path, downloading and verifying it first if it's missing.

    An explicit ``path`` that already exists is trusted as the user's own
    file. Raises AssetError with what to do when it can't be fetched.
    """
    asset = ASSETS[key]
    target = Path(path).expanduser() if path else default_path(key)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"downloading {asset.title} ({asset.size / 1e6:.1f} MB, {asset.license}) "
              f"-> {target}", file=sys.stderr, flush=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{asset.filename}.", dir=target.parent)
    tmp = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(asset.url, timeout=60) as src:
            while True:
                chunk = src.read(1 << 16)
                if not chunk:
                    break
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != asset.sha256:
            raise AssetError(
                f"{asset.title}: checksum mismatch (got {digest.hexdigest()[:12]}..., "
                f"expected {asset.sha256[:12]}...); not using it")
        os.replace(tmp, target)
    except AssetError:
        raise
    except Exception as exc:
        raise AssetError(
            f"couldn't download {asset.title} ({type(exc).__name__}: {exc}). "
            f"Fetch {asset.url} yourself and save it as {target}, or run "
            f"`python -m juno_core.assets` once you're online.") from exc
    finally:
        tmp.unlink(missing_ok=True)
    return target


def verify(path: Path, key: str) -> bool:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest() == ASSETS[key].sha256


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m juno_core.assets",
                                     description="Download Juno's model files.")
    parser.add_argument("--list", action="store_true", help="show paths and licences")
    args = parser.parse_args(argv)
    status = 0
    for key, asset in ASSETS.items():
        path = default_path(key)
        if args.list:
            state = "present" if path.exists() else "missing"
            print(f"{asset.title}: {path} ({state})\n  {asset.attribution}")
            continue
        try:
            ensure(key)
            ok = verify(path, key)
            print(f"{'ok' if ok else 'CHECKSUM MISMATCH'}  {asset.title}  {path}")
            status |= 0 if ok else 1
        except AssetError as exc:
            print(f"failed  {exc}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
