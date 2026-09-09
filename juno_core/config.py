"""Configuration loading.

Everything is driven by one YAML file. Values come back as a lightweight
attribute/dict hybrid: ``cfg.vad.threshold`` raises a clear error if that key
is missing (use this when the key is required), ``cfg.get("key", default)``
never raises (use this when it's optional).

Environment overrides use the form JUNO_<SECTION>__<KEY>, e.g.
``JUNO_LLM__MODEL=gpt-4o-mini``. Handy for secrets and for CI, without editing
the file.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PREFIX = "JUNO_"


class Section(Mapping):
    """Read-only mapping that also supports attribute access."""

    def __init__(self, data: dict):
        self._data = {
            k: Section(v) if isinstance(v, dict) else v for k, v in data.items()
        }

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(
                f"no config key {name!r} (available: {sorted(self._data)})"
            ) from exc

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict:
        return {
            k: v.to_dict() if isinstance(v, Section) else copy.deepcopy(v)
            for k, v in self._data.items()
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Section({sorted(self._data)})"


def _coerce(raw: str) -> Any:
    """Parse an environment override with YAML semantics (true/12/1.5/null)."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _apply_env_overrides(data: dict) -> list[str]:
    applied = []
    for env_key, raw in os.environ.items():
        if not env_key.startswith(ENV_PREFIX) or "__" not in env_key:
            continue
        path = env_key[len(ENV_PREFIX):].lower().split("__")
        cursor = data
        for part in path[:-1]:
            if not isinstance(cursor.get(part), dict):
                cursor = None
                break
            cursor = cursor[part]
        if cursor is None or path[-1] not in cursor:
            continue
        cursor[path[-1]] = _coerce(raw)
        applied.append(".".join(path))
    return applied


def load_config(path: str | Path | None = None) -> Section:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"no config file at {path} -- copy config.example.yaml to "
            f"config.yaml (or pass --config path/to/yours.yaml) and edit it"
        )
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    _apply_env_overrides(data)
    data.setdefault("_meta", {})["path"] = str(path)
    return Section(data)


def set_scalar(lines: list[str], section: str, key: str, value: str) -> str | None:
    """Set one `key: value` inside a top-level section, in place, preserving
    every comment and everything else about the file. Returns the old value.

    Used by enroll.py to write the measured voice threshold back into
    config.yaml without a full YAML round-trip, which would silently drop
    every comment in the file -- and this file is meant to be read.
    """
    depth = 0
    path = section.split(".")
    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if depth < len(path):
            if indent == depth * 2 and line.strip().startswith(path[depth] + ":"):
                depth += 1
            continue
        if indent < depth * 2:
            return None
        stripped = line.strip()
        if stripped.startswith(key + ":"):
            comment = ""
            if "#" in line:
                comment = "   " + line[line.index("#"):].rstrip()
            old = stripped[len(key) + 1:].split("#")[0].strip()
            lines[index] = f"{' ' * indent}{key}: {value}{comment}"
            return old
    return None
