"""Reflex SLU was called System One: this keeps ``juno_core.slu.system_one`` importable.

New code: ``from juno_core.slu.reflex import Reflex``.
"""

from juno_core.slu.reflex import CORE_SCHEMA, MODES, Reflex

SystemOne = Reflex

__all__ = ["SystemOne", "MODES", "CORE_SCHEMA"]
