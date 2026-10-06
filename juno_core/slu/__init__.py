"""Spoken-language understanding without transcribing: Juno's Reflex SLU.

Juno used to transcribe everything it heard and then decide what to do with
the words. This package makes transcription optional. A small model listens
to the audio itself -- borrowing a speech recogniser's encoder, never its
decoder -- and answers, in a schema the agent declares, the two questions
that matter: was that meant for me, and what was wanted? When it is sure,
the agent gets typed JSON and no transcript is ever made. When it is not,
or when the request needs exact words, Juno escalates to the cascade
(speech-to-text plus the text-side engines, the old path), which answers in
the same shape.

The student is trained by distillation from the cascade (the teacher), on
data whose licence allows it. Modules:

    schema.py      the contract: Schema (what the agent declares), Decision
                   (what Juno answers), the core intents
    encoder.py     audio -> pooled vector (parakeet | whisper | logmel)
    student.py     the heads, their training and calibration
    router.py      probabilities -> act | ignore | escalate
    reflex.py      the runtime fast path (system_one.py: its old import path)
    teacher.py     the cascade, and the offline teacher built from it
    parse.py       words -> intent and slots (the cascade's typed half)
    data.py        synthetic speech, augmentation, public corpora, manifests
    training.py    the offline CLI: synth, label, embed, train, evaluate
    bench.py       the benchmark: Reflex against always-transcribe baselines

    python -m juno_core.slu --help
"""

from juno_core.slu.schema import (
    ADDRESSEES, CORE_SCHEMA, OPEN_REQUEST, Decision, Field, IntentSpec, Schema,
    SchemaError, SlotSpec, validate_decision,
)

__all__ = [
    "ADDRESSEES", "CORE_SCHEMA", "OPEN_REQUEST", "Decision", "Field", "IntentSpec",
    "Schema", "SchemaError", "SlotSpec", "validate_decision",
]
