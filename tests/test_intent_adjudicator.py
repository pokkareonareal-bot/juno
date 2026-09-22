"""The placeholder model must never be consulted as the second opinion."""

from __future__ import annotations

import unittest

from juno_core.intelligence.context import ConversationContext
from juno_core.intelligence.intent import IntentEngine
from juno_core.llm import BaseModel, EchoModel, ModelResponse


class Always(BaseModel):
    name = "always-9"

    def generate(self, messages, **kwargs):
        return ModelResponse(text="9")


class Adjudicator(unittest.TestCase):
    def engine(self, model):
        return IntentEngine({"llm_adjudicator": True}, ConversationContext(), model=model)

    def test_echo_is_not_an_adjudicator(self):
        self.assertIsNone(self.engine(EchoModel()).adjudicator)

    def test_real_model_is(self):
        self.assertIsNotNone(self.engine(Always()).adjudicator)

    def test_ambiguous_with_echo_stays_heuristic(self):
        engine = self.engine(EchoModel())
        for text in ("I think it's in the other room, maybe 9 or so",
                     "hmm, what about the other one", "so we should probably go"):
            decision = engine.classify(text, reliable=True)
            self.assertNotEqual(decision.method, "llm")
            self.assertIsNone(decision.llm_confidence)


if __name__ == "__main__":
    unittest.main()
