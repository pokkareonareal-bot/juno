"""System One: the typed contract, the router's rules, the student, the wiring.

Synthetic vectors and signals only -- no speech, no downloaded model. The
encoder used here is the numpy log-mel baseline, so these run anywhere.
Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from juno_core.slu import data as D
from juno_core.slu import training as TR
from juno_core.slu.encoder import LogMelEncoder, parse_spec, pool
from juno_core.slu.parse import TextParser, parse_duration
from juno_core.slu.router import ConversationState, Router, Thresholds
from juno_core.slu.schema import (
    ADDRESSEES, ASSISTANT, CORE_SCHEMA, OPEN_REQUEST, Decision, Field, IntentSpec, Schema,
    SchemaError, SlotSpec, validate_decision,
)
from juno_core.slu.student import OTHER, Prediction, StudentModel, build_targets, fit
from juno_core.slu.system_one import SystemOne


def prediction(assistant=0.9, intent="stop", p_intent=0.95, slots=None):
    rest = (1 - assistant) / 2
    intents = {name: (1 - p_intent) / (len(CORE_SCHEMA.intent_names) - 1)
               for name in CORE_SCHEMA.intent_names}
    intents[intent] = p_intent
    return Prediction(
        addressed={ASSISTANT: assistant, "human_directed": rest, "background_or_media": rest},
        intent=intents, slots=slots or {})


OPEN_ALL = Thresholds(ignore_below=0.2, act_addressed=0.8, act_intent=0.8, act_slot=0.8)


class SchemaContract(unittest.TestCase):
    def test_core_schema_round_trips(self):
        again = Schema.from_dict(json.loads(json.dumps(CORE_SCHEMA.to_dict())))
        self.assertEqual(again.intent_names, CORE_SCHEMA.intent_names)
        self.assertEqual(again.slot_keys(), ("timer.set:duration",))

    def test_open_request_is_always_present(self):
        schema = Schema.from_dict({"name": "x", "intents": [{"name": "lights.off"}]})
        self.assertIn(OPEN_REQUEST, schema.intent_names)
        self.assertTrue(schema.intent(OPEN_REQUEST).needs_transcript)

    def test_bad_declarations_are_refused(self):
        with self.assertRaises(SchemaError):
            IntentSpec("Lights Off")
        with self.assertRaises(SchemaError):
            SlotSpec("level", "number")             # no values to choose among
        with self.assertRaises(SchemaError):
            IntentSpec("x.y", when=("whenever",))
        with self.assertRaises(SchemaError):
            Schema.from_dict({"intents": [{"name": "a"}, {"name": "a"}]})

    def test_extend_lets_the_agent_win(self):
        mine = Schema.from_dict({"name": "agent", "intents": [
            {"name": "stop", "min_confidence": 0.99},
            {"name": "lights.off", "examples": ["lights off"]}]})
        merged = CORE_SCHEMA.extend(mine)
        self.assertEqual(merged.intent("stop").min_confidence, 0.99)
        self.assertIn("lights.off", merged.intent_names)
        self.assertEqual(merged.intent_names.count(OPEN_REQUEST), 1)

    def test_decisions_validate_and_serialise(self):
        act = Decision(route="act", source="system_one",
                       addressed=Field(ASSISTANT, 0.97, {a: 0.01 for a in ADDRESSEES}),
                       intent=Field("timer.set", 0.9), slots={"duration": Field(420, 0.88)})
        data = json.loads(act.to_json(full=True))
        self.assertEqual(validate_decision(data, CORE_SCHEMA), [])
        self.assertEqual(act.summary(), "timer.set(duration=420)")
        missing = dict(data, slots={})
        self.assertTrue(validate_decision(missing, CORE_SCHEMA))
        open_act = dict(data, intent={"value": OPEN_REQUEST, "p": 0.9}, slots={})
        self.assertTrue(any("transcript" in p for p in validate_decision(open_act, CORE_SCHEMA)))


class TextSide(unittest.TestCase):
    def test_durations(self):
        cases = {"7 minutes": 420, "half an hour": 1800, "an hour and a half": 5400,
                 "two and a half minutes": 150, "three quarters of an hour": 2700,
                 "1 hour 30 minutes": 5400, "ninety seconds": 90, "twenty five minutes": 1500,
                 "nothing here": None}
        for text, seconds in cases.items():
            self.assertEqual(parse_duration(text), seconds, text)

    def test_short_commands_are_anchored(self):
        parser = TextParser()
        self.assertEqual(parser.parse("Juno, stop.").intent, "stop")
        self.assertEqual(parser.parse("I told him to stop").intent, OPEN_REQUEST)
        self.assertEqual(parser.parse("play").intent, "media.resume")
        self.assertEqual(parser.parse("play some jazz").intent, OPEN_REQUEST)

    def test_timer_without_a_length_is_open(self):
        parser = TextParser()
        self.assertEqual(parser.parse("set a timer").intent, OPEN_REQUEST)
        got = parser.parse("Hey Juno, could you set a timer for 7 minutes please?")
        self.assertEqual((got.intent, got.slots), ("timer.set", {"duration": 420}))

    def test_agent_examples_with_slots(self):
        schema = CORE_SCHEMA.extend(Schema.from_dict({"name": "a", "intents": [
            {"name": "lights.set", "examples": ["set the lights to {level}"],
             "slots": [{"name": "level", "type": "enum", "values": ["low", "high"]}]}]}))
        got = TextParser(schema).parse("Juno, set the lights to high")
        self.assertEqual((got.intent, got.slots, got.method), ("lights.set", {"level": "high"}, "example"))

    def test_aliases(self):
        parser = TextParser(aliases=("Jumo",))
        self.assertEqual(parser.parse("Hey Jumo, cancel the timer").intent, "timer.cancel")


class RouterRules(unittest.TestCase):
    def router(self, thresholds=OPEN_ALL, schema=CORE_SCHEMA):
        return Router(schema, thresholds)

    def test_no_thresholds_means_escalate(self):
        for pred in (prediction(0.01), prediction(0.999, "stop", 0.999)):
            self.assertEqual(self.router(Thresholds()).route(pred).route, "escalate")

    def test_ignore_and_its_vetoes(self):
        r = self.router()
        self.assertEqual(r.route(prediction(0.05)).route, "ignore")
        for state in (ConversationState(since_ai=3.0), ConversationState(awaiting_answer=True),
                      ConversationState(confirming=True)):
            routed = r.route(prediction(0.05), state)
            self.assertEqual(routed.route, "escalate", state)

    def test_act_needs_a_typed_confident_intent(self):
        r = self.router()
        self.assertEqual(r.route(prediction(0.95, "stop", 0.95)).route, "act")
        self.assertEqual(r.route(prediction(0.95, OPEN_REQUEST, 0.99)).route, "escalate")
        self.assertEqual(r.route(prediction(0.95, "stop", 0.6)).route, "escalate")
        self.assertEqual(r.route(prediction(0.7, "stop", 0.99)).route, "escalate")

    def test_context_bound_intents(self):
        r = self.router()
        yes = prediction(0.95, "confirm.yes", 0.97)
        self.assertEqual(r.route(yes).route, "escalate")
        self.assertEqual(r.route(yes, ConversationState(awaiting_answer=True, since_ai=1)).route, "act")
        cancel = prediction(0.95, "timer.cancel", 0.97)
        self.assertEqual(r.route(cancel).route, "escalate")
        self.assertEqual(r.route(cancel, ConversationState(active=frozenset({"timer_running"}))).route,
                         "act")

    def test_slots(self):
        r = self.router()
        values = {v: 0.0 for v in CORE_SCHEMA.intent("timer.set").slot("duration").values}
        confident = {**values, 420: 0.95}
        other = {**values, OTHER: 0.95}
        unsure = {**values, 420: 0.5, 480: 0.5}
        for dist, route in ((confident, "act"), (other, "escalate"), (unsure, "escalate")):
            pred = prediction(0.95, "timer.set", 0.95, {"timer.set:duration": dist})
            self.assertEqual(r.route(pred).route, route)

    def test_agent_min_confidence_only_tightens(self):
        schema = CORE_SCHEMA.extend(Schema.from_dict({"name": "a", "intents": [
            {"name": "stop", "min_confidence": 0.99}]}))
        r = self.router(schema=schema)
        self.assertEqual(r.route(prediction(0.95, "stop", 0.95)).route, "escalate")
        self.assertEqual(r.route(prediction(0.95, "stop", 0.995)).route, "act")


def toy_rows(n=300, seed=0, dim=24):
    """Vectors whose first coordinates encode who and what -- learnable."""
    rng = np.random.default_rng(seed)
    rows, X = [], []
    kinds = [("assistant_directed", "stop", {}), ("assistant_directed", "time.now", {}),
             ("assistant_directed", "timer.set", {"duration": 420}),
             ("assistant_directed", OPEN_REQUEST, {}), ("human_directed", None, {}),
             ("background_or_media", None, {})]
    for k in range(n):
        addressed, intent, slots = kinds[k % len(kinds)]
        v = rng.standard_normal(dim) * 0.3
        v[k % len(kinds)] += 3.0
        X.append(v)
        rows.append({"id": f"r{k}", "speaker": f"s{k % 15}", "g_addressed": addressed,
                     "g_intent": intent, "g_slots": slots, "source": "noise", "license": "CC0-1.0",
                     "category": "toy",
                     "t_addressed": {a: (0.9 if a == addressed else 0.05) for a in ADDRESSEES},
                     "t_accept": addressed == ASSISTANT, "t_intent": intent or OPEN_REQUEST,
                     "t_intent_p": 1.0, "t_slots": slots, "stt_ms": 50.0})
    return rows, np.asarray(X, np.float32)


class Student(unittest.TestCase):
    def train(self, rows, X, mode="teacher"):
        T = build_targets(rows, CORE_SCHEMA, mode)
        return fit(X, T, X, T, schema=CORE_SCHEMA, encoder_spec="logmel:64:stats",
                   hidden=16, epochs=60, lr=1e-2, seed=0)

    def test_learns_and_round_trips(self):
        rows, X = toy_rows()
        model = self.train(rows, X)
        preds = TR.predict_all(model, X)
        top = [max(p.addressed, key=p.addressed.get) for p in preds]
        self.assertGreater(np.mean([t == r["g_addressed"] for t, r in zip(top, rows)]), 0.95)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.npz"
            model.save(path)
            again = StudentModel.load(path)
            a, b = model.probabilities(X[:5]), again.probabilities(X[:5])
            for head in a:
                np.testing.assert_allclose(a[head], b[head], rtol=1e-6)
            self.assertEqual(again.encoder_spec, "logmel:64:stats")

    def test_slot_off_the_list_targets_other(self):
        rows, _ = toy_rows(12)
        rows[2]["g_slots"] = {"duration": 777}
        T = build_targets(rows, CORE_SCHEMA, "gold")
        target = T.slots["timer.set:duration"][2]
        classes = CORE_SCHEMA.intent("timer.set").slot("duration").values
        self.assertEqual(int(np.argmax(target)), len(classes))     # the OTHER column

    def test_intent_head_ignores_speech_for_people(self):
        rows, _ = toy_rows(12)
        T = build_targets(rows, CORE_SCHEMA, "gold")
        for row, mask in zip(rows, T.intent_mask):
            self.assertEqual(bool(mask), row["g_addressed"] == ASSISTANT)

    def test_load_refuses_non_finite(self):
        rows, X = toy_rows(60)
        model = self.train(rows, X)
        model.params["Wa"][0, 0] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp) / "bad.npz")
            with self.assertRaises(ValueError):
                StudentModel.load(Path(tmp) / "bad.npz")

    def test_thresholds_respect_budgets(self):
        rows, X = toy_rows(600, seed=1)
        model = self.train(rows[:300], X[:300])
        th, info = TR.choose_thresholds(model, X[300:], rows[300:], CORE_SCHEMA,
                                        max_false_ignore=0.01, max_wrong_act=0.01,
                                        max_false_activation=0.01)
        self.assertTrue(th.can_ignore and th.can_act, info)
        report = TR.evaluate(model, X[300:], rows[300:], thresholds=th)
        self.assertLessEqual(report["false_ignore"]["rate"], 0.01)
        self.assertLessEqual(report["false_activation"]["rate"], 0.01)
        self.assertGreater(report["stt_avoided"], 0.3)
        self.assertIn("end_to_end", report)


class FastPath(unittest.TestCase):
    def model(self):
        rows, X = toy_rows(120)
        enc = LogMelEncoder()
        X = np.random.default_rng(0).standard_normal((120, enc.vector_dim)).astype(np.float32)
        T = build_targets(rows, CORE_SCHEMA, "gold")
        model = fit(X, T, X, T, schema=CORE_SCHEMA, encoder_spec=enc.spec, hidden=8, epochs=3)
        model.meta["thresholds"] = OPEN_ALL.to_dict()
        return model, enc

    def test_off_by_default_and_safe_when_broken(self):
        self.assertFalse(SystemOne().enabled)
        broken = SystemOne({"mode": "on", "model_path": "/nonexistent/model.npz"})
        self.assertFalse(broken.enabled)
        self.assertTrue(broken.error)

    def test_mismatched_encoder_is_refused(self):
        model, _ = self.model()
        s1 = SystemOne({"mode": "on"}, model=model, encoder=LogMelEncoder(n_mels=40))
        self.assertFalse(s1.enabled)
        self.assertIn("trained on", s1.error)

    def test_decides(self):
        model, enc = self.model()
        s1 = SystemOne({"mode": "shadow"}, model=model, encoder=enc)
        self.assertTrue(s1.enabled)
        audio = (0.05 * np.random.default_rng(3).standard_normal(16000)).astype(np.float32)
        d = s1.decide(audio)
        self.assertIn(d.route, ("act", "ignore", "escalate"))
        self.assertEqual(d.source, "system_one")
        self.assertEqual(validate_decision(d.as_json(), CORE_SCHEMA) if d.route == "act" else [], [])
        self.assertGreater(d.latency_ms["system_one"], 0.0)

    def test_config_only_tightens(self):
        model, enc = self.model()
        s1 = SystemOne({"mode": "on", "thresholds": {"act_intent": 0.5, "ignore_below": 0.4}},
                       model=model, encoder=enc)
        self.assertEqual(s1.router.thresholds.act_intent, 0.8)
        self.assertEqual(s1.router.thresholds.ignore_below, 0.2)

    def test_schema_with_untrained_intents_is_refused(self):
        model, enc = self.model()
        agent = CORE_SCHEMA.extend(Schema.from_dict({"name": "a", "intents": [{"name": "lights.off"}]}))
        s1 = SystemOne({"mode": "on"}, model=model, encoder=enc, schema=agent)
        self.assertFalse(s1.enabled)


class Encoders(unittest.TestCase):
    def test_logmel_and_pooling(self):
        enc = LogMelEncoder()
        audio = (0.1 * np.sin(np.arange(24000) * 0.05)).astype(np.float32)
        v = enc.encode(audio)
        self.assertEqual(v.shape, (enc.vector_dim,))
        self.assertTrue(np.all(np.isfinite(v)))
        np.testing.assert_allclose(enc.encode(audio * 0.1), v, atol=1e-3)   # level-invariant
        self.assertEqual(pool(np.ones((9, 4)), "stats_thirds").shape, (24,))
        self.assertTrue(np.all(np.isfinite(enc.encode(np.zeros(100, np.float32)))))

    def test_gate_features_as_an_encoder(self):
        from juno_core.slu.encoder import build_encoder

        enc = build_encoder("gate")
        v = enc.encode((0.1 * np.sin(np.arange(24000) * 0.05)).astype(np.float32))
        self.assertEqual((enc.spec, v.shape), ("gate:20", (20,)))
        self.assertEqual(build_encoder(enc.spec).spec, "gate:20")

    def test_spec_round_trip(self):
        c = parse_spec("parakeet:mlx-community/parakeet-tdt_ctc-110m@L8:stats_thirds")
        self.assertEqual((c.kind, c.repo, c.layers, c.pooling),
                         ("parakeet", "mlx-community/parakeet-tdt_ctc-110m", 8, "stats_thirds"))
        self.assertEqual(parse_spec("logmel:64:stats").kind, "logmel")


class DataPolicy(unittest.TestCase):
    def test_provenance(self):
        self.assertFalse(D.row_trainable({"source": "synthetic_say"}))
        self.assertTrue(D.row_trainable({"source": "synthetic_say"}, allow_internal=True))
        self.assertTrue(D.row_trainable({"source": "ami", "license": "CC-BY-4.0"}))
        self.assertFalse(D.row_trainable({"source": "shadow_log"}, allow_internal=True))
        self.assertTrue(D.row_trainable({"source": "consented", "consent": "release"}))
        self.assertFalse(D.row_distributable({"source": "consented", "consent": "internal"}))
        self.assertFalse(D.provenance([{"source": "ami", "license": "CC-BY-4.0"},
                                       {"source": "synthetic_say"}])["distributable"])

    def test_augment_is_finite_and_bounded(self):
        rng = np.random.default_rng(0)
        clean = (0.3 * np.sin(np.arange(16000) * 0.03)).astype(np.float32)
        for _ in range(10):
            out, info = D.augment(clean, rng)
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertLessEqual(float(np.max(np.abs(out))), 1.0)
            self.assertGreater(out.size, clean.size)
        noise, _ = D.noise_clip(rng)
        self.assertTrue(np.all(np.isfinite(noise)))


class CommandLine(unittest.TestCase):
    def test_train_and_evaluate_from_files(self):
        rows, X = toy_rows(360, seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            D.write_manifest(tmp / "rows.jsonl", rows)
            np.savez(tmp / "emb.npz", ids=np.asarray([r["id"] for r in rows]), X=X,
                     encode_ms=np.full(len(rows), 1.0), spec=np.asarray("logmel:64:stats"))
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = TR.main(["train", "--rows", str(tmp / "rows.jsonl"), "--embeddings",
                                str(tmp / "emb.npz"), "--out", str(tmp / "m.npz"),
                                "--hidden", "16", "--epochs", "60", "--lr", "0.01",
                                "--targets", "gold"])
            self.assertIn(code, (0, 2))
            summary = json.loads(out.getvalue())
            self.assertIn("holdout", summary)
            model = StudentModel.load(tmp / "m.npz")
            self.assertTrue(model.meta["splits"]["holdout_ids"])
            self.assertTrue(model.distributable)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(TR.main(["evaluate", "--rows", str(tmp / "rows.jsonl"),
                                          "--embeddings", str(tmp / "emb.npz"),
                                          "--model", str(tmp / "m.npz"), "--only-holdout"]), 0)

    def test_sweep_writes_one_table(self):
        rows, X = toy_rows(360, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            D.write_manifest(tmp / "rows.jsonl", rows)
            (tmp / "emb").mkdir()
            np.savez(tmp / "emb" / "logmel.npz", ids=np.asarray([r["id"] for r in rows]), X=X,
                     encode_ms=np.full(len(rows), 1.0), spec=np.asarray("logmel:64:stats"))
            with contextlib.redirect_stderr(io.StringIO()):
                code = TR.main(["sweep", "--rows", str(tmp / "rows.jsonl"), "--emb-dirs",
                                str(tmp / "emb"), "--targets", "teacher", "gold", "--seeds", "0",
                                "--hidden", "16", "--epochs", "30", "--lr", "0.01",
                                "--out", str(tmp / "sweep")])
            self.assertEqual(code, 0)
            lines = (tmp / "sweep" / "sweep.csv").read_text().strip().splitlines()
            self.assertEqual(len(lines), 3)                 # header + two runs
            self.assertIn("| logmel:64:stats | gold |", (tmp / "sweep" / "sweep.md").read_text())

    def test_shadow_report(self):
        from juno_core.slu.bench import shadow_report

        events = [
            {"event": "slu_decided", "turn": "a", "route": "ignore",
             "addressed": {"value": "human_directed", "p": 0.9, "probs": {ASSISTANT: 0.05}},
             "latency_ms": {"system_one": 15.0}},
            {"event": "slu_system_two", "turn": "a", "route": "act", "transcript": "louder",
             "intent": {"value": "volume.up", "p": 1.0}},
            {"event": "slu_decided", "turn": "b", "route": "act", "intent": {"value": "stop", "p": 0.97},
             "slots": {}, "latency_ms": {"system_one": 14.0}},
            {"event": "slu_system_two", "turn": "b", "route": "act",
             "intent": {"value": "stop", "p": 1.0}, "slots": {}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "events.jsonl"
            log.write_text("\n".join(json.dumps(e) for e in events) + "\n")
            report = shadow_report(log)
        self.assertEqual(report["turns"], 2)
        self.assertEqual(report["would_have_avoided_stt"], 2)
        self.assertEqual([d["kind"] for d in report["disagreements_that_matter"]], ["would_have_missed"])


class Pipeline(unittest.TestCase):
    """The fast path inside JunoPipeline, with fakes for STT and the agent."""

    def pipeline(self, mode, route, on_decision=True):
        from juno_core.audio.vad import SpeechSegment
        from juno_core.config import Section
        from juno_core.pipeline import JunoPipeline
        from juno_core.stt import STTEngine, Transcript

        calls = {"stt": 0, "decisions": [], "accepted": []}

        class FakeSTT(STTEngine):
            def transcribe(self, audio, sample_rate=16000):
                calls["stt"] += 1
                return Transcript(text="Juno, what's the capital of Mongolia?")

        config = Section({"audio": {"sample_rate": 16000, "channels": 1, "frame_ms": 32},
                          "vad": {"backend": "energy", "threshold": 0.5, "min_speech_ms": 200,
                                  "min_silence_ms": 550, "min_segment_ms": 350,
                                  "max_segment_ms": 20000},
                          "intent": {"gate": {"mode": "off"}}, "system_one": {"mode": "off"}})
        p = JunoPipeline(config, stt=FakeSTT(),
                         on_accept=lambda text, d, c: calls["accepted"].append(text),
                         on_decision=(lambda d, c: calls["decisions"].append(d)) if on_decision else None)
        model, enc = FastPath().model()

        class Fixed(SystemOne):
            def decide(self, audio, sample_rate=16000, state=None, turn=None):
                d = super().decide(audio, sample_rate, state, turn)
                d.route = route
                if route == "act":
                    d.intent, d.slots = Field("stop", 0.97), {}
                return d

        p.system_one = Fixed({"mode": mode}, model=model, encoder=enc)
        segment = SpeechSegment(audio=np.zeros(16000, np.float32), start_time=0.0, end_time=1.0,
                                confidence=0.95)
        p._process_segment(segment)
        return calls

    def test_on_ignore_skips_stt(self):
        calls = self.pipeline("on", "ignore")
        self.assertEqual((calls["stt"], calls["decisions"]), (0, []))

    def test_on_act_skips_stt_and_delivers(self):
        calls = self.pipeline("on", "act")
        self.assertEqual(calls["stt"], 0)
        self.assertEqual([d.source for d in calls["decisions"]], ["system_one"])

    def test_act_without_a_decision_handler_escalates(self):
        calls = self.pipeline("on", "act", on_decision=False)
        self.assertEqual(calls["stt"], 1)
        self.assertEqual(len(calls["accepted"]), 1)

    def test_escalate_delivers_system_two_with_transcript(self):
        calls = self.pipeline("on", "escalate")
        self.assertEqual(calls["stt"], 1)
        (d,) = calls["decisions"]
        self.assertEqual((d.source, d.intent.value), ("system_two", OPEN_REQUEST))
        self.assertTrue(d.transcript)

    def test_shadow_changes_nothing(self):
        calls = self.pipeline("shadow", "ignore")
        self.assertEqual(calls["stt"], 1)
        self.assertEqual(len(calls["decisions"]), 1)


if __name__ == "__main__":
    unittest.main()


class Collect(unittest.TestCase):
    """The consented session: vectors and teacher labels, never audio."""

    def test_session_keeps_vectors_and_labels_only(self):
        from test_pre_stt_gate import FakeMic

        from juno_core.slu.collect import SLU_SCRIPT, SLUCollector
        from juno_core.slu.teacher import TeacherLabel

        class FakeTeacher:
            def label(self, audio, rate=16000):
                return TeacherLabel(transcript="juno stop", reliable=True, accepted=True,
                                    confidence=0.9, method="heuristic",
                                    addressed={a: 1 / 3 for a in ADDRESSEES},
                                    intent="stop", intent_p=1.0)

        enc = LogMelEncoder()
        collector = SLUCollector(
            {"vad": {"backend": "energy"}}, encoders=[enc], teacher=FakeTeacher(),
            speakers=["p1", "p2"], session="s1", room="kitchen", mic="fake", consent="release",
            capture=FakeMic(lambda: 100.0), ask=lambda prompt: "", say=lambda *a: None,
            clock=lambda: 100.0)
        rows = collector.run(SLU_SCRIPT, scale=0.2)
        self.assertTrue(rows)
        self.assertEqual(len(collector.vectors[enc.spec]), len(rows))
        self.assertEqual({r["g_addressed"] for r in rows},
                         {"assistant_directed", "human_directed", "background_or_media"})
        for r in rows:
            self.assertIsNone(r["path"])                         # nothing to replay: no audio
            self.assertEqual((r["source"], r["consent"]), ("consented", "release"))
            self.assertTrue(D.row_trainable(r) and D.row_distributable(r))
            if r["g_addressed"] != ASSISTANT:
                self.assertIsNone(r["g_intent"])
        self.assertIn("timer.set", {r["g_intent"] for r in rows})


class BetterTeacher(unittest.TestCase):
    """The language-model judge, with a scripted backend standing in for the model."""

    class Scripted:
        name = "scripted"

        def __init__(self, who, intent_letter=None):
            self.who, self.intent_letter, self.calls = who, intent_letter, 0

        def choose(self, system, question, item, n):
            self.calls += 1
            p = np.full(n, 0.01)
            letter = self.who if n == 3 else (self.intent_letter or "A")
            p["ABCDEFGHIJKLM".index(letter)] = 1.0
            return p / p.sum()

    def teacher(self, backend, weight=1.0):
        from juno_core.slu.judge import LLMJudge
        from juno_core.slu.teacher import Teacher

        return Teacher(None, judge=LLMJudge(backend), judge_weight=weight)

    def test_judge_overrules_a_cold_engine_on_who(self):
        # "a little louder" heard cold: the engine rejects it; the judge says Juno.
        intents = [i.name for i in CORE_SCHEMA.intents if i.name != OPEN_REQUEST]
        letter = "ABCDEFGHIJKLM"[intents.index("volume.up")]
        label = self.teacher(self.Scripted("A", letter)).judge_text("a little louder")
        self.assertTrue(label.accepted)
        self.assertEqual(label.intent, "volume.up")
        row = label.as_row()
        self.assertGreater(row["t_judge_addressed"][ASSISTANT], 0.9)
        self.assertLess(row["t_engine_addressed"][ASSISTANT], 0.5)
        self.assertAlmostEqual(sum(row["t_intent_probs"].values()), 1.0, places=3)

    def test_rules_keep_exact_parses_and_slots(self):
        # The judge points at stop; the parser read a timer with a length -- rules win.
        label = self.teacher(self.Scripted("A", "A")).judge_text("set a timer for 7 minutes")
        self.assertEqual((label.intent, label.slots), ("timer.set", {"duration": 420}))
        self.assertGreater(label.intent_probs["timer.set"], 0.85)

    def test_timer_without_a_length_stays_open(self):
        intents = [i.name for i in CORE_SCHEMA.intents if i.name != OPEN_REQUEST]
        letter = "ABCDEFGHIJKLM"[intents.index("timer.set")]
        label = self.teacher(self.Scripted("A", letter)).judge_text("could you time my eggs")
        self.assertEqual(label.intent, OPEN_REQUEST)

    def test_not_for_juno_skips_the_intent_question_and_repeats_are_free(self):
        backend = self.Scripted("B")
        t = self.teacher(backend)
        a = t.judge_text("did you feed the cat")
        t.judge_text("Did you feed the cat")
        self.assertFalse(a.accepted)
        self.assertEqual(backend.calls, 1)              # one question, asked once

    def test_blend_weight(self):
        label = self.teacher(self.Scripted("A"), weight=0.5).judge_text("a little louder")
        row = label.as_row()
        expected = 0.5 * row["t_judge_addressed"][ASSISTANT] + 0.5 * row["t_engine_addressed"][ASSISTANT]
        self.assertAlmostEqual(row["t_addressed"][ASSISTANT], expected, places=3)

    def test_soft_intent_targets_reach_training(self):
        rows, _ = toy_rows(6)
        rows[0]["t_intent_probs"] = {"stop": 0.7, "cancel": 0.3}
        T = build_targets(rows, CORE_SCHEMA, "teacher")
        names = list(CORE_SCHEMA.intent_names)
        self.assertAlmostEqual(float(T.intent[0][names.index("stop")]), 0.7, places=5)
        self.assertAlmostEqual(float(T.intent[0][names.index("cancel")]), 0.3, places=5)
