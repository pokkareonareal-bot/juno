"""Teaching it your voice, from enroll.py, on the first run.

audio/voiceprint.py can tell your voice from anybody else's, and has shipped
switched off until somebody enrols a voice, because there is nothing to
compare against until then.

So it happens here: say a few sentences, and the profile comes out of that.
Somebody else saying a few sharpens where the line goes, but is no longer
required -- the mechanism this replaced needed two voices to have any line at
all, and verifying a voice only needs examples of that voice. The recording is done by the assistant --
it already owns the microphone and the segmenter that finds where an utterance
starts and stops -- and the caller (enroll.py) only says what to do and how far along it is.

WHAT IT IS MEASURING, AND WHAT IT KEEPS
---------------------------------------
Who is speaking. This used to say the opposite -- it measured how near the
microphone a talker was and kept no model of anybody's voice -- and that was
true right up until the version that measured distance was replaced, because
distance is exactly wrong for somebody who wants to speak from across the
room. It is not true any more and should not be left standing: enrolment
writes 192 numbers derived from recordings of you.

That is a voice template. No audio is kept, nothing leaves the machine, it is
readable JSON, and deleting the file switches the gate off. But it exists, and
enroll.py says so rather than repeating the older, better-sounding claim.

The result is still only ever used one way. A confident "that was somebody
else" is evidence against answering. A confident "that was you" is not
evidence for it -- the wearer is also the person talking to everybody else in
the room.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# Enough to see a distribution, few enough that nobody gives up halfway. Each
# is one utterance, so this is under a minute of talking per side.
WANTED = 8

PROMPTS = [
    "What's the weather like tomorrow?",
    "Set a timer for ten minutes.",
    "How far is Tokyo from Kyoto?",
    "Remind me to call them this evening.",
    "What time does the last train leave?",
    "Tell me something I don't know.",
    "How long does it take to boil an egg?",
    "What's on my calendar today?",
]


@dataclass
class Progress:
    """What the caller needs to show the person enrolling."""

    stage: str = "idle"          # idle | wearer | bystander | done | failed
    # Whether the gate is already on, so the caller knows not to offer this
    # to somebody who has already done it.
    already_on: bool = False
    collected: int = 0
    wanted: int = WANTED
    prompt: str = ""
    message: str = ""
    result: dict | None = None


@dataclass
class Calibration:
    """Collects labelled utterances and works out where the line goes.

    Deliberately not a subclass of anything in the audio path: it borrows the
    segments the VAD has already found rather than listening on its own.
    """

    stage: str = "idle"
    # Where the measured threshold is written. None keeps it in memory, which
    # is what the tests want.
    config_path: Path | None = None
    already_on: bool = False
    wearer: list = field(default_factory=list)
    bystander: list = field(default_factory=list)
    message: str = ""
    result: dict | None = None
    started_at: float = 0.0
    # The microphone this was measured on, read at start(). The threshold is
    # only meaningful on it, so it travels with the result.
    device: str = ""
    # The own_voice section, so enrolment uses the model and profile the
    # config names. Passing a bare {"enabled": False} meant both silently
    # fell back to data/, and a configured path did nothing.
    own_voice: object = None
    # The enrolled voice, built by finish() and written by apply(). Held in
    # memory in between so that measuring and saving stay separable -- the
    # caller shows the result before anybody agrees to keep it.
    profile: object = None

    # -- the flow ---------------------------------------------------------

    @property
    def collecting(self) -> bool:
        return self.stage in ("wearer", "bystander")

    @property
    def bucket(self) -> list:
        return self.wearer if self.stage == "wearer" else self.bystander

    def start(self) -> Progress:
        self.stage = "wearer"
        self.wearer, self.bystander = [], []
        self.result, self.message = None, ""
        self.started_at = time.time()
        # Noted now rather than at the end, so that unplugging headphones
        # halfway through cannot have the result attributed to whatever is
        # connected when it finishes.
        from juno_core.audio.voiceprint import current_microphone

        self.device = current_microphone()
        return self.progress()

    def cancel(self) -> Progress:
        self.stage, self.message = "idle", ""
        self.wearer, self.bystander = [], []
        return self.progress()

    def collect(self, audio: np.ndarray) -> Progress:
        """Take one utterance for whichever side is being recorded."""
        if not self.collecting:
            return self.progress()
        # Too short to say anything about, and long enough that it is probably
        # not one sentence. Neither tells us much about distance.
        seconds = len(audio) / 16000
        if not 0.4 <= seconds <= 12.0:
            self.message = "too short" if seconds < 0.4 else "too long"
            return self.progress()
        self.message = ""
        self.bucket.append(np.asarray(audio, dtype=np.float32))
        if len(self.bucket) >= WANTED:
            self.stage = "bystander" if self.stage == "wearer" else "scoring"
            if self.stage == "scoring":
                return self.finish()
        return self.progress()

    def skip(self) -> Progress:
        """Move on without the rest of this side.

        The bystander half needs another person in the room, which is the
        part people cannot always arrange. It used to be the half that made
        the measurement possible at all. It is not any more: verifying a
        voice needs examples of that voice, and a second speaker only sharpens
        where the line goes. Skipping it gives a working, more cautious gate
        rather than none.
        """
        if self.stage == "wearer" and self.wearer:
            self.stage = "bystander"
        elif self.stage == "bystander":
            return self.finish()
        return self.progress()

    # -- the number -------------------------------------------------------

    def finish(self) -> Progress:
        """Build the voice profile, and work out where the line goes.

        The wearer's clips become the profile. The bystander's, when there
        are any, set the threshold by the same three-to-one sweep the old
        distance calibration used -- a wrong "that was not you" costs an
        answer, a wrong "that was you" costs it answering somebody else.

        WITHOUT ANYBODY ELSE IT STILL WORKS, which is the thing distance
        calibration could not do. Speaker verification only needs examples of
        the voice it is verifying; the negatives just sharpen the threshold.
        With none, half the clips enrol and the other half are scored against
        them, which says how much this voice varies between sentences on this
        microphone -- and the line goes below that, with room to spare.
        """
        from juno_core.audio.voiceprint import VoicePrint

        voiceprint = VoicePrint(self._voice_config())
        if not voiceprint.available:
            self.stage = "done"
            self.result = {"measured": False, "device": self.device,
                           "note": ("The speaker model is not downloaded. Run "
                                    "scripts/fetch_assets.py --voiceprint.")}
            return self.progress()

        profile = voiceprint.enrol(self.wearer)
        if profile is None:
            self.stage = "done"
            self.result = {"measured": False, "device": self.device,
                           "note": ("Not enough usable speech to build a voice "
                                    "profile. Nothing was saved.")}
            return self.progress()

        voiceprint._profile = profile
        mine = [s for s in (voiceprint.similarity(c) for c in self.wearer)
                if s is not None]
        theirs = [s for s in (voiceprint.similarity(c) for c in self.bystander)
                  if s is not None]

        if theirs:
            table, costs = [], []
            for step in range(0, 19):
                threshold = step / 20.0
                silenced = sum(1 for x in mine if x < threshold) / len(mine)
                passed = sum(1 for x in theirs if x >= threshold) / len(theirs)
                table.append({"threshold": round(threshold, 2),
                              "wearer_silenced": round(silenced, 3),
                              "bystander_passed": round(passed, 3)})
                costs.append(silenced + 3.0 * passed)

            # The MIDDLE of the thresholds that score equally, not the first.
            #
            # Taking the first meant taking the lowest, and the lowest is
            # 0.0 -- where nobody is silenced and, if the second voice scores
            # negative throughout (routine for two quite different voices),
            # nobody is let through either. Cost zero, unbeatable by anything
            # later, and the gate is written wide open with a line at zero
            # while reporting that it separated the voices. Reproduced: wearer
            # +0.80..+0.66 against bystander -0.05..-0.015 gave threshold 0.0.
            #
            # Every threshold in that tie is equally good on the eight clips
            # measured, so the one to pick is the one furthest from both
            # distributions -- the middle of the run, which is the usual
            # max-margin argument and is what a person would draw by eye.
            cheapest = min(costs)
            tied = [row["threshold"] for row, cost in zip(table, costs)
                    if cost <= cheapest + 1e-9]
            best = tied[len(tied) // 2]
            chosen = next(r for r in table if r["threshold"] == round(best, 2))
            separated = (float(np.median(mine)) - float(np.median(theirs))) > 0.15
            self.result = {
                "threshold": round(best, 2), "measured": True,
                "separated": separated, "device": self.device,
                "wearer_samples": len(mine), "bystander_samples": len(theirs),
                "wearer_median": round(float(np.median(mine)), 3),
                "bystander_median": round(float(np.median(theirs)), 3),
                "wearer_silenced": chosen["wearer_silenced"],
                "bystander_passed": chosen["bystander_passed"],
                "table": table,
            }
            if not separated:
                self.result["note"] = (
                    "Your voice and theirs measured too alike for this to "
                    "separate them. Left switched off."
                )
        else:
            # Half in, half scored: how much one voice varies sentence to
            # sentence. The line goes a clear margin below the worst of them,
            # because everything below it gets silenced and that is the
            # failure worth avoiding.
            half = max(2, len(self.wearer) // 2)
            solo = voiceprint.enrol(self.wearer[:half])
            if solo is None:
                self.stage = "done"
                self.result = {"measured": False, "device": self.device,
                               "note": "Not enough usable speech. Nothing saved."}
                return self.progress()
            voiceprint._profile = solo
            held = [s for s in (voiceprint.similarity(c) for c in self.wearer[half:])
                    if s is not None]
            if not held:
                self.stage = "done"
                self.result = {"measured": False, "device": self.device,
                               "note": "Not enough usable speech. Nothing saved."}
                return self.progress()
            floor = float(np.min(held))
            threshold = round(max(0.15, min(0.5, floor - 0.15)), 2)
            self.result = {
                "threshold": threshold, "measured": True, "separated": True,
                "device": self.device, "wearer_samples": len(mine),
                "bystander_samples": 0,
                "wearer_median": round(float(np.median(mine)), 3),
                "self_consistency": round(floor, 3),
                "note": ("Measured from your voice alone, with nobody else "
                         "recorded. That works here -- it only needs examples "
                         "of the voice it is checking for -- but the line is "
                         "set cautiously, so it will let more through than it "
                         "would with a second voice to measure against."),
            }

        self.profile = profile
        self.stage = "done"
        return self.progress()

    def apply(self) -> Progress:
        """Write the measured threshold into the config and switch it on.

        Refused unless the measurement actually separated. A threshold drawn
        through two overlapping distributions is not a threshold, it is a coin
        toss with a decimal point, and switching the gate on with one would
        silence the user about as often as it silenced anybody else.
        """
        if self.stage != "done" or not self.result:
            self.message = "nothing measured yet"
            return self.progress()
        if not self.result.get("measured") or not self.result.get("separated"):
            self.message = self.result.get("note") or "not measurable here"
            return self.progress()
        if self.config_path is None:
            self.message = "measured, but there is nowhere to write it"
            return self.progress()
        if not self.result.get("device"):
            self.message = (
                "Measured, but I could not tell which microphone I heard it "
                "on -- and the number only means anything on the one it was "
                "measured on. Left switched off."
            )
            return self.progress()

        from juno_core.config import set_scalar

        if self.profile is None:
            self.message = "measured, but the voice profile is missing"
            return self.progress()

        from juno_core.audio.voiceprint import VoicePrint

        try:
            VoicePrint(self._voice_config()).save_profile(
                self.profile, device=self.result.get("device", "")
            )
        except OSError as exc:
            self.message = f"could not save the voice profile: {exc}"
            return self.progress()

        try:
            text = self.config_path.read_text(encoding="utf-8")
            lines = text.splitlines()
            set_scalar(lines, "own_voice", "threshold",
                       str(self.result["threshold"]))
            # Quoted: device names have spaces in them.
            set_scalar(lines, "own_voice", "calibrated_for",
                       '"%s"' % self.result.get("device", ""))
            set_scalar(lines, "own_voice", "enabled", "true")
            self.config_path.write_text(
                "\n".join(lines) + "\n", encoding="utf-8", newline=""
            )
        except OSError as exc:
            self.message = f"could not write the config: {exc}"
            return self.progress()
        self.message = (
            f"Saved. Own-voice gating is on at {self.result['threshold']}. "
            "It takes effect next time Juno starts."
        )
        self.stage = "applied"
        return self.progress()

    def _voice_config(self):
        """The own_voice section with the gate forced off.

        Off because enrolment must not be judged by a threshold that has not
        been measured yet -- but the model and profile paths are the ones the
        config names.
        """
        from juno_core.config import Section

        data = {}
        if self.own_voice is not None:
            data = dict(self.own_voice.to_dict()
                        if hasattr(self.own_voice, "to_dict") else self.own_voice)
        data["enabled"] = False
        return Section(data)

    def progress(self) -> Progress:
        collected = len(self.bucket) if self.collecting else 0
        return Progress(
            stage=self.stage,
            already_on=self.already_on,
            collected=collected,
            wanted=WANTED,
            prompt=PROMPTS[collected % len(PROMPTS)] if self.collecting else "",
            message=self.message,
            result=self.result,
        )
