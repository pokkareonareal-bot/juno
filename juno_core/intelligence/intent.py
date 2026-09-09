"""Is this utterance addressed to the assistant?

There is no wake word, so this decision has to be made after the fact, from
the text and the conversational situation (design sections 8 and 9). It is the
component most likely to make the system feel either broken or magic, so it is
built to be measurable and tunable rather than clever:

* every decision decomposes into named, weighted signals that are logged;
* the output is a calibrated confidence, not a boolean;
* the cheap heuristic runs always, and the LLM is consulted *only* inside the
  ambiguity band. That keeps the median utterance on a sub-millisecond path
  and spends latency only where the answer is genuinely unclear.

The default on an unresolved ambiguity is to stay quiet. A missed activation
costs the user one repetition; a false activation means the device talks over
a human conversation, which is both more annoying and, for a device worn in
company, more socially expensive.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Sequence

from juno_core.intelligence.context import ConversationContext, Utterance
from juno_core.textutil import split_sentences

# --------------------------------------------------------------------------
# Signal definitions. Weights are log-odds contributions; the base rate is set
# by BIAS. Tuning these against logs/events-*.jsonl is the intended workflow.
# --------------------------------------------------------------------------

# Signals about *who is being addressed* rather than what is being asked.
# These are established by the whole utterance, so they survive when only the
# trailing clause is re-read -- "we should go to Paris" is evidence about the
# audience for the question that follows it, and dropping it lets a request
# aimed at a person score as though the planning had never been said.
_AUDIENCE_SIGNALS = frozenset({
    "bystander_addressed", "human_vocative", "social_register",
    "joint_planning", "reported_speech", "third_person_about_ai",
    "human_discourse_lead", "backchannel", "self_talk", "not_own_voice",
})

BIAS = -1.05  # prior: most speech in a room is not for the assistant (p~0.26)

# Driving something on the screen, and it took two attempts to write.
#
# The first version matched these words ANYWHERE in the utterance and, measured
# against twelve things a person says to another person while a browser happens
# to be open, produced TEN false activations: "click the kettle on would you",
# "she chose the first one in the end", "type it up when you get home". Which
# is this file's own warning ignored -- the words say what is being done, and
# say nothing about who is being asked to do it.
#
# So the shape matters more than the vocabulary. A person driving a page issues
# an IMPERATIVE: the command starts the sentence, and there is nobody in it.
# The moment a second or third person appears -- you, your, we, let's, she,
# I'll -- it is somebody talking to somebody, whatever verbs it contains.
_DRIVING_COMMAND = re.compile(
    r"^(?:ok(?:ay)?|then|now|and|no|yes|yeah|right|please)?[,\s]*"
    r"(?:can\s+you\s+|could\s+you\s+)?"
    r"(click|press|tap|select|choose|scroll|swipe|zoom|"
    r"skip|rewind|replay|mute|unmute|fullscreen|pause|resume|"
    r"go\s+back|go\s+forward|back\s+to|next|previous|"
    r"the\s+(?:first|second|third|last|other|next|previous)\s+one|"
    r"that\s+(?:button|link|tab|video)|"
    r"type\s|search\s+for|open\s+the|close\s+the\s+tab)\b",
    re.IGNORECASE,
)

# Somebody else is in the sentence, so it is not an instruction to a machine.
#
# "you" has to be in here -- "type it up when you get home" and "click the
# kettle on would you" were the two false activations the first narrowing left
# behind -- but a bare ban on it would also refuse "can you scroll down", which
# is how people talk to assistants. So the leading politeness is removed first
# and the test is applied to what is left: an addressee at the FRONT is
# probably this machine, an addressee anywhere else is a person.
_POLITE_LEAD = re.compile(r"^(?:ok(?:ay)?|then|now|and|no|yes|yeah|right)?[,\s]*"
                          r"(?:please\s+)?(?:can|could|would|will)\s+you\s+",
                          re.IGNORECASE)
_SOMEBODY_ELSE = re.compile(
    r"\b(you|your|yours|we|we'?re|we'?ll|us|let'?s|she|he|they|them|"
    r"i'?ll|i'?m|i\s+need|i\s+want|somebody|someone)\b",
    re.IGNORECASE,
)


def _addressed_to_a_person(body: str) -> bool:
    """Whether somebody other than the assistant is in the sentence."""
    return bool(_SOMEBODY_ELSE.search(_POLITE_LEAD.sub("", body, count=1)))

# Asking for something only a body can do, which this assistant does not have.
#
# "Can you pass me that one", "could you hold this for me", "can you grab that"
# all scored 0.61 to 0.74 and were ACCEPTED inside the follow-up window,
# because every signal the engine had said request: a question lead, a polite
# form, an anaphor, and the assistant having just spoken. Nothing in it noticed
# that the thing being asked for is impossible. It is one of the cleanest
# discriminators available -- a request for a physical act is a request to a
# person, always, and the room is full of people whose conversation this device
# should stay out of.
#
# Kept to verbs with no other sense at all, and the list shrank twice while
# being written. "put" is absent because "put on some music" is the commonest
# media request there is; "give" and "show" because "give me the time" is
# ordinary; "take" because of "take a note". "lift" went because "the lift" is
# a noun and "remind me to hold the lift" was being refused for it. "push" and
# "pull" went because "push the button" is how somebody drives a page and "pull
# up the map" is an ordinary request. "catch" went because "did you catch that"
# is about hearing. "hold" survives, with "hold on", "hold that thought" and
# "hold tight" excluded as the discourse markers they are.
#
# The negative lookbehind is load-bearing. Without it, "Juno, remind me to grab
# the parcel" scored 0.11 and was REJECTED -- a perfectly ordinary thing to ask,
# refused because of a verb in the thing being remembered rather than in the
# request. After "to" the physical act is the CONTENT of the instruction, not
# what is being asked of the listener.
_PHYSICAL_REQUEST = re.compile(
    r"(?<!\bto\s)\b(pass|hand|grab|carry|fetch|"
    r"hold(?!\s+(?:on|that\s+thought|tight|fire|still|steady|your\s+horses)\b)|"
    r"pick\s+(?:it|that|this|them|those)\s+up|"
    r"put\s+(?:it|that|this|them)\s+(?:down|back|over|there|here|away)|"
    r"throw|shut\s+the\s+(?:door|window|curtains)|"
    r"open\s+the\s+(?:door|window|curtains|fridge|box|bottle|jar))\b",
    re.IGNORECASE,
)

_QUESTION_LEAD = re.compile(
    r"^(what|whats|what's|who|whos|who's|when|where|why|how|which|is|are|am|"
    r"can|could|would|will|should|do|does|did|has|have|tell|explain)\b"
)
_REQUEST_VERB = re.compile(
    r"^(set|start|stop|cancel|pause|resume|remind|tell me|give me|show me|"
    r"look up|search|google|find out|calculate|compute|convert|translate|"
    r"define|spell|note|remember|add|time me)\b"
)
_TOOL_DOMAIN = re.compile(
    r"\b(timer|alarm|countdown|stopwatch|weather|forecast|temperature|"
    r"raining|humidity|what time|the time|the date|today's date|calculate|"
    r"plus|minus|times|divided by|percent of|square root|convert|"
    r"celsius|fahrenheit|kilometers|miles|kilograms|pounds)\b"
)
_KNOWLEDGE_SEEKING = re.compile(
    r"\b(capital of|population of|who is|who was|who's a|who's the|"
    r"whos a|whos the|what is|what's the|"
    r"how many|how much|how far|how long|how old|difference between|"
    r"meaning of|definition of|when did|when was|where is)\b"
)
_BACKCHANNEL = re.compile(
    r"^(yeah|yep|yup|nope|nah|no|ok|okay|k|right|sure|mhm|mm|hmm|huh|uh huh|"
    r"cool|nice|wow|oh|ah|aw|haha|hah|lol|exactly|totally|true|fair|"
    r"i know|i see|got it|for sure|no way|oh my god|jesus|damn)"
    r"[\s,.!?]*$"
)
_HUMAN_VOCATIVE = re.compile(
    r"\b(dude|bro|man|mate|guys|y'all|yall|folks|honey|babe|sweetie|"
    r"mom|mum|dad|sis|bruh)\b"
)
_SOCIAL_TO_HUMAN = re.compile(
    r"\b(how are you|how've you been|how was your|how'd it go|did you see|"
    r"did you hear|are you free|you free|wanna grab|want to grab|see you|"
    r"talk to you later|catch you later|good to see you|nice to meet you|"
    r"congrats|congratulations|happy birthday|thank you so much)\b"
)
_JOINT_PLANNING = re.compile(
    r"\b(let's|lets |shall we|should we|we should|do you wanna|do you want to|"
    r"you wanna|are we|we could|we'll|our )\b"
)
_REPORTED_SPEECH = re.compile(
    # Third-person reporting of a past conversation. Deliberately excludes
    # "I said" and "I asked", which are also how someone repeats themselves
    # to an assistant that did not hear them the first time.
    r"\b(he|she|they) (said|told|asked|replied|mentioned|goes|was like)\b"
    r"|\bi told (him|her|them)\b"
    r"|\b(apparently|supposedly|reckons)\b"
)
_ABOUT_THE_AI = re.compile(
    r"\b(the ai|my assistant|the assistant|this thing|the bot|it said|"
    r"it told me|i asked it|it thinks|the model)\b"
)
_SELF_TALK = re.compile(
    r"^(where did i|where's my|where is my|i need to|i should|i forgot|"
    r"note to self|god i|ugh|oh no|oh shoot|oh crap)\b"
)
_POLITE_FRAME = re.compile(r"\b(please|could you|can you|would you mind)\b")
# Reference back to something already said: "make it ten", "the second one",
# "what about those?". These carry no topic words at all -- the antecedent is
# in the assistant's previous answer, not in the utterance -- so lexical
# overlap cannot see them.
_ANAPHORA = re.compile(
    r"\b(it|its|that one|this one|those|these|them|the (?:first|second|third|"
    r"last|other|next|cheaper|bigger|smaller) one|the other|the same)\b"
)
_HUMAN_LEAD = re.compile(
    r"^(yeah|yep|yup|nah|nope|oh|ah|aw|haha|hah|lol|well|i mean|like|honestly|"
    r"actually|wait|dude|man|bro|okay so|ok so|anyway|besides)\b"
)

# Words too common to indicate anything about topic.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here it its is
are was were be been being am do does did doing have has had having will
would shall should can could may might must i you he she we they me him her
us them my your his their our of in on at to for with from by about as into
like through after over between out against during without before under
around among not no yes so very just really quite too also only even still
some any all both each few more most other another such own same what which
who whom when where why how one two three thing things get got go going
know think say said tell told good bad well now new old
""".split())

# A trailing "s" is a plural after most letters, but part of the word after
# these -- without the guard, "famous" becomes "famou" and "Paris" becomes
# "Pari", which then collide with nothing and match things they should not.
_NOT_PLURAL_ENDINGS = ("ss", "us", "is", "os", "as")


def _stem(word: str) -> str:
    """Crude stem. It only has to make "name", "names" and "named" collide."""
    if word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif (
        word.endswith("s")
        and len(word) > 3
        and not word.endswith(_NOT_PLURAL_ENDINGS)
    ):
        word = word[:-1]
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def content_words(text: str) -> set[str]:
    """Topic-bearing words, lightly stemmed.

    Contractions are reduced to their head before anything else: "I'm" is the
    pronoun "I" and carries no topic at all, but left whole it survives the
    stopword list and reads as a content word shared between any two
    first-person sentences.
    """
    words = set()
    for raw in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()):
        head = raw.split("'")[0]
        # "don't" -> "don" -> "do"; "isn't" -> "isn" -> "is".
        if "'" in raw and raw.split("'")[1].startswith("t") and head.endswith("n"):
            head = head[:-1]
        if len(head) < 3 or head in _STOPWORDS:
            continue
        stemmed = _stem(head)
        if stemmed and stemmed not in _STOPWORDS:
            words.add(stemmed)
    return words


def build_name_pattern(name: str, aliases: Sequence[str] = ()) -> "re.Pattern | None":
    """Match the assistant's name across transcription variants.

    A speech recogniser has no idea how you spell your assistant's name, and
    will happily write "co-pilot", "Co Pilot" or "copilot" for the same sound.
    An exact match on one spelling silently fails on the others -- which is the
    worst possible failure for the one signal meant to be unambiguous.

    Separators are therefore optional between every character, and any number
    of extra spellings can be supplied as aliases for whatever your recogniser
    actually produces.
    """
    forms = [n for n in [name, *aliases] if n and n.strip()]
    if not forms:
        return None
    patterns = []
    for form in forms:
        letters = [c for c in form.lower() if c.isalnum()]
        if not letters:
            continue
        patterns.append(r"[\s\-\.']*".join(re.escape(c) for c in letters))
    if not patterns:
        return None
    return re.compile(r"\b(?:" + "|".join(patterns) + r")\b")


# Words that may precede a name while still addressing someone.
_GREETING = r"(?:hey|hi|hello|ok|okay|yo|um|uh|so|excuse me|sorry)"
_VOCATIVE_LEAD = re.compile(rf"^\s*(?:{_GREETING}[\s,]+)*$")


def is_vocative(body: str, match: "re.Match") -> bool:
    """Is the name being used to address someone, rather than mentioned?

    Three positions count as addressing: opening the utterance (optionally
    after a greeting), closing it, or set off by a comma. Everything else is
    the name appearing as an ordinary noun.
    """
    before = body[: match.start()]
    after = body[match.end() :]
    if not _VOCATIVE_LEAD.match(before):
        # Something substantive precedes it -- unless a comma sets it off,
        # as in "so, what do you think, Copilot?"
        if not before.rstrip().endswith(","):
            return False
    # Addressing is followed by a pause, a question, or nothing at all.
    return after.strip() == "" or after[:1] in ",.?!" or after[:1].isspace()


@dataclass
class IntentSignal:
    """One piece of evidence, with the text that triggered it."""

    name: str
    weight: float
    detail: str | None = None

    def as_json(self) -> dict:
        out = {"signal": self.name, "w": round(self.weight, 3)}
        if self.detail:
            out["match"] = self.detail
        return out


@dataclass
class IntentDecision:
    ai_intent: bool
    confidence: float
    band: str  # accept | ambiguous | reject
    method: str  # heuristic | llm | heuristic_default
    signals: list[IntentSignal] = field(default_factory=list)
    heuristic_confidence: float = 0.0
    llm_confidence: float | None = None
    latency: float = 0.0
    # The clause the score came from, when it was not the whole utterance.
    scored_clause: str | None = None
    # Why the adjudicator produced nothing, when it was asked and did not
    # answer. "Rate limited" and "decided it could not tell" both arrive as a
    # missing score, and reading a measurement without separating them means
    # reporting the network as if it were the model's judgement.
    llm_error: str | None = None

    def explain(self) -> str:
        parts = sorted(self.signals, key=lambda s: -abs(s.weight))
        rendered = ", ".join(
            f"{s.name}{'+' if s.weight >= 0 else ''}{s.weight:.2f}" for s in parts[:6]
        )
        return f"p={self.confidence:.2f} [{self.band}/{self.method}] {rendered}"

    def as_json(self) -> dict[str, Any]:
        return {
            "ai_intent": self.ai_intent,
            "confidence": round(self.confidence, 4),
            "band": self.band,
            "method": self.method,
            "heuristic": round(self.heuristic_confidence, 4),
            "llm": None if self.llm_confidence is None else round(self.llm_confidence, 4),
            "signals": [s.as_json() for s in self.signals],
        }


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class HeuristicScorer:
    """Fast, explainable, zero-dependency first pass."""

    def __init__(self, config, assistant_name: str = "") -> None:
        self.followup_window = float(config.get("followup_window", 20.0))
        # Log-odds, so 1.5 is about four and a half to one. Enough to carry a
        # plain imperative from the ambiguity band into accept, and not enough
        # to carry something the engine was rejecting outright -- "go back" on
        # its own stays ambiguous and goes to the adjudicator, which is the
        # right answer for two words that mean something to a person too.
        self.driving_bonus = float(config.get("driving_bonus", 1.5))
        # Enough to take a polite physical request out of accept, and not so
        # much that a sentence which merely mentions one is thrown away: it is
        # evidence about the audience, not a veto.
        self.physical_penalty = float(config.get("physical_penalty", -1.6))
        self.followup_bonus = float(config.get("followup_bonus", 0.3))
        self.topical_window = float(config.get("topical_window", 45.0))
        self.topical_bonus = float(config.get("topical_bonus", 1.3))
        self.bystanders = [
            n.lower() for n in (config.get("bystander_names") or []) if n
        ]
        self._name_re = build_name_pattern(
            assistant_name, config.get("assistant_aliases") or []
        )

    def score(
        self,
        text: str,
        context: ConversationContext,
        reliable: bool = True,
        spoken_at: float | None = None,
        own_voice: float | None = None,
        driving: bool = False,
    ) -> tuple[float, list[IntentSignal]]:
        body = _clean(text)
        words = body.split()
        signals: list[IntentSignal] = [IntentSignal("bias", BIAS)]

        def add(name: str, weight: float, match: re.Match | None = None) -> None:
            signals.append(
                IntentSignal(name, weight, match.group(0) if match else None)
            )

        # ---- addressing -------------------------------------------------
        if self._name_re and (m := self._name_re.search(body)):
            # Where the name sits matters more than that it appears. "Copilot,
            # what's the time?" is addressing it; "the co-pilot of the plane"
            # is talking about something else entirely. Treating any mention
            # as certainty makes an assistant that interrupts whenever its
            # name comes up in conversation -- and any name short enough to
            # say is a word someone will eventually say.
            if is_vocative(body, m):
                add("assistant_name", 2.4, m)
            else:
                # Weak on purpose, and weakened further once the recogniser
                # was told to expect the name. Biasing for "Juno" took recall
                # of it in noise from 4/10 to 9/10 and also turned "is June a
                # good month" into "is Juno a good month" twice in twenty --
                # at 0.9 that crossed the accept threshold on its own, which
                # is a false activation bought with a missed one. At 0.55 it
                # lands in the band and the adjudicator reads the sentence.
                #
                # Nothing in eval/intent_cases.yaml exercises a non-vocative
                # mention, so the case set is silent on this: it scores the
                # same at every value from 0.45 to 0.9. The number is set by
                # the property above, not by the eval -- it has to leave a
                # bare mention below accept_threshold.
                #
                # 0.45, not 0.55, because accept_threshold moved to 0.60. At
                # 0.55 "do you know if the copilot is any good?" scores 0.611
                # and answers on its own, which is exactly the failure this
                # weight exists to prevent. 0.45 puts it at 0.587 -- about
                # the same margin under the new threshold as 0.55 had under
                # the old one. Move them together or not at all.
                add("assistant_name_mentioned", 0.45, m)

        for bystander in self.bystanders:
            if re.search(rf"\b{re.escape(bystander)}\b", body):
                add("bystander_addressed", -2.2, None)
                signals[-1].detail = bystander
                break

        if m := _HUMAN_VOCATIVE.search(body):
            add("human_vocative", -1.1, m)

        # ---- request shape ----------------------------------------------
        if m := _REQUEST_VERB.search(body):
            add("request_verb", 1.70, m)
        if m := _QUESTION_LEAD.search(body):
            add("question_lead", 0.60, m)
        if body.endswith("?"):
            add("question_mark", 0.35)
        if m := _KNOWLEDGE_SEEKING.search(body):
            add("knowledge_seeking", 0.80, m)
        if m := _TOOL_DOMAIN.search(body):
            add("tool_domain", 0.90, m)
        if m := _POLITE_FRAME.search(body):
            add("polite_request", 0.30, m)

        # ---- talking to a person ----------------------------------------
        if m := _HUMAN_LEAD.match(body):
            # A whole-utterance backchannel is handled below; this catches the
            # opener that introduces a remark to a person, which is what a
            # comment about the assistant's answer usually sounds like.
            add("human_discourse_lead", -0.65, m)
        if m := _BACKCHANNEL.match(body):
            # "yeah", "cool", "haha" -- conversational glue, never a request.
            add("backchannel", -2.0, m)
        if m := _SOCIAL_TO_HUMAN.search(body):
            add("social_register", -1.3, m)
        if m := _JOINT_PLANNING.search(body):
            add("joint_planning", -0.9, m)
        if m := _REPORTED_SPEECH.search(body):
            add("reported_speech", -1.0, m)
        if m := _ABOUT_THE_AI.search(body):
            # Talking *about* the assistant, not to it.
            add("third_person_about_ai", -0.8, m)
        if m := _SELF_TALK.search(body):
            add("self_talk", -0.6, m)

        # ---- who was speaking -------------------------------------------
        # Acoustic evidence that this did not come from the person wearing the
        # microphone. Strictly one-sided: it can argue against answering and
        # never for it. Measured on the 87-case eval, a positive prior of the
        # size a perfect detector would supply clears all six missed
        # activations and creates five false ones doing it -- because
        # p(the wearer spoke) says nothing about p(the wearer meant *us*).
        # The wearer is also the one having a conversation with someone else.
        #
        # `presence` scales the momentum signals below rather than gating them
        # outright, so degraded evidence degrades them gently. With no
        # estimate at all it is 1.0 and everything here behaves as it did.
        presence = 1.0
        if own_voice is not None:
            odds = min(max(own_voice, 1e-3), 1.0 - 1e-3)
            weight = max(-3.0, min(0.0, math.log(odds / (1.0 - odds))))
            if weight < 0.0:
                add("not_own_voice", round(weight, 3))
                signals[-1].detail = f"p_own={own_voice:.2f}"
            presence = min(1.0, own_voice / 0.5)

        # ---- asked for something it has no body to do --------------------
        if (m := _PHYSICAL_REQUEST.search(body)):
            add("physical_request", self.physical_penalty, m)

        # ---- driving something on the screen ----------------------------
        # Only when a browser the wearer opened is in front of them. "Click the
        # first one", "scroll down", "go back" are ordinary furniture in a
        # conversation between people and unambiguous while somebody is driving
        # a page -- so the evidence is the SITUATION, not the words, and the
        # words are consulted only inside it. With no browser open this
        # contributes nothing at all and the engine behaves exactly as it did.
        if (driving and not _addressed_to_a_person(body)
                and (m := _DRIVING_COMMAND.match(body))):
            add("driving_browser", self.driving_bonus, m)

        # ---- conversational timing --------------------------------------
        # Measured from when the speaker *started* talking, not from now.
        # Classification happens after the utterance ends and after
        # transcription, so timing it from "now" makes a long sentence decay
        # its own follow-up bonus -- the longer you speak, the less the system
        # believes you were continuing the conversation.
        since = context.seconds_since_ai_response(at=spoken_at)
        if since < self.followup_window:
            # Quadratic: stays near full for the first few seconds, where a
            # continuation is genuinely likely, then falls away quickly.
            decay = 1.0 - (since / self.followup_window) ** 2
            # Scaled by `presence`: "the assistant spoke recently" is only
            # evidence if the voice now speaking is the one it was speaking
            # to. Measured over 213 real accepted turns in logs/, this signal
            # was the decisive one on 126 of them -- 59% -- which means the
            # commonest reason this assistant answered was not that it was
            # addressed, but that it had recently talked. That is exactly the
            # "it answers when I'm chatting to someone else" complaint.
            if (bonus := self.followup_bonus * decay * 2.0 * presence) > 0.0:
                add("followup_window", bonus)
            if context.ai_awaiting_answer():
                # The assistant asked something; the next thing said answers
                # it -- if the same person is answering. `ai_awaiting_answer`
                # is only a test that our last line ended in a question mark,
                # so unscaled this hands +1.2 to whoever in the room happens
                # to speak next, at the moment they are most likely to react
                # to what the device just said out loud.
                if (bonus := 1.2 * presence) > 0.0:
                    add("answering_ai_question", bonus)

        # Still talking about what the assistant just said. Shared topic plus
        # recency is the strongest evidence available for a follow-up that
        # carries none of the usual markers -- no name, no request verb, often
        # not even a question.
        overlap = self._topical_overlap(text, context, since)
        if overlap:
            shared, weight = overlap
            add("topical_continuity", weight)
            signals[-1].detail = ", ".join(sorted(shared)[:3])

        # Referring back to something with no antecedent in this sentence.
        # Only meaningful while the previous exchange is still live: "make it
        # ten instead" is a follow-up seconds after a timer was set, and
        # nothing at all an hour later.
        if since < self.followup_window and (m := _ANAPHORA.search(body)):
            add("anaphoric_reference", 0.6, m)

        # Conversations have momentum: if the last thing said nearby was aimed
        # at a person, the next thing probably is too.
        if self._in_human_exchange(context):
            add("human_exchange_in_progress", -0.45)

        # ---- shape and length -------------------------------------------
        # Length is weak evidence and is evaluated last, because it only
        # argues against intent when nothing stronger argues for it. Without
        # this guard, "hey Copilot" and "define perfunctory" both score as
        # not-for-the-assistant purely for being short.
        fired = {s.name for s in signals}
        strong_positive = fired & {
            "assistant_name",
            "assistant_name_mentioned",
            "request_verb",
            "answering_ai_question",
            "tool_domain",
            "knowledge_seeking",
        }
        if len(words) <= 2 and not body.endswith("?") and not strong_positive:
            add("very_short_fragment", -0.9)
        elif len(words) > 28:
            # Long, flowing speech is narrative, not a command to a small model.
            add("long_utterance", -0.5)

        # A verbatim repeat of something we ignored usually means the user is
        # trying again, louder.
        if repeat := self._repeat_of_ignored(body, context):
            add("retry_of_ignored", 0.9, None)
            signals[-1].detail = repeat

        # ---- transcript quality -----------------------------------------
        if not reliable:
            add("unreliable_transcript", -1.4)

        logit = sum(s.weight for s in signals)
        return _sigmoid(logit), signals

    def _topical_overlap(
        self, text: str, context: ConversationContext, since: float
    ) -> tuple[set[str], float] | None:
        """Content words shared with what the assistant just said."""
        if since > self.topical_window:
            return None
        previous = context.last_ai_text
        if not previous:
            return None
        shared = content_words(text) & content_words(previous)
        if not shared:
            return None
        # One shared word is weak evidence, three is strong; more than that
        # adds nothing and would let a long answer dominate the score.
        weight = self.topical_bonus * min(len(shared), 3) / 3.0
        # Recency multiplies it: the same words minutes later mean much less.
        weight *= 1.0 - 0.5 * (since / self.topical_window)
        return shared, round(weight, 3)

    @staticmethod
    def _in_human_exchange(context: ConversationContext) -> bool:
        """Was the last thing heard, recently, aimed at a person?"""
        for previous in reversed(context.recent_utterances(2)):
            if previous.age > 25.0:
                continue
            return previous.ai_directed is False
        return False

    @staticmethod
    def _repeat_of_ignored(body: str, context: ConversationContext) -> str | None:
        for previous in reversed(context.recent_utterances(4)):
            if previous.ai_directed or previous.age > 20.0:
                continue
            if SequenceMatcher(None, body, _clean(previous.text)).ratio() > 0.82:
                return previous.text[:60]
        return None


ADJUDICATOR_SYSTEM = """\
You judge who a spoken sentence was addressed to.

The speaker is wearing a voice assistant called {name}. They may be speaking
to it, or to people around them.

{setting}

Being a question is not by itself a reason to think it was for {name}. People
ask each other questions constantly.

Reasons it was probably for {name}: it asks for a fact nobody present would
be expected to know; it asks for something only a device can do (a timer, a
calculation, opening an app); it follows on from something {name} just said.

Reasons it was probably for a person: it carries on a conversation with
somebody else that is already under way; it is about a shared plan; it asks the listener for something
physical; it is about somebody's own life or opinion; it refers to a person
by "he", "she" or "they" as the subject of the discussion.

"{name}" may also be an ordinary word or somebody's name. Hearing it in the
middle of a sentence is not the same as being called by it.

Answer with a single digit 0-9. 0 means certainly to a person, 9 means
certainly to {name}, and the middle of the range means you are unsure -- use
it, rather than picking a side.

{tiebreak}

Output the digit and nothing else.\
"""

# The room the adjudicator is asked to imagine used to be fixed: "is also
# talking with people around them, almost everything they say is to a person".
# When that is true it is the right prior. When the speaker is alone with the
# assistant it is exactly backwards, and it is the strongest thing in the
# prompt -- in logs/ it answered 0, "certainly to a person", to four sentences
# said to it by someone alone in their bedroom, including a repeat of one it
# had already missed.
#
# So the setting is now read from the record rather than assumed. False in
# that record means audience evidence was actually seen, not merely that we
# declined to answer -- see IntentEngine.record.
_SETTING_OTHERS = """\
There is a conversation with other people going on around them. Almost
everything they say is to a person, not to {name}: assume the sentence was
said to a person unless there is a reason to think otherwise.\
"""

_SETTING_SOLO = """\
Nobody else has been heard speaking. The recent exchanges below are with
{name}, so a sentence that carries one of them on is probably also for {name}
-- including a correction, a complaint about the answer, or a follow-up that
names nothing.\
"""

_SETTING_UNKNOWN = """\
It is not known whether anybody else is present. Judge from the exchanges
below and from the sentence itself.\
"""

_TIEBREAK_CAUTIOUS = """\
When you cannot tell, answer 4 or lower: speaking into a conversation it was
not part of costs {name} more than staying quiet does.\
"""

_TIEBREAK_BALANCED = """\
When you genuinely cannot tell, answer 4 or 5. Do not answer low merely
because the sentence is conversational -- with nobody else present, that is
how someone talks to an assistant they are already in the middle of using.\
"""


class LLMAdjudicator:
    """Second opinion from the local model, for the ambiguity band only.

    Constrained to emit a single digit, so this costs one forward pass plus
    one token rather than a full generation.
    """

    def __init__(self, model, timeout: float = 1.5, assistant_name: str = "") -> None:
        self.model = model
        self.timeout = timeout
        self.assistant_name = assistant_name
        # Why the last call produced nothing. Read by the evaluator, and worth
        # having in the log: an adjudicator that is quietly failing looks
        # exactly like one that is quietly abstaining.
        self.last_error: str | None = None

    def _setting(self, context: ConversationContext) -> tuple[str, str]:
        """Describe the room from the record instead of assuming one.

        Returns the setting paragraph and the matching tie-break rule. The
        cautious tie-break is kept wherever there is any evidence of other
        people, because that is the case it was written for and the case where
        being wrong is worst.
        """
        recent = context.recent_utterances(6)
        others = sum(1 for u in recent if u.ai_directed is False)
        ours = sum(1 for u in recent if u.ai_directed is True)
        if others:
            return _SETTING_OTHERS, _TIEBREAK_CAUTIOUS
        if ours >= 2:
            return _SETTING_SOLO, _TIEBREAK_BALANCED
        return _SETTING_UNKNOWN, _TIEBREAK_CAUTIOUS

    def _prompt(self, text: str, context: ConversationContext) -> list[dict]:
        lines = []
        for utterance in context.recent_utterances(3):
            if utterance.text.strip() == text.strip():
                continue
            # Three states, three tags. Reporting "to someone else" for an
            # utterance we merely declined to answer tells the adjudicator a
            # human exchange is under way on no evidence at all, which is the
            # loop that made misses cluster.
            if utterance.ai_directed is True:
                tag = "to the assistant"
            elif utterance.ai_directed is False:
                tag = "to someone else"
            else:
                tag = "unclear who to"
            lines.append(f'- earlier, {tag}: "{utterance.text.strip()}"')
        if context.last_ai_text:
            since = context.seconds_since_ai_response()
            if since < 60:
                lines.append(
                    f'- the assistant said {since:.0f}s ago: "{context.last_ai_text.strip()}"'
                )
        history = "\n".join(lines) if lines else "- (no recent context)"
        name = self.assistant_name or "the assistant"
        setting, tiebreak = self._setting(context)
        user = (
            f"Context:\n{history}\n\n"
            f'Sentence just spoken: "{text.strip()}"\n\n'
            "Digit:"
        )
        return [
            {
                "role": "system",
                "content": ADJUDICATOR_SYSTEM.format(
                    name=name, setting=setting.format(name=name),
                    tiebreak=tiebreak.format(name=name),
                ),
            },
            {"role": "user", "content": user},
        ]

    def score(self, text: str, context: ConversationContext) -> float | None:
        self.last_error = None
        try:
            raw = self.model.complete(
                self._prompt(text, context),
                # Four tokens is enough for the digit but not for a model that
                # reasons before answering: it spends the budget thinking and
                # returns nothing, so the adjudicator silently never works and
                # every ambiguous utterance is rejected by default. 24 left
                # one case in twenty empty and 64 still left two; the ceiling
                # is not what it costs, only what it is allowed.
                max_tokens=192,
                temperature=0.0,
                timeout=self.timeout,
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:120]
            return None
        match = re.search(r"[0-9]", raw or "")
        if not match:
            self.last_error = f"no digit in {(raw or '')[:40]!r}"
            return None
        return int(match.group(0)) / 9.0


class IntentEngine:
    """Heuristic gate, with LLM adjudication inside the ambiguity band."""

    def __init__(self, config, context: ConversationContext, model=None, observer=None,
                 assistant_name: str = "") -> None:
        self.config = config
        self.context = context
        self.observer = observer
        self.accept_threshold = float(config.get("accept_threshold", 0.62))
        self.reject_threshold = float(config.get("reject_threshold", 0.34))
        self.scorer = HeuristicScorer(config, assistant_name=assistant_name)
        self.adjudicator = (
            LLMAdjudicator(
                model,
                timeout=float(config.get("llm_adjudicator_timeout", 1.5)),
                assistant_name=assistant_name,
            )
            if (model is not None and config.get("llm_adjudicator", True))
            else None
        )
        # How much the adjudicator is trusted relative to the heuristic.
        self.llm_weight = float(config.get("llm_weight", 0.65))
        # Below this, acoustic evidence overrides even hearing the name.
        self.own_voice_veto = float(config.get("own_voice_veto", 0.35))

    def classify(
        self,
        text: str,
        reliable: bool = True,
        spoken_at: float | None = None,
        tail_reliable: bool | None = None,
        on_ambiguous=None,
        own_voice: float | None = None,
        driving: bool = False,
    ) -> IntentDecision:
        """Decide whether this speech was addressed to the assistant.

        ``on_ambiguous`` is called once, immediately before the adjudicator
        runs, for anything that wants to use that time -- the adjudicator is
        the slowest thing in the classifier and nothing else depends on it.
        Whatever it starts must be safe to throw away, because the answer may
        well be that this was not our conversation.
        """
        started = time.monotonic()
        heuristic_p, signals = self.scorer.score(
            text, self.context, reliable=reliable, spoken_at=spoken_at,
            own_voice=own_voice,
            driving=driving,
        )
        clause = None

        # One captured segment is not one utterance. Talking to a person and
        # then turning to the assistant lands in a single blob, and scoring
        # the blob buries the request under the conversation around it. When
        # the address comes, it comes at the end, so the trailing clause is
        # scored on its own and the better of the two is used.
        #
        # The contextual signals -- an exchange with a person in progress, a
        # bystander named, how long ago the assistant spoke -- are unchanged
        # by this, because they come from the conversation rather than from
        # the words. Only the linguistic evidence is re-read.
        tail_p, tail_signals, tail_text = self._score_trailing_clause(
            text,
            reliable if tail_reliable is None else tail_reliable,
            spoken_at,
            signals,
            own_voice=own_voice,
        )
        if tail_p is not None and tail_p > heuristic_p:
            heuristic_p, signals, clause = tail_p, tail_signals, tail_text

        # Saying the name settles it. This is not a wake word -- the assistant
        # works perfectly well without it -- but when you want certainty
        # rather than inference, there has to be a way to say so, and it must
        # not cost a round trip to the adjudicator to confirm.
        # ...but not if the acoustics say this was not the wearer speaking.
        # The recogniser is deliberately biased towards hearing the name
        # (config: stt.bias_vocabulary), and that bias is measured to produce
        # it falsely 2 times in 20 at 5 dB SNR. Without this veto a misheard
        # name in somebody else's sentence returns confidence 1.0 before any
        # band logic runs, and nothing downstream can argue with it.
        named = any(s.name == "assistant_name" for s in signals)
        if named and own_voice is not None and own_voice < self.own_voice_veto:
            # Unreachable as the pipeline now stands, and kept deliberately.
            # juno_core/pipeline.py drops a segment before transcription
            # whenever the voiceprint says it was not the wearer, and
            # p_own >= 0.5 exactly when it says it was -- so nothing below
            # own_voice_veto (0.35)
            # can arrive here. It stays because the drop is the thing that
            # might change: relax that gate and this becomes the only thing
            # standing between a misheard name in somebody else's sentence
            # and confidence 1.0. There is no cost to keeping it.
            named = False
        if named:
            return IntentDecision(
                ai_intent=True,
                confidence=1.0,
                band="accept",
                method="named",
                signals=signals,
                heuristic_confidence=heuristic_p,
                latency=time.monotonic() - started,
                scored_clause=clause,
            )

        # Saying it again, after we ignored it, settles it too.
        #
        # Repeating yourself is what a person does when they have been missed,
        # and it is the one correction available without a wake word. Measured
        # over the 15 turns in logs/ where this signal fired, letting it decide
        # recovers three real misses -- "Open a tab on Safari about YouTube",
        # "Is it working, Copilot?", "So this is a song called Manifest
        # Destiny" -- and the three it would wrongly admit are all Whisper
        # looping on noise, which `reliable` now excludes.
        #
        # Guarded three ways: the transcript must be trustworthy, the acoustics
        # must not say somebody else said it, and _repeat_of_ignored already
        # demands 0.82 similarity to something ignored within 20 seconds.
        retry = any(s.name == "retry_of_ignored" for s in signals)
        if retry and not reliable:
            retry = False
        if retry and own_voice is not None and own_voice < self.own_voice_veto:
            # Unreachable for the same reason as the veto above.
            retry = False
        if retry:
            return IntentDecision(
                ai_intent=True,
                confidence=1.0,
                band="accept",
                method="retry",
                signals=signals,
                heuristic_confidence=heuristic_p,
                latency=time.monotonic() - started,
                scored_clause=clause,
            )

        llm_error = None
        if heuristic_p >= self.accept_threshold:
            band, method, confidence, llm_p = "accept", "heuristic", heuristic_p, None
        elif heuristic_p < self.reject_threshold:
            band, method, confidence, llm_p = "reject", "heuristic", heuristic_p, None
        else:
            band = "ambiguous"
            if on_ambiguous is not None:
                on_ambiguous()
            llm_p = self.adjudicator.score(text, self.context) if self.adjudicator else None
            if llm_p is None:
                # No second opinion available: stay quiet rather than guess.
                method, confidence = "heuristic_default", heuristic_p
                llm_error = getattr(self.adjudicator, "last_error", None)
            else:
                method = "llm"
                confidence = (
                    self.llm_weight * llm_p + (1.0 - self.llm_weight) * heuristic_p
                )

        decision = IntentDecision(
            ai_intent=confidence >= self.accept_threshold,
            confidence=confidence,
            band=band,
            method=method,
            signals=signals,
            heuristic_confidence=heuristic_p,
            llm_confidence=llm_p,
            latency=time.monotonic() - started,
            scored_clause=clause,
            llm_error=llm_error,
        )
        return decision

    def _score_trailing_clause(self, text: str, reliable: bool, spoken_at,
                               whole_signals: list[IntentSignal],
                               own_voice: float | None = None):
        """Score the last sentence alone, if there is more than one.

        What is being *asked* is in the trailing clause. Who is being *spoken
        to* is established by the whole utterance, so audience evidence from
        the earlier part is carried across -- otherwise "we should go to Paris.
        Who's a famous actor from there?" scores exactly like the same question
        asked cold, and the planning that made it a remark to a person counts
        for nothing.
        """
        sentences = split_sentences(text)
        if len(sentences) < 2:
            return None, None, None
        tail = sentences[-1].strip()
        # A one-word tail carries no evidence worth re-reading, and scoring it
        # alone only strips away context that was doing useful work.
        if len(tail.split()) < 2:
            return None, None, None
        score, signals = self.scorer.score(
            tail, self.context, reliable=reliable, spoken_at=spoken_at,
            own_voice=own_voice,
        )
        already = {s.name for s in signals}
        carried = [
            s
            for s in whole_signals
            if s.name in _AUDIENCE_SIGNALS and s.name not in already and s.weight < 0
        ]
        if carried:
            signals = signals + carried
            score = _sigmoid(sum(s.weight for s in signals))
        return score, signals, tail

    def record(self, text: str, decision: IntentDecision, duration: float = 0.0) -> None:
        """Log the utterance into context so later decisions can use it.

        ``ai_directed`` is deliberately tri-state, and writing plain False for
        everything we declined to answer was a feedback loop. Every signal
        that reads this log -- `human_exchange_in_progress`, the adjudicator's
        context lines -- treats False as "that one was aimed at a person", so
        one wrong rejection became evidence that a human conversation was
        under way, which made the next rejection likelier. In logs/ that is
        visible as a spiral: an utterance ignored, the same thing said again
        and ignored again, and only the third attempt getting through by
        naming the assistant.

        So False now means we saw actual evidence of a human audience -- a
        bystander addressed, a vocative, joint planning, reported speech --
        and None means only that we chose not to act. Not knowing is not the
        same as knowing it was not for us.
        """
        audience = any(
            s.name in _AUDIENCE_SIGNALS and s.weight < 0 for s in decision.signals
        )
        self.context.add_utterance(
            Utterance(
                text=text,
                timestamp=time.monotonic(),
                ai_directed=True if decision.ai_intent else (False if audience else None),
                confidence=decision.confidence,
                duration=duration,
            )
        )
