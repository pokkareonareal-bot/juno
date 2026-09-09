"""Handing a slow job to something else and carrying on talking.

Juno answers on a very small model because almost everything said to it is
small. Some things are not, and today those block: the turn runs the stronger
model inline, the microphone is closed for the duration, and the assistant is
deaf until it finishes. Ask it something hard and you cannot ask it anything
else, which is the wrong trade for a thing you wear.

This is the seam that fixes it. An Executor is anything that can be handed a
Task and eventually produce an Outcome. What runs behind it is not this
module's business -- a stronger Gemini model on the same key, a self-hosted
Hermes Agent, somebody's own framework, a shell script.

WHY THE INTERFACE IS THIS NARROW
--------------------------------
Because the interesting half of Juno is upstream of it. Deciding whether the
person is talking to *you*, with no wake word, is the part that is hard and
the part worth publishing on its own; the tool loop and the Gemini wiring are
product. That boundary is only real if something enforces it, so Task and
Outcome are deliberately plain: text in, text out, plus a provider-neutral
history of the conversation so far. Nothing here imports a provider, mentions
a model name, or knows that audio exists.

An executor that needed more than this would be an executor that had reached
back through the seam.

WHAT AN EXECUTOR MAY ASSUME
---------------------------
That ``run`` is called on a worker thread, that it may block for a long time,
and that ``cancel`` may be set at any point -- on shutdown, or because the
user asked for it back. It should check ``cancel`` where it can and return
promptly when it is set. It must not raise: a failure is an Outcome with
``ok`` false and a sentence a person could hear, because that sentence is
going to be read out.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class Task:
    """A job, in terms an executor can act on without knowing what Juno is."""

    request: str
    # Conversation so far as [{"role": "user"|"assistant", "content": str}].
    # The shape ConversationContext.messages() already produces, which is also
    # the shape every provider takes, so neither end has to translate.
    history: list[dict[str, Any]] = field(default_factory=list)
    # Free-form, for an executor that wants more and a caller willing to give
    # it. Nothing in the core reads this; it exists so that needing one extra
    # field is not a reason to widen Task for everybody.
    extra: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.monotonic)

    @property
    def age(self) -> float:
        return time.monotonic() - self.created


@dataclass(frozen=True)
class Outcome:
    """What came back. `text` is spoken, so it is a sentence either way."""

    task_id: str
    text: str
    ok: bool = True
    seconds: float = 0.0
    executor: str = ""
    # True when this text came from somewhere untrustworthy -- a web page, a
    # search result -- rather than from the machine's own arithmetic. The
    # caller uses it to decide what the NEXT turn is allowed to do.
    external: bool = False

    @classmethod
    def failed(cls, task_id: str, text: str, executor: str = "",
               seconds: float = 0.0) -> "Outcome":
        return cls(task_id=task_id, text=text, ok=False,
                   seconds=seconds, executor=executor)


class Executor(Protocol):
    """Anything that can do slow work. Three members, all of them obvious."""

    name: str

    def run(self, task: Task, cancel: threading.Event) -> Outcome:
        """Do the work. Called on a worker thread. Must not raise."""
        ...


class CallableExecutor:
    """An Executor from a plain function, which is most of them.

    The function takes (task, cancel) and returns a string, or raises. Turning
    a raise into a spoken failure happens here so that every executor does not
    have to remember to.
    """

    def __init__(self, name: str, work: Callable[[Task, threading.Event], str]):
        self.name = name
        self._work = work

    def run(self, task: Task, cancel: threading.Event) -> Outcome:
        started = time.monotonic()
        try:
            text = self._work(task, cancel)
        except Exception as exc:
            return Outcome.failed(
                task.id,
                # Read aloud, eventually, so it says what happened rather than
                # naming a class. The detail goes to the event log instead.
                f"I could not finish that one. {type(exc).__name__}.",
                executor=self.name, seconds=time.monotonic() - started,
            )
        return Outcome(
            task_id=task.id, text=str(text or "").strip() or "It came back empty.",
            ok=True, seconds=time.monotonic() - started, executor=self.name,
            external=bool(getattr(self, "external", False)),
        )


# Set on the worker thread while a job runs. Anything that would normally ask
# the user a question needs to know it is running with nobody there.
_running = threading.local()


def unattended() -> bool:
    """True on a thread that is doing background work.

    Nobody is waiting on it and nobody has been asked anything, so a
    confirmation raised from here cannot be answered -- and must not pretend
    it might be.
    """
    return bool(getattr(_running, "flag", False))


class BackgroundWork:
    """One slow job at a time, running beside a conversation that continues.

    ONE, deliberately. Not a queue and not a pool: the result has to be said
    out loud to somebody who has been talking about other things since they
    asked, and two finished jobs competing to interrupt the same person is a
    worse experience than being told the first one is still going. If a second
    is asked for, that is a thing to say, not a thing to schedule.

    Nothing here touches the microphone, the intent engine or the speaker. The
    caller collects finished work when it is ready to say something, which is
    what keeps this off the audio thread entirely.
    """

    def __init__(self, observer=None) -> None:
        self._observer = observer
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._task: Task | None = None
        self._executor_name = ""
        self._done: list[Outcome] = []

    # -- state the caller asks about --------------------------------------

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._task is not None

    @property
    def current(self) -> Task | None:
        with self._lock:
            return self._task

    def waiting(self) -> list[Outcome]:
        """Finished work, handed over once and then forgotten."""
        with self._lock:
            done, self._done = self._done, []
        return done

    # -- doing it ----------------------------------------------------------

    def submit(self, task: Task, executor: Executor) -> bool:
        """Start the job, or refuse because one is already running."""
        with self._lock:
            if self._task is not None:
                return False
            self._task = task
            self._executor_name = getattr(executor, "name", "executor")
            self._cancel = threading.Event()
            cancel = self._cancel
        self._emit("work_started", task, executor=self._executor_name)
        self._thread = threading.Thread(
            target=self._run, args=(task, executor, cancel),
            name=f"work-{task.id}", daemon=True,
        )
        self._thread.start()
        return True

    def _run(self, task: Task, executor: Executor, cancel: threading.Event) -> None:
        started = time.monotonic()
        _running.flag = True
        try:
            outcome = executor.run(task, cancel)
        except Exception as exc:
            # An executor is told not to raise. This is here because "told not
            # to" is not a mechanism, and a background thread dying silently
            # would leave the caller waiting for an answer that never comes.
            outcome = Outcome.failed(
                task.id, f"I could not finish that one. {type(exc).__name__}.",
                executor=getattr(executor, "name", "executor"),
                seconds=time.monotonic() - started,
            )
        _running.flag = False
        with self._lock:
            self._task = None
            if not cancel.is_set():
                self._done.append(outcome)
        if cancel.is_set():
            self._emit("work_cancelled", task, seconds=round(outcome.seconds, 2))
        else:
            self._emit("work_finished", task, ok=outcome.ok,
                       seconds=round(outcome.seconds, 2),
                       executor=outcome.executor)

    def cancel(self) -> bool:
        """Ask the running job to stop. Its result is dropped, not spoken."""
        with self._lock:
            if self._task is None:
                return False
            self._cancel.set()
            return True

    def close(self, timeout: float = 2.0) -> None:
        self.cancel()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # Daemon, so a wedged executor cannot hold shutdown open past this.
            thread.join(timeout)

    def _emit(self, name: str, task: Task, **fields) -> None:
        if self._observer is None:
            return
        from juno_core.events import Stage

        # Not task=, not name=, not stage=: Observer.emit fills stage and name
        # positionally. See tests/test_observer_calls.py.
        self._observer.emit(Stage.LLM, name, None, task_id=task.id, **fields)
