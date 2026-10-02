"""Listen while acting.

The listener thread only captures speech and enqueues commands. One executor
thread plans and runs them in order. A speculator plans from the live
transcript while the user is still talking; the executor uses that plan only
if the final words match and nothing has run since it was made, so a plan
never describes an older screen.
"""
from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" .,!?;:")


@dataclass
class Command:
    text: str
    needs_gate: bool = False          # unnamed speech: Jev decides whether it was meant for us
    speculate: bool = True
    on_accept: Callable[[], None] | None = None
    queued_at: float = field(default_factory=time.monotonic)


class CommandQueue:
    def __init__(self) -> None:
        self._items: deque[Command] = deque()
        self._cv = threading.Condition()

    def put(self, cmd: Command) -> None:
        with self._cv:
            self._items.append(cmd)
            self._cv.notify()

    def get(self, timeout: float | None = None) -> Command | None:
        with self._cv:
            if not self._items:
                self._cv.wait(timeout)
            return self._items.popleft() if self._items else None

    def clear(self) -> int:
        with self._cv:
            n = len(self._items)
            self._items.clear()
            return n

    def pending(self) -> int:
        with self._cv:
            return len(self._items)


class Executor:
    """Runs queued commands one at a time. `run` returns False when the user asked to stop."""

    def __init__(self, queue: CommandQueue, run: Callable[[Command], bool], on_stop: Callable[[], None] | None = None) -> None:
        self.queue = queue
        self._run = run
        self.on_stop = on_stop
        self.generation = 0
        self.busy = False
        self.stopped = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="jev-exec", daemon=True)
            self._thread.start()

    def idle(self) -> bool:
        return self.snapshot()[0]

    def snapshot(self) -> tuple[bool, int]:
        """Whether nothing is running or waiting, and how many commands have run."""
        with self._lock:
            return not self.busy and self.queue.pending() == 0, self.generation

    def _loop(self) -> None:
        while not self.stopped.is_set():
            cmd = self.queue.get(timeout=0.2)
            if cmd is None:
                continue
            with self._lock:
                self.busy = True
            ok = True
            try:
                ok = self._run(cmd)
            except Exception as e:  # noqa: BLE001
                print(f"  ! {e}")
            finally:
                with self._lock:
                    self.generation += 1
                    self.busy = False
            if not ok:
                self.queue.clear()
                self.stopped.set()
                if self.on_stop:
                    self.on_stop()


@dataclass
class _Early:
    generation: int
    front: str | None
    plan: Any


class Speculator:
    """Plans from partial transcripts so the plan is ready when the phrase ends."""

    def __init__(self, evaluate: Callable[[str], Any], frontmost: Callable[[], str | None], executor: Executor,
                 max_per_phrase: int = 3, min_words: int = 3, keep: int = 8) -> None:
        self.evaluate = evaluate
        self.frontmost = frontmost
        self.executor = executor
        self.max_per_phrase = max_per_phrase
        self.min_words = min_words
        self.keep = keep
        self.prepare: Callable[[str], str | None] = lambda text: text
        self.requests = 0
        self._lock = threading.Lock()
        self._done = threading.Condition(self._lock)
        self._cache: OrderedDict[str, _Early] = OrderedDict()
        self._last = ""
        self._count = 0
        self._inflight: str | None = None
        self._next: tuple[str, str] | None = None

    def reset(self) -> None:
        """A new phrase started."""
        with self._lock:
            self._last = ""
            self._count = 0
            self._next = None

    def offer(self, partial: str) -> None:
        cmd = self.prepare(partial)
        if not cmd:
            return
        key = normalize(cmd)
        with self._lock:
            stable = key == self._last
            self._last = key
            if not key or not (stable or len(key.split()) >= self.min_words):
                return
            if key in self._cache or key == self._inflight or self._count >= self.max_per_phrase:
                return
            if self._inflight is not None:
                self._next = (key, cmd)
                return
            self._launch(key, cmd)

    def _launch(self, key: str, cmd: str) -> None:
        self._count += 1
        self._inflight = key
        threading.Thread(target=self._work, args=(key, cmd), name="jev-spec", daemon=True).start()

    def _work(self, key: str, cmd: str) -> None:
        early = None
        try:
            idle, generation = self.executor.snapshot()
            if idle:
                front = self.frontmost()
                self.requests += 1
                plan = self.evaluate(cmd)
                early = _Early(generation, front, plan)
        except Exception:
            early = None
        with self._lock:
            if early is not None:
                self._cache[key] = early
                while len(self._cache) > self.keep:
                    self._cache.popitem(last=False)
            self._inflight = None
            nxt, self._next = self._next, None
            if nxt and nxt[0] not in self._cache and self._count < self.max_per_phrase:
                self._launch(*nxt)
            self._done.notify_all()

    def take(self, cmd: str, wait: float = 3.0) -> Any:
        """The early plan for `cmd`, or None when the words differ or something ran since."""
        key = normalize(cmd)
        deadline = time.monotonic() + wait
        with self._lock:
            while self._inflight == key and time.monotonic() < deadline:
                self._done.wait(max(0.0, deadline - time.monotonic()))
            early = self._cache.pop(key, None)
        if early is None or early.generation != self.executor.generation:
            return None
        try:
            if self.frontmost() != early.front:
                return None
        except Exception:
            return None
        return early.plan


class StatusFeed:
    """Wraps the island. The live transcript owns it while the user speaks; action
    updates wait and show afterward, with a count of queued commands."""

    _VOICE = ("heard", "listening")

    def __init__(self, inner: Any, pending: Callable[[], int] = lambda: 0) -> None:
        self.inner = inner
        self.pending = pending
        self._speaking = False
        self._deferred: tuple[str, str, float | None] | None = None
        self._lock = threading.Lock()

    def run(self, worker: Callable[[], None]) -> None:
        self.inner.run(worker)

    def _badge(self, state: str, text: str) -> str:
        n = self.pending()
        return f"{text}  ·  +{n} queued" if n and state not in self._VOICE else text

    def set(self, state: str, text: str, revert_after: float | None = None) -> None:
        with self._lock:
            if self._speaking and state not in self._VOICE:
                self._deferred = (state, text, revert_after)
                return
        self.inner.set(state, self._badge(state, text), revert_after)

    def voice(self, on: bool) -> None:
        with self._lock:
            self._speaking = on
            deferred = None if on else self._deferred
            if not on:
                self._deferred = None
        if deferred:
            state, text, revert = deferred
            self.inner.set(state, self._badge(state, text), revert)
