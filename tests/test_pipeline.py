"""Listening while acting: queue, cancel, early plans. No microphone, no network."""

import threading
import time
from types import SimpleNamespace

import pytest

from jev_voice.audio import should_drop
from jev_voice.pipeline import Command, CommandQueue, Executor, Speculator, StatusFeed, normalize


def _wait(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.005)
    return cond()


def _executor(run):
    q = CommandQueue()
    ex = Executor(q, run)
    ex.start()
    return q, ex


def test_commands_spoken_during_an_action_run_afterward_in_order():
    release = threading.Event()
    ran: list[str] = []

    def run(cmd):
        ran.append(cmd.text)
        if cmd.text == "open notes":
            release.wait(2)
        return True

    q, ex = _executor(run)
    q.put(Command("open notes"))
    assert _wait(lambda: ex.busy)
    q.put(Command("write good morning"))
    q.put(Command("press enter"))
    assert ran == ["open notes"]
    release.set()
    assert _wait(lambda: ran == ["open notes", "write good morning", "press enter"])
    assert _wait(ex.idle)
    assert ex.generation == 3


def test_clear_drops_waiting_commands_but_not_the_running_one():
    release = threading.Event()
    ran: list[str] = []

    def run(cmd):
        ran.append(cmd.text)
        release.wait(2)
        return True

    q, ex = _executor(run)
    q.put(Command("first"))
    assert _wait(lambda: ex.busy)
    q.put(Command("second"))
    q.put(Command("third"))
    assert q.clear() == 2
    release.set()
    assert _wait(ex.idle)
    assert ran == ["first"]


def test_stop_action_ends_the_executor():
    stopped = threading.Event()
    q = CommandQueue()
    ex = Executor(q, lambda cmd: cmd.text != "stop", on_stop=stopped.set)
    ex.start()
    q.put(Command("stop"))
    q.put(Command("never runs"))
    assert stopped.wait(2)
    assert ex.stopped.is_set()


class _Brain:
    def __init__(self, gate: threading.Event | None = None):
        self.calls: list[str] = []
        self.gate = gate
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def evaluate(self, text):
        with self.lock:
            self.calls.append(text)
            self.active += 1
            self.peak = max(self.peak, self.active)
        if self.gate:
            self.gate.wait(2)
        with self.lock:
            self.active -= 1
        return SimpleNamespace(utterance=text, action="open_app")


def _speculator(brain, front="Notes"):
    q = CommandQueue()
    ex = Executor(q, lambda cmd: True)
    state = {"front": front}
    spec = Speculator(brain.evaluate, lambda: state["front"], ex)
    return spec, ex, q, state


def test_an_early_plan_is_used_when_the_words_match():
    brain = _Brain()
    spec, ex, _, _ = _speculator(brain)
    spec.reset()
    spec.offer("open google chrome")
    plan = spec.take("Open Google Chrome.")
    assert plan is not None and plan.utterance == "open google chrome"
    assert brain.calls == ["open google chrome"]


def test_an_early_plan_is_dropped_when_the_words_changed():
    brain = _Brain()
    spec, *_ = _speculator(brain)
    spec.reset()
    spec.offer("open google chrome")
    assert spec.take("open google chrome and search cats") is None


def test_an_early_plan_is_dropped_after_an_action_ran():
    brain = _Brain()
    spec, ex, q, _ = _speculator(brain)
    spec.reset()
    spec.offer("write good morning")
    assert _wait(lambda: normalize("write good morning") in spec._cache)
    ex.start()
    q.put(Command("open notes"))
    assert _wait(lambda: ex.generation == 1)
    assert spec.take("write good morning") is None


def test_an_early_plan_is_dropped_when_the_front_app_changed():
    brain = _Brain()
    spec, _, _, state = _speculator(brain)
    spec.reset()
    spec.offer("write good morning")
    assert _wait(lambda: normalize("write good morning") in spec._cache)
    state["front"] = "Safari"
    assert spec.take("write good morning") is None


def test_no_early_plan_while_something_is_running():
    brain = _Brain()
    release = threading.Event()
    q = CommandQueue()
    ex = Executor(q, lambda cmd: release.wait(2) or True)
    ex.start()
    q.put(Command("long task"))
    assert _wait(lambda: ex.busy)
    spec = Speculator(brain.evaluate, lambda: "Notes", ex)
    spec.reset()
    spec.offer("open google chrome")
    time.sleep(0.05)
    release.set()
    assert brain.calls == []


def test_one_request_at_a_time_and_three_per_phrase():
    gate = threading.Event()
    brain = _Brain(gate)
    spec, *_ = _speculator(brain)
    spec.reset()
    for partial in ["open the notes app", "open the notes app now", "open the notes app now please",
                    "open the notes app now please jeff", "open the notes app now please jeff thanks"]:
        spec.offer(partial)
    time.sleep(0.05)
    assert len(brain.calls) == 1
    gate.set()
    assert _wait(lambda: len(brain.calls) == 2)
    time.sleep(0.05)
    assert brain.peak == 1
    assert len(brain.calls) <= 3


def test_short_partials_wait_until_they_settle():
    brain = _Brain()
    spec, *_ = _speculator(brain)
    spec.reset()
    spec.offer("open")
    spec.offer("open chrome")
    time.sleep(0.05)
    assert brain.calls == []
    spec.offer("open chrome")
    assert _wait(lambda: brain.calls == ["open chrome"])


def test_speech_not_meant_for_us_is_not_planned():
    brain = _Brain()
    spec, *_ = _speculator(brain)
    spec.prepare = lambda text: None
    spec.reset()
    spec.offer("I was telling my friend about chrome")
    time.sleep(0.05)
    assert brain.calls == []


class _Island:
    def __init__(self):
        self.shown: list[tuple[str, str]] = []

    def set(self, state, text, revert_after=None):
        self.shown.append((state, text))


def test_live_transcript_owns_the_island_while_speaking():
    island = _Island()
    feed = StatusFeed(island, pending=lambda: 2)
    feed.voice(True)
    feed.set("heard", "open chr")
    feed.set("done", "Open Google Chrome")
    assert island.shown == [("heard", "open chr")]
    feed.voice(False)
    assert island.shown[-1] == ("done", "Open Google Chrome  ·  +2 queued")


def _session(busy: bool, waiting: list[str]):
    q = CommandQueue()
    for text in waiting:
        q.put(Command(text))
    said: list[str] = []
    return SimpleNamespace(queue=q, executor=SimpleNamespace(busy=busy), paused=False,
                           speaker=SimpleNamespace(say=said.append)), said


@pytest.fixture
def quiet(monkeypatch):
    from jev_voice import main

    monkeypatch.setattr(main, "ding", lambda path: None)
    return main


def test_cancel_clears_waiting_commands(quiet):
    s, _ = _session(busy=True, waiting=["write hello", "press enter"])
    assert quiet.cancel_queued(s, "Cancel that.")
    assert s.queue.pending() == 0


def test_hey_jeff_never_mind_also_cancels(quiet):
    s, _ = _session(busy=False, waiting=["open notes"])
    assert quiet.cancel_queued(s, "hey Jeff, never mind")
    assert s.queue.pending() == 0


def test_bare_stop_with_nothing_queued_still_reaches_jev(quiet):
    s, _ = _session(busy=False, waiting=[])
    assert not quiet.cancel_queued(s, "stop")
    assert not quiet.cancel_queued(s, "open chrome")


def test_stop_while_busy_clears_instead_of_quitting(quiet):
    s, _ = _session(busy=True, waiting=["open notes"])
    assert quiet.cancel_queued(s, "stop")
    assert s.queue.pending() == 0


def test_deactivate_clears_the_queue(quiet):
    s, _ = _session(busy=True, waiting=["open notes", "write hello"])
    quiet._ack_pause(s)
    assert s.paused
    assert s.queue.pending() == 0


@pytest.mark.parametrize(("now", "until", "muted", "drop"), [
    (1.0, 0.0, False, False),
    (1.0, 2.0, False, True),
    (1.0, 0.0, True, True),
])
def test_listener_drops_audio_while_jev_is_making_sound(now, until, muted, drop):
    assert should_drop(now, until, muted) is drop
