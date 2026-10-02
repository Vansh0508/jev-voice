"""Wake, pause, and reactivate. No microphone."""

from jev_voice.main import apply_listen_gate


def test_hey_jeff_wakes_and_keeps_the_command():
    assert apply_listen_gate("hey Jeff", paused=True) == ("resume", "")
    assert apply_listen_gate("Hey, Jeff, open chrome", paused=True) == ("resume", "open chrome")
    assert apply_listen_gate("Jeff open chrome", paused=True) == ("resume", "open chrome")


def test_deactivate_stops_and_reactivate_starts():
    assert apply_listen_gate("deactivate", paused=False) == ("pause", "")
    assert apply_listen_gate("de activate.", paused=False) == ("pause", "")
    assert apply_listen_gate("hey Jeff, deactivate", paused=False) == ("pause", "")
    assert apply_listen_gate("Pause Jev", paused=False) == ("pass", "Pause Jev")
    assert apply_listen_gate("reactivate", paused=True) == ("resume", "")
    assert apply_listen_gate("re activate", paused=True) == ("resume", "")


def test_paused_drops_every_other_utterance():
    assert apply_listen_gate("open chrome", paused=True) == ("ignore", "")
    assert apply_listen_gate("write good morning", paused=True) == ("ignore", "")
    assert apply_listen_gate("open chrome", paused=False) == ("pass", "open chrome")
