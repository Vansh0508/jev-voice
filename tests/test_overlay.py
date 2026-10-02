"""Island sizing. No window."""

from jev_voice.overlay import BODY, island_size


def test_idle_merges_into_the_notch():
    assert island_size("idle", "Listening", 80, 208, 37.5) == (208, 37.5, True)
    assert island_size("idle", "", 0, 208, 37.5) == (208, 37.5, True)


def test_speech_grows_below_the_notch():
    width, height, merged = island_size("heard", "open chrome", 90, 208, 37.5)
    assert not merged
    assert width > 208
    assert height == 37.5 + BODY


def test_a_long_line_stays_within_the_cap():
    width, _, merged = island_size("thinking", "x" * 400, 2000, 208, 37.5)
    assert not merged
    assert width == 480
