"""PyAutoGUI is the mouse and keyboard backend. No screen, no real input."""

from unittest.mock import Mock

from jev_voice import actions
from jev_voice.desktop import Desktop


def test_type_text_goes_through_pyautogui(monkeypatch):
    gui = Mock()
    monkeypatch.setattr(actions, "_gui", lambda: gui)
    monkeypatch.setattr(actions, "_gui_can_type", lambda text: True)
    actions.type_text("youtube.com")
    gui.write.assert_called_once_with("youtube.com", interval=0)


def test_shortcut_goes_through_pyautogui(monkeypatch):
    gui = Mock()
    monkeypatch.setattr(actions, "_gui", lambda: gui)
    actions.press("copy")
    gui.hotkey.assert_called_once_with("command", "c")


def test_screen_click_uses_the_resolved_point(monkeypatch):
    seen = {}
    monkeypatch.setattr(actions, "click_at", lambda x, y: seen.update(x=x, y=y))
    Desktop._click(12.5, 40.0)
    assert seen == {"x": 12.5, "y": 40.0}
