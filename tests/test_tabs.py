"""Tab choice and which browser window is on this desktop. No screen, no AppleScript."""

from jev_voice import tabs
from jev_voice.brain import ACTIONS, Brain
from jev_voice.tabs import BrowserWindow, choose_browser, is_tab_control, open_on_this_desktop


class _Tab:
    def __init__(self, key: str, browser: str, title: str, label: str) -> None:
        self.key, self.browser, self.title, self.label = key, browser, title, label


def test_switch_tab_is_a_choice_over_open_tabs():
    questions = Brain.__new__(Brain)._questions({}, ["Google Chrome"], [
        _Tab("t1", "Google Chrome", "YouTube", "Google Chrome: YouTube (current tab)"),
    ])
    assert "switch_tab" in ACTIONS
    assert "switch_tab" in questions["action"]["criteria"]
    assert questions["tab_target"]["criteria"]["t1"] == "Google Chrome: YouTube (current tab)"
    assert "none" in questions["tab_target"]["criteria"]


def test_switch_tab_plan_keeps_only_the_chosen_tab():
    plan = Brain.__new__(Brain)._to_plan(
        "switch to the youtube tab",
        {
            "action": {"choice": "switch_tab", "confidence": 0.92},
            "tab_target": {"choice": "t1", "confidence": 0.7},
            "compound": {"noul": 0.0},
            "recommend": {"noul": 0.0},
            "message_seller": {"noul": 0.0},
            "addressed": {"noul": 0.9},
            "in_app": {"noul": 0.0},
        },
        {},
        10,
        [_Tab("t1", "Google Chrome", "YouTube", "Google Chrome: YouTube")],
    )
    assert plan.action == "switch_tab"
    assert plan.args["browser"] == "Google Chrome"
    assert plan.args["title"] == "YouTube"
    assert plan.confidence == 0.7


def test_a_page_control_is_not_a_tab():
    window = (0.0, 0.0, 1400.0, 900.0)
    assert is_tab_control("AXRadioButton", "tab", "YouTube", (200.0, 12.0, 160.0, 28.0), window)
    assert not is_tab_control("AXRadioButton", "radio button", "Blue", (100.0, 400.0, 20.0, 20.0), window)
    assert not is_tab_control("AXButton", "close button", "Close", (8.0, 8.0, 14.0, 14.0), window)


def test_the_front_browser_on_this_desktop_wins():
    assert choose_browser(["Safari", "Google Chrome"], None, "Google Chrome") == "Google Chrome"
    assert choose_browser(["Safari"], "Google Chrome", "Safari") is None


def test_a_url_opens_in_the_browser_already_on_this_desktop(monkeypatch):
    window = BrowserWindow("Brave Browser", 1, None, (0.0, 0.0, 1200.0, 800.0))
    monkeypatch.setattr(tabs, "browser_windows_on_this_desktop", lambda: [window])
    monkeypatch.setattr("jev_voice.actions.frontmost_app", lambda: "Notes")
    typed: dict[str, str] = {}
    monkeypatch.setattr(tabs, "_type_url_in", lambda win, url: typed.update(browser=win.browser, url=url))
    assert open_on_this_desktop("https://www.youtube.com") is None
    assert typed == {"browser": "Brave Browser", "url": "https://www.youtube.com"}


def test_a_named_browser_on_another_desktop_gets_a_new_window_here(monkeypatch):
    window = BrowserWindow("Google Chrome", 1, None, (0.0, 0.0, 1200.0, 800.0))
    monkeypatch.setattr(tabs, "browser_windows_on_this_desktop", lambda: [window])
    monkeypatch.setattr("jev_voice.actions.frontmost_app", lambda: "Google Chrome")
    made: dict[str, str] = {}
    monkeypatch.setattr(tabs, "_new_window", lambda browser, url: made.update(browser=browser, url=url) or None)
    assert open_on_this_desktop("https://www.youtube.com", "Safari") is None
    assert made == {"browser": "Safari", "url": "https://www.youtube.com"}
