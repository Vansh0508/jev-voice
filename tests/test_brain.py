"""Offline plan mapping for single commands. No paid APIs."""

from unittest.mock import Mock

from jev_voice.brain import ACTIONS, Brain, Plan
from jev_voice.main import handle


def _brain() -> Brain:
    return object.__new__(Brain)


def _answers(**overrides):
    base = {
        "action": {"choice": "none", "confidence": 0.9},
        "close_target": {"choice": "none", "confidence": 0.9},
        "compound": {"noul": 0.0},
        "recommend": {"noul": 0.0},
        "message_seller": {"noul": 0.0},
        "addressed": {"noul": 0.9},
        "in_app": {"noul": 0.0},
    }
    base.update(overrides)
    return base


def test_close_app_is_a_choice_and_has_its_own_target_head():
    questions = _brain()._questions({}, ["Notes", "Google Chrome"])
    assert "close_app" in ACTIONS
    assert "close_app" in questions["action"]["criteria"]
    criteria = questions["close_target"]["criteria"]
    assert "Google Chrome" in criteria
    assert "frontmost" in criteria
    assert "none" in criteria


def test_close_app_plan_uses_only_the_close_target():
    plan = _brain()._to_plan(
        "close chrome",
        _answers(
            action={"choice": "close_app", "confidence": 0.95},
            close_target={"choice": "Google Chrome", "confidence": 0.8},
        ),
        {},
        12,
    )
    assert plan.action == "close_app"
    assert plan.args["app"] == "Google Chrome"
    assert plan.confidence == 0.8


def test_unnamed_close_targets_the_frontmost_app():
    plan = _brain()._to_plan(
        "close this app",
        _answers(
            action={"choice": "close_app", "confidence": 0.9},
            close_target={"choice": "frontmost", "confidence": 0.7},
        ),
        {},
        10,
    )
    assert plan.args["app"] == "frontmost"
    assert plan.confidence == 0.7


def test_type_into_a_named_app_keeps_the_payload_without_the_place():
    from jev_voice.brain import text_candidates

    spans = set(text_candidates("write youtube.com on google chrome").values())
    assert "youtube.com" in spans
    assert "screen" in ACTIONS


def test_a_write_request_keeps_only_the_words_to_insert():
    from jev_voice.brain import text_candidates

    spans = list(text_candidates("I want you to write good morning").values())
    assert spans[0] == "good morning"
    said = "I want you to write this message has been written completely hands-free"
    assert "this message has been written completely hands-free" in text_candidates(said).values()
    cut = text_candidates("I want you to write the model is unable to understand when I want to write some text on the")
    assert "the" not in cut.values()
    assert any(span.startswith("the model is unable") for span in cut.values())


def test_click_in_a_named_app_is_a_screen_action():
    assert "click the image on photo booth" in ACTIONS["screen"]
    assert "do not search" in ACTIONS["screen"]


def test_missing_accessibility_does_not_end_the_voice_session(monkeypatch):
    monkeypatch.setattr("jev_voice.main.execute", Mock(side_effect=RuntimeError(
        "Accessibility permission missing: System Settings → Privacy & Security → Accessibility → add your terminal."
    )))
    plan = Plan("click the picture", "screen", 1.0, {"compound": False})
    assert handle(Mock(), Mock(), "click the picture", dry=False, plan=plan) is True
