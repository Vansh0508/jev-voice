"""Profile choices. No browser window and no Local State file."""

from jev_voice.brain import ACTIONS, Brain
from jev_voice.profiles import (
    other_profile,
    pick_menu_profile,
    pick_profile_button,
    profiles_from_info,
    resolve_browser,
)


def test_one_profile_is_not_a_choice():
    assert profiles_from_info("Brave Browser", {"Default": {"name": "Personal"}}, "Default") == []


def test_the_other_profile_is_the_one_not_in_front():
    profiles = profiles_from_info("Brave Browser", {
        "Default": {"name": "Personal"},
        "Profile 1": {"name": "Work"},
    }, "Default")
    picked = other_profile(profiles, "Brave Browser")
    assert picked is not None
    assert picked.name == "Work"
    assert picked.directory == "Profile 1"


def test_the_avatar_is_the_button_left_of_the_browser_menu():
    buttons = [
        ("Back", "", 20.0),
        ("Personal", "", 460.0),
        ("Customize and control Brave", "", 520.0),
    ]
    assert pick_profile_button(buttons) == 1


def test_another_profile_skips_the_current_name():
    assert pick_menu_profile(["Personal", "Work", "Guest", "Add profile"], "Personal", None) == "Work"
    assert pick_menu_profile(["Personal"], "Personal", None) is None


def test_switch_profile_plan_uses_the_named_profile():
    profiles = profiles_from_info("Google Chrome", {
        "Default": {"name": "Personal"},
        "Profile 1": {"name": "Work"},
    }, "Default")
    for i, profile in enumerate(profiles, start=1):
        profile.key = f"p{i}"
    questions = Brain.__new__(Brain)._questions({}, ["Google Chrome"], profiles=profiles)
    assert "switch_profile" in ACTIONS
    assert questions["profile_target"]["criteria"]["p2"] == "Google Chrome: Work"
    plan = Brain.__new__(Brain)._to_plan(
        "switch to the work profile",
        {
            "action": {"choice": "switch_profile", "confidence": 0.9},
            "profile_target": {"choice": "p2", "confidence": 0.8},
            "compound": {"noul": 0.0},
            "recommend": {"noul": 0.0},
            "message_seller": {"noul": 0.0},
            "addressed": {"noul": 0.9},
            "in_app": {"noul": 0.0},
        },
        {},
        10,
        None,
        profiles,
    )
    assert plan.args["browser"] == "Google Chrome"
    assert plan.args["name"] == "Work"
    assert plan.args["directory"] == "Profile 1"
    assert plan.confidence == 0.8


def test_an_unnamed_switch_uses_the_browser_on_this_desktop():
    assert resolve_browser(None, [], "Notes", ["Brave Browser"]) == "Brave Browser"
    assert resolve_browser("chrome", [], "Safari", ["Brave Browser"]) == "Google Chrome"
