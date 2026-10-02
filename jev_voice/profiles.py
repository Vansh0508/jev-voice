"""Browser profiles. Code lists them; Jev picks one. A browser with a single profile is not a choice.

Profile names come from the browser's Local State file. macOS hides that file unless the
app running Jev Voice has Full Disk Access. Without it, "another profile" still uses the
profile button in the browser window on this desktop.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOTS = {
    "Google Chrome": "Google/Chrome",
    "Brave Browser": "BraveSoftware/Brave-Browser",
    "Microsoft Edge": "Microsoft Edge",
    "Chromium": "Chromium",
    "Vivaldi": "Vivaldi",
    "Opera": "com.operasoftware.Opera",
}
_SKIP_MENU = {
    "add", "add profile", "manage profiles", "guest", "close", "edit", "edit profile",
    "customize profile", "other profiles", "sign in", "passwords and autofill", "settings",
}


@dataclass
class BrowserProfile:
    key: str
    browser: str
    name: str
    directory: str
    current: bool

    @property
    def label(self) -> str:
        mark = " (current profile)" if self.current else ""
        return f"{self.browser}: {self.name}{mark}"


def profiles_from_info(browser: str, info: dict[str, Any], last_used: str | None) -> list[BrowserProfile]:
    """One browser's profiles. Empty when it does not have at least two."""
    found: list[BrowserProfile] = []
    for directory, meta in info.items():
        if not isinstance(meta, dict):
            continue
        name = str(meta.get("name") or meta.get("gaia_name") or directory).strip()
        if not name:
            continue
        found.append(BrowserProfile("", browser, name, str(directory), str(directory) == str(last_used or "")))
    if len(found) < 2:
        return []
    return found


def load_profiles() -> list[BrowserProfile]:
    """Profiles of every Chromium browser that has more than one. Unreadable files are skipped."""
    found: list[BrowserProfile] = []
    support = Path.home() / "Library" / "Application Support"
    for browser, folder in _ROOTS.items():
        path = support / folder / "Local State"
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        profile = data.get("profile") or {}
        found.extend(profiles_from_info(browser, profile.get("info_cache") or {}, profile.get("last_used")))
    for i, profile in enumerate(found, start=1):
        profile.key = f"p{i}"
    return found


def other_profile(profiles: list[BrowserProfile], browser: str | None) -> BrowserProfile | None:
    group = [p for p in profiles if browser is None or p.browser == browser]
    if len(group) < 2:
        return None
    rest = [p for p in group if not p.current]
    return rest[0] if rest else group[1]


def resolve_browser(preferred: str | None, profiles: list[BrowserProfile], front: str | None, on_desktop: list[str]) -> str | None:
    from .tabs import canonical_browser, is_browser

    if preferred and is_browser(preferred):
        return canonical_browser(preferred)
    have = []
    for profile in profiles:
        if profile.browser not in have:
            have.append(profile.browser)
    if front and is_browser(front) and (not have or canonical_browser(front) in have):
        return canonical_browser(front)
    for name in on_desktop:
        if not have or name in have:
            return name
    if len(have) == 1:
        return have[0]
    return None


def pick_profile_button(buttons: list[tuple[str, str, float]]) -> int | None:
    """Index of the avatar button. `buttons` are (title, description, x) along the toolbar."""
    for i, (title, desc, _) in enumerate(buttons):
        blob = f"{title} {desc}".casefold()
        if "profile" in blob and "add" not in blob and "manage" not in blob:
            return i
    menu = [i for i, (title, desc, _) in enumerate(buttons) if "customize and control" in f"{title} {desc}".casefold()]
    if not menu:
        return None
    edge = buttons[menu[0]][2]
    left = [(i, x) for i, (_, _, x) in enumerate(buttons) if x < edge - 8]
    if not left:
        return None
    return max(left, key=lambda item: item[1])[0]


def pick_menu_profile(titles: list[str], current: str, wanted: str | None) -> str | None:
    choices = [title for title in titles if _is_profile_choice(title)]
    if wanted:
        folded = wanted.casefold()
        return next((title for title in choices if title.casefold() == folded or folded in title.casefold()), None)
    if len(choices) < 2:
        return None
    if current:
        others = [title for title in choices if title.casefold() != current.casefold()]
        return others[0] if others else None
    return choices[1]


def switch_profile(browser: str | None, directory: str | None, name: str | None, other: bool) -> str:
    from .actions import frontmost_app

    profiles = load_profiles()
    if other:
        try:
            from .tabs import browser_windows_on_this_desktop

            on_desktop = [window.browser for window in browser_windows_on_this_desktop()]
        except Exception:
            on_desktop = []
        browser = resolve_browser(browser, profiles, frontmost_app(), on_desktop)
        target = other_profile(profiles, browser)
        if target is not None:
            return _launch(target.browser, target.directory, target.name)
        return _switch_from_menu(browser, None)
    if directory and name and browser:
        return _launch(browser, directory, name)
    if name:
        return _switch_from_menu(_named(browser), name)
    return "I don't see another profile."


def _named(browser: str | None) -> str | None:
    from .tabs import canonical_browser, is_browser

    if browser and is_browser(browser):
        return canonical_browser(browser)
    return None


def _launch(browser: str, directory: str, name: str) -> str:
    subprocess.Popen(
        ["open", "-a", browser, "--args", f"--profile-directory={directory}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return f"Switching to {name}."


def _switch_from_menu(browser: str | None, wanted: str | None) -> str:
    from .actions import click_at, frontmost_app, press
    from .tabs import browser_windows_on_this_desktop, focus_browser_on_this_desktop

    err = focus_browser_on_this_desktop(browser)
    if err:
        return err
    windows = [w for w in browser_windows_on_this_desktop() if browser is None or w.browser == browser]
    if not windows:
        front = frontmost_app()
        windows = [w for w in browser_windows_on_this_desktop() if w.browser == front]
    if not windows:
        return "I don't see a browser on this desktop."
    window = windows[0]
    buttons = _toolbar_buttons(window)
    index = pick_profile_button([(title, desc, frame[0]) for title, desc, frame, _ in buttons])
    if index is None:
        return "I don't see another profile."
    title, desc, frame, el = buttons[index]
    fresh = _frame(el) or frame
    click_at(fresh[0] + fresh[2] / 2, fresh[1] + fresh[3] / 2)
    time.sleep(0.35)
    choices = _menu_titles(window.pid)
    picked = pick_menu_profile(choices, title or desc, wanted)
    if picked is None:
        press("escape")
        return "This browser has one profile." if not wanted else "I don't see that profile."
    target = _menu_element(window.pid, picked)
    if target is None:
        press("escape")
        return "I don't see that profile."
    box = _frame(target)
    if box is None:
        press("escape")
        return "I don't see that profile."
    click_at(box[0] + box[2] / 2, box[1] + box[3] / 2)
    return f"Switching to {picked}."


def _is_profile_choice(title: str) -> bool:
    text = title.strip()
    return bool(text) and text.casefold() not in _SKIP_MENU and not text.casefold().startswith("add ")


def _toolbar_buttons(window: Any) -> list[tuple[str, str, tuple[float, float, float, float], Any]]:
    from .desktop import _list, _multi, _point, _size, _str

    attrs = ["AXRole", "AXTitle", "AXDescription", "AXPosition", "AXSize", "AXChildren"]
    found: list[tuple[str, str, tuple[float, float, float, float], Any]] = []
    stack = [(window.el, 0)]
    seen = 0
    while stack and seen < 300:
        el, depth = stack.pop(0)
        if depth > 8:
            continue
        vals = _multi(el, attrs)
        if not vals:
            continue
        seen += 1
        role = _str(vals[0])
        if role == "AXWebArea":
            continue
        pos, size = _point(vals[3]), _size(vals[4])
        frame = (pos[0], pos[1], size[0], size[1]) if pos and size else None
        if role == "AXButton" and frame and window.frame[1] <= frame[1] <= window.frame[1] + 90:
            found.append((_str(vals[1]), _str(vals[2]), frame, el))
        if depth < 8:
            stack.extend((child, depth + 1) for child in _list(vals[5]))
    return found


def _menu_titles(pid: int) -> list[str]:
    return [title for title, _ in _menu_items(pid)]


def _menu_element(pid: int, title: str) -> Any:
    for label, el in _menu_items(pid):
        if label == title:
            return el
    return None


def _menu_items(pid: int) -> list[tuple[str, Any]]:
    import ApplicationServices as AS  # type: ignore

    from .desktop import _list, _multi, _str

    app = AS.AXUIElementCreateApplication(pid)
    err, windows = AS.AXUIElementCopyAttributeValue(app, "AXWindows", None)
    if err != 0 or not windows:
        return []
    attrs = ["AXRole", "AXTitle", "AXDescription", "AXChildren"]
    found: list[tuple[str, Any]] = []
    stack = [(el, 0) for el in windows]
    seen = 0
    while stack and seen < 400:
        el, depth = stack.pop(0)
        if depth > 8:
            continue
        vals = _multi(el, attrs)
        if not vals:
            continue
        seen += 1
        role = _str(vals[0])
        title = _str(vals[1]) or _str(vals[2])
        if role in {"AXMenuItem", "AXButton", "AXStaticText"} and title:
            found.append((title, el))
        if depth < 8:
            stack.extend((child, depth + 1) for child in _list(vals[3]))
    return found


def _frame(el: Any) -> tuple[float, float, float, float] | None:
    from .desktop import _multi, _point, _size

    vals = _multi(el, ["AXPosition", "AXSize"])
    if not vals:
        return None
    pos, size = _point(vals[0]), _size(vals[1])
    if not pos or not size or size[0] <= 0 or size[1] <= 0:
        return None
    return (pos[0], pos[1], size[0], size[1])
