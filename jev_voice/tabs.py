"""Browser tabs that are actually on the current macOS desktop (Space).

A window on another desktop is not a candidate. New tabs are opened in a browser
window already on this desktop. Switching chooses one of the tabs code listed;
the model never emits a selector.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Any

BROWSER_NAMES = (
    "Google Chrome", "Safari", "Brave Browser", "Microsoft Edge", "Arc",
    "Chromium", "Opera", "Vivaldi", "Firefox", "Dia", "Orion",
)
ALIASES = {
    "chrome": "Google Chrome",
    "brave": "Brave Browser",
    "edge": "Microsoft Edge",
    "firefox": "Firefox",
    "safari": "Safari",
    "arc": "Arc",
}
# Toolbar controls that sit in the same band as the tab strip.
_NOT_TABS = {
    "close", "minimize", "zoom", "back", "forward", "reload", "reload this page", "stop",
    "home", "new tab", "customize and control", "more", "menu", "extensions", "downloads",
    "find", "address and search bar", "search or enter address",
}
TAB_SHORTCUTS = {"new_tab", "close_tab_or_window", "reopen_closed_tab", "next_tab", "previous_tab"}
_TAB_ROLES = {"AXRadioButton", "AXTab", "AXTabButton"}


@dataclass
class BrowserWindow:
    browser: str
    pid: int
    el: Any
    frame: tuple[float, float, float, float]


@dataclass
class LiveTab:
    key: str
    browser: str
    title: str
    selected: bool
    el: Any
    window: BrowserWindow

    @property
    def label(self) -> str:
        mark = " (current tab)" if self.selected else ""
        return f"{self.browser}: {self.title}{mark}"


def is_browser(name: str | None) -> bool:
    if not name:
        return False
    folded = name.casefold()
    if folded in ALIASES or folded in {b.casefold() for b in BROWSER_NAMES}:
        return True
    return any(folded in b.casefold() or b.casefold() in folded for b in BROWSER_NAMES)


def canonical_browser(name: str) -> str:
    folded = name.casefold()
    if folded in ALIASES:
        return ALIASES[folded]
    for browser in BROWSER_NAMES:
        if browser.casefold() == folded:
            return browser
    return name


def choose_browser(owners: list[str], preferred: str | None, front: str | None) -> str | None:
    """Which on-screen browser should receive a tab. A named browser that is not here is absent."""
    names: list[str] = []
    for owner in owners:
        if owner not in names:
            names.append(owner)
    if preferred:
        return _match(preferred, names)
    if front:
        hit = _match(front, names)
        if hit:
            return hit
    return names[0] if names else None


def _match(wanted: str, names: list[str]) -> str | None:
    target = canonical_browser(wanted).casefold()
    for name in names:
        if name.casefold() == target or name.casefold() == wanted.casefold():
            return name
    if wanted.casefold() == "chrome":
        return next((name for name in names if "chrome" in name.casefold()), None)
    return None


def is_tab_control(role: str, role_desc: str, title: str, frame: tuple[float, float, float, float] | None,
                   window_frame: tuple[float, float, float, float]) -> bool:
    """A tab strip (or sidebar) control, not a button inside the web page."""
    if not title or title.casefold() in _NOT_TABS or frame is None:
        return False
    tab_role = role in _TAB_ROLES or "tab" in role_desc.casefold()
    if not tab_role:
        return False
    x, y, w, h = frame
    wx, wy, ww, wh = window_frame
    in_top = wy <= y <= wy + 120 and wx - 4 <= x <= wx + ww
    # A sidebar tab names itself as a tab. A radio button further down the page does not.
    in_side = "tab" in role_desc.casefold() and wx <= x <= wx + 320 and h <= 48 and wy <= y <= wy + wh
    return (in_top or in_side) and w > 8 and h > 8


def list_tabs() -> list[LiveTab]:
    """Tabs belonging to browser windows on this desktop. Empty when none are here."""
    found: list[LiveTab] = []
    for window in browser_windows_on_this_desktop():
        for title, selected, el in _tab_controls(window):
            found.append(LiveTab("", window.browser, title, selected, el, window))
            if len(found) >= 40:
                break
        if len(found) >= 40:
            break
    for i, tab in enumerate(found, start=1):
        tab.key = f"t{i}"
    return found


def browser_windows_on_this_desktop() -> list[BrowserWindow]:
    import ApplicationServices as AS  # type: ignore

    from .desktop import _bool, _multi, _point, _size

    if not _trusted():
        return []
    windows: list[BrowserWindow] = []
    apps: dict[int, Any] = {}
    seen: set[int] = set()
    for browser, pid, cg_frame in _on_screen_browsers():
        if pid not in apps:
            apps[pid] = _enable(pid)
        app = apps[pid]
        err, ax_windows = AS.AXUIElementCopyAttributeValue(app, "AXWindows", None)
        if err != 0 or not ax_windows:
            continue
        for el in ax_windows:
            if id(el) in seen:
                continue
            vals = _multi(el, ["AXPosition", "AXSize", "AXMinimized"])
            if not vals:
                continue
            if _bool(vals[2]) is True:
                continue
            pos, size = _point(vals[0]), _size(vals[1])
            if not pos or not size or size[0] < 400 or size[1] < 200:
                continue
            frame = (pos[0], pos[1], size[0], size[1])
            if _near(frame, cg_frame):
                seen.add(id(el))
                windows.append(BrowserWindow(browser, pid, el, frame))
                break
    return windows


def open_on_this_desktop(url: str, browser: str | None = None) -> str | None:
    """Open url as a new tab of a browser window on this desktop. None means it happened."""
    from .actions import frontmost_app

    if browser and not is_browser(browser):
        browser = None
    windows = browser_windows_on_this_desktop()
    chosen = choose_browser([w.browser for w in windows], browser, frontmost_app())
    if chosen:
        window = next(w for w in windows if w.browser == chosen)
        _type_url_in(window, url)
        return None
    target = canonical_browser(browser) if browser else default_browser_name()
    return _new_window(target, url)


def new_tab_on_this_desktop(browser: str | None = None) -> str | None:
    """Command-T in the browser on this desktop, or a new window here when none is open."""
    err = focus_browser_on_this_desktop(browser)
    if err is None:
        from .actions import press

        press("new_tab")
        return None
    target = canonical_browser(browser) if browser and is_browser(browser) else default_browser_name()
    return _new_window(target, "")


def focus_browser_on_this_desktop(preferred: str | None = None) -> str | None:
    """Raise the browser window on this desktop. Returns an error sentence, or None."""
    from .actions import frontmost_app

    if preferred and not is_browser(preferred):
        preferred = None
    windows = browser_windows_on_this_desktop()
    chosen = choose_browser([w.browser for w in windows], preferred, frontmost_app())
    if not chosen:
        if preferred:
            return f"{canonical_browser(preferred)} isn't on this desktop."
        return "I don't see a browser on this desktop."
    _raise(next(w for w in windows if w.browser == chosen))
    return None


def switch_tab(browser: str, title: str) -> str:
    """Re-find the tab on this desktop and click it. A stale coordinate is never reused."""
    matches = [t for t in list_tabs() if t.browser == browser and t.title == title]
    if not matches:
        return "I don't see that tab."
    tab = next((t for t in matches if not t.selected), matches[0])
    _raise(tab.window)
    if tab.selected:
        return f"Switching to {title}."
    frame = _fresh_frame(tab.el)
    if frame is None or not _hit_ok(tab.el, frame):
        return "I don't see that tab."
    from .actions import click_at

    click_at(frame[0] + frame[2] / 2, frame[1] + frame[3] / 2)
    return f"Switching to {title}."


def default_browser_name() -> str:
    try:
        from AppKit import NSURL, NSWorkspace  # type: ignore

        url = NSWorkspace.sharedWorkspace().URLForApplicationToOpenURL_(NSURL.URLWithString_("https://example.com"))
        if url is not None:
            name = str(url.lastPathComponent() or "")
            if name.endswith(".app"):
                name = name[:-4]
            if name:
                return name
    except Exception:
        pass
    return "Safari"


def _on_screen_browsers() -> list[tuple[str, int, tuple[float, float, float, float]]]:
    import Quartz  # type: ignore

    info = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    found = []
    known = {name.casefold(): name for name in BROWSER_NAMES}
    for win in info or []:
        if int(win.get("kCGWindowLayer", 0)) != 0:
            continue
        owner = str(win.get("kCGWindowOwnerName") or "")
        browser = known.get(owner.casefold())
        if not browser:
            continue
        bounds = win.get("kCGWindowBounds") or {}
        frame = (float(bounds.get("X", 0)), float(bounds.get("Y", 0)),
                 float(bounds.get("Width", 0)), float(bounds.get("Height", 0)))
        if frame[2] < 400 or frame[3] < 200:
            continue
        found.append((browser, int(win.get("kCGWindowOwnerPID") or 0), frame))
    return found


def _tab_controls(window: BrowserWindow) -> list[tuple[str, bool, Any]]:
    from .desktop import _bool, _list, _multi, _point, _size, _str

    attrs = ["AXRole", "AXTitle", "AXDescription", "AXValue", "AXSelected", "AXPosition", "AXSize",
             "AXChildren", "AXRoleDescription", "AXVisibleChildren"]
    out: list[tuple[str, bool, Any, float, float]] = []
    stack = [(window.el, 0)]
    seen = 0
    deadline = time.perf_counter() + 0.35
    while stack and seen < 400 and time.perf_counter() < deadline:
        el, depth = stack.pop(0)
        if depth > 10:
            continue
        vals = _multi(el, attrs)
        if not vals:
            continue
        seen += 1
        role = _str(vals[0])
        if role == "AXWebArea":
            continue
        title = _str(vals[1]) or _str(vals[2])
        pos, size = _point(vals[5]), _size(vals[6])
        frame = (pos[0], pos[1], size[0], size[1]) if pos and size else None
        selected = _bool(vals[4]) is True or _str(vals[3]) in {"1", "true"}
        if is_tab_control(role, _str(vals[8], 40), title, frame, window.frame):
            out.append((title, selected, el, frame[0], frame[1]))  # type: ignore[index]
        if depth < 10:
            stack.extend((child, depth + 1) for child in (_list(vals[7]) or _list(vals[9])))
    out.sort(key=lambda item: (item[4], item[3]))
    return [(title, selected, el) for title, selected, el, _, _ in out]


def _type_url_in(window: BrowserWindow, url: str) -> None:
    from .actions import press, type_text

    _raise(window)
    press("new_tab")
    time.sleep(0.2)
    press("address_bar")
    time.sleep(0.05)
    type_text(url)
    press("enter")


def _raise(window: BrowserWindow) -> None:
    import ApplicationServices as AS  # type: ignore

    from .actions import _activate

    try:
        AS.AXUIElementPerformAction(window.el, "AXRaise")
    except Exception:
        pass
    _activate(window.browser)
    time.sleep(0.12)


def _new_window(browser: str, url: str) -> str | None:
    safe = url.replace("\\", "\\\\").replace('"', '\\"') if url else ""
    app = browser.replace("\\", "\\\\").replace('"', '\\"')
    if browser == "Safari":
        body = f'make new document with properties {{URL:"{safe}"}}' if url else "make new document"
    else:
        body = f'make new window with properties {{URL:"{safe}"}}' if url else "make new window"
    script = f'tell application "{app}"\n{body}\nactivate\nend tell'
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=2.5)
    except subprocess.TimeoutExpired:
        return f"I couldn't open a window of {browser} on this desktop."
    if out.returncode != 0:
        err = out.stderr.strip()
        if "not authorized" in err.casefold() or "-1743" in err:
            return f"I need permission to control {browser}."
        return f"I couldn't open a window of {browser} on this desktop."
    return None


def _fresh_frame(el: Any) -> tuple[float, float, float, float] | None:
    from .desktop import _multi, _point, _size

    vals = _multi(el, ["AXPosition", "AXSize"])
    if not vals:
        return None
    pos, size = _point(vals[0]), _size(vals[1])
    if not pos or not size or size[0] <= 0 or size[1] <= 0:
        return None
    return (pos[0], pos[1], size[0], size[1])


def _hit_ok(el: Any, frame: tuple[float, float, float, float]) -> bool:
    import ApplicationServices as AS  # type: ignore

    x, y = frame[0] + frame[2] / 2, frame[1] + frame[3] / 2
    try:
        err, hit = AS.AXUIElementCopyElementAtPosition(AS.AXUIElementCreateSystemWide(), x, y, None)
    except Exception:
        return True
    if err != 0 or hit is None or hit == el:
        return True
    cur = hit
    for _ in range(12):
        try:
            err, cur = AS.AXUIElementCopyAttributeValue(cur, "AXParent", None)
        except Exception:
            return True
        if err != 0 or cur is None:
            return False
        if cur == el:
            return True
    return False


def _enable(pid: int) -> Any:
    import ApplicationServices as AS  # type: ignore

    el = AS.AXUIElementCreateApplication(pid)
    for attr in ("AXManualAccessibility", "AXEnhancedUserInterface"):
        try:
            AS.AXUIElementSetAttributeValue(el, attr, True)
        except Exception:
            pass
    return el


def _trusted() -> bool:
    try:
        import ApplicationServices as AS  # type: ignore

        return bool(AS.AXIsProcessTrusted())
    except Exception:
        return False


def _near(a: tuple[float, float, float, float], b: tuple[float, float, float, float], slop: float = 80) -> bool:
    return all(abs(a[i] - b[i]) <= slop for i in range(4))
