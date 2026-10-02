"""Jev intent layer.

One HTTP request per utterance: a speculative fan-out of every question the
executor could need. Jev never generates text; every free-text value (text to
type, search query, domain) is produced as *candidates* in code and Jev selects
the right one (docs: "select instead of generate").
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import actions, config

ACTIONS: dict[str, str] = {
    "open_app": "Launch, open, switch to, or bring up an application program on the Mac (for example Chrome, Cursor, Slack, Finder, Terminal, Notes). Quitting or closing an application is not this. Switching a browser's user profile is switch_profile, not this.",
    "close_app": "Quit or close an application program entirely (for example 'close chrome', 'quit notes', 'exit slack', 'close this app', 'close the current app'). Closing a tab, a window, or a dialog is not this.",
    "open_website": "Go to a website or web page by name or domain, with no search query (for example 'go to youtube', 'open reddit', 'pull up gmail'). Writing or typing a domain into an app is screen, not this. A site that is already one of the open tabs in `tabs` is switch_tab, not this.",
    "web_search": "Search for something on the web or on a specific site: Google it, look it up, find videos of, search YouTube for, search Amazon for. Typing text into a field without searching is screen, not this.",
    "type_text": "Type, write, or dictate into the text area that is already open, with no application named. If `focused_field` is true, what they say is the text to insert into that field, even when they never say 'write', unless they clearly ask for a different action. 'I want you to write good morning' inserts good morning. If they name the app to type in, or a control to click, that is screen.",
    "screen": "Look at an application's own window and either type text into one of its text fields or click one of its controls. Examples: 'write youtube.com on google chrome' (type those characters, do not search and do not open the page), 'type hello in notes', 'click the image on photo booth'. This is one app's visible controls, not a website, not a web search, and not a keyboard shortcut.",
    "new_item": "Create something new inside an app: a new note, document, file, tab, window, message, email, or page (for example 'new note', 'open a new note in the notes app', 'make a new note called groceries', 'new document')",
    "switch_tab": "Switch to a browser tab that is already open on this desktop, in that same browser (for example 'switch to the youtube tab', 'go to the gmail tab', 'the github tab', 'the other tab'). The open tabs are listed in `tabs`. Opening a page that is not already a tab is open_website. Next or previous, with no tab named, is a shortcut. A browser profile is switch_profile, not this.",
    "switch_profile": "Switch a browser to a different user profile when that browser has more than one, listed in `profiles`. Examples: 'switch profile', 'go to another profile', 'the other profile', 'switch to the work profile', 'change brave profile'. A tab is switch_tab. Opening the browser itself is open_app.",
    "shortcut": "Press a single key or keyboard shortcut: enter, escape, tab, copy, paste, undo, save, select all, new tab, close a tab or window, reload, go back, switch app, and similar. Quitting an application is close_app, not a shortcut. Switching to a named open tab is switch_tab, not a shortcut.",
    "scroll": "Scroll the current page or document up or down, to the top or bottom",
    "volume": "Change the system sound volume: louder, quieter, mute, unmute, max",
    "media": "Control music or video playback: play, pause, resume, next track, previous track, skip",
    "screenshot": "Take a screenshot of the screen",
    "open_folder": "Open a folder like Downloads, Desktop, Documents, or the home folder in Finder",
    "system": "System-level action: lock the screen, put the display to sleep, show the desktop, toggle dark mode, empty the trash",
    "task": "A multi-step task that needs looking at the screen and doing several things inside an app or website: fill in a form, find and click something specific, reply to a message, search a site and open a result, change a setting, compose and send an email (for example 'reply to the last email from Sam saying yes', 'find the cheapest flight to London on google flights', 'turn on dark mode in the settings app'). Quitting or closing one application is close_app. Typing one value into a field, or clicking one control, is screen.",
    "stop": "Tell the assistant to stop listening, go to sleep, or shut itself down (for example 'stop listening', 'goodbye'). Quitting another application is close_app.",
    "none": "Not a command for the computer: conversation, thinking aloud, background chatter, or unintelligible. If `focused_field` is true, they are dictating into the open text area, so this is type_text, not conversation.",
}

SHORTCUT_CRITERIA: dict[str, str] = {
    "enter": "press enter / return / submit",
    "escape": "press escape / cancel / dismiss",
    "tab": "press the tab key",
    "space": "press the space bar",
    "backspace": "delete the previous character / backspace",
    "arrow_up": "press the up arrow",
    "arrow_down": "press the down arrow",
    "arrow_left": "press the left arrow",
    "arrow_right": "press the right arrow",
    "copy": "copy the selection",
    "paste": "paste from the clipboard",
    "cut": "cut the selection",
    "undo": "undo the last change",
    "redo": "redo",
    "select_all": "select all / select everything",
    "save": "save the file / document",
    "find": "open find / search within the page or document",
    "new": "create a new file, document, note, or message in the current app",
    "new_tab": "open a new browser tab",
    "close_tab_or_window": "close the current tab or window",
    "reopen_closed_tab": "reopen the last closed tab",
    "quit_app": "press command-q, the quit keyboard shortcut, in whatever is already focused",
    "minimize_window": "minimize the window",
    "hide_app": "hide the current app",
    "fullscreen": "toggle full screen",
    "next_tab": "switch to the next tab, without naming which tab",
    "previous_tab": "switch to the previous tab, without naming which tab",
    "browser_back": "go back to the previous page",
    "browser_forward": "go forward",
    "reload": "reload / refresh the page",
    "address_bar": "focus the browser address bar / URL bar",
    "spotlight": "open Spotlight search",
    "switch_app": "switch to the previous / next application (command-tab)",
    "next_window": "switch to the next window of the current app",
    "delete_word": "delete the previous word",
    "delete_line": "delete the current line / everything before the cursor on this line",
    "zoom_in": "zoom in / make text bigger",
    "zoom_out": "zoom out / make text smaller",
    "bold": "make the selection bold",
    "italic": "make the selection italic",
    "send_message": "send the message (command-enter)",
    "emoji_picker": "open the emoji picker",
}

# regexes that peel the payload text off a spoken command
_TEXT_PATTERNS = [
    r"^(?:please\s+)?(?:i\s+(?:want|need)\s+you\s+to\s+|i(?:'d|\s+would)\s+like\s+you\s+to\s+|can\s+you\s+|could\s+you\s+)?(?:type|write|enter|dictate|input|put|insert|say|send|text|paste)(?:\s+in|\s+out|\s+down|\s+the\s+words?|\s+the\s+text)?[:,]?\s+(?P<t>.+)$",
    r"^(?:please\s+)?(?:can you\s+|could you\s+)?(?:search|google|look\s*up|find|look\s+for|show\s+me|pull\s+up)(?:\s+(?:on|in)\s+\w+(?:\s+\w+)?)?(?:\s+for)?[:,]?\s+(?P<t>.+)$",
    r"^.*?\b(?:for|about|of|on)\s+(?P<t>.+)$",
    r"[\"“'](?P<t>[^\"”']+)[\"”']",
]
_TRIVIAL = {"the", "a", "an", "to", "on", "in", "of", "for", "and", "it", "that", "this"}
_TITLE = re.compile(r"\b(?:called|titled|named|labeled|that says|saying|with the title)\s+(?P<t>.+)$", re.I)
_TRAILING_IN_APP = re.compile(r"\s+(?:in|into|inside|on)\s+(?:the\s+)?(?:[A-Z][\w.]*|notes|chrome|cursor|safari|slack|mail|messages|terminal|finder)(?:\s+app)?\s*[.!?]?$")
# "write youtube.com on google chrome" → the payload is the span before the place.
_TRAILING_PLACE = re.compile(
    r"\s+(?:in|into|inside|on)\s+(?:the\s+)?[a-z0-9][\w.&'-]*(?:\s+[a-z0-9][\w.&'-]*){0,3}(?:\s+app)?\s*$",
    re.I,
)
_TRAILING_SUBMIT = re.compile(
    r"[\s,.]*(?:and|then)?\s*(?:hit|press|and)\s+(?:enter|return|send|submit)\s*[.!]?$", re.I
)
_SPLIT_COMPOUND = re.compile(r"\s*(?:,\s*)?\b(?:and then|then|and also|and)\b\s*", re.I)


def _clean(s: str) -> str:
    s = s.strip().strip('"“”\'')
    s = _TRAILING_SUBMIT.sub("", s).strip()
    return s.rstrip(" .").strip()


def text_candidates(utterance: str) -> dict[str, str]:
    """Candidate spans that might be the payload text. Jev picks; code never guesses."""
    cands: list[str] = []

    def add(t: str) -> None:
        t = _clean(t)
        if t and t not in cands:
            cands.append(t)

    m = _TITLE.search(utterance)
    if m:
        add(m.group("t"))
    for pat in _TEXT_PATTERNS:
        m = re.search(pat, utterance, flags=re.I)
        if m:
            add(m.group("t"))
            add(_TRAILING_IN_APP.sub("", m.group("t")))
            add(_TRAILING_PLACE.sub("", m.group("t")))
    whole = _clean(utterance)
    if whole and whole not in cands:
        cands.append(whole)
    useful = [c for c in cands if c.casefold() not in _TRIVIAL]
    if useful:
        cands = useful
    if not cands:
        cands.append(utterance.strip() or "(nothing)")
    return {f"c{i}": c for i, c in enumerate(cands[:6])}


_DOMAIN = re.compile(r"\b([a-z0-9-]+(?:\.[a-z0-9-]+)+)\b", re.I)
_SITE_WORD = re.compile(
    r"\b(?:go to|open|visit|pull up|bring up|load|navigate to|take me to)\s+(?:the\s+)?(?:website\s+|site\s+)?([a-z0-9][a-z0-9 .-]*?)(?:\s+(?:website|site|dot com|\.com|page|homepage))?\s*[.!?]?$",
    re.I,
)


def domain_guess(utterance: str) -> str | None:
    m = _DOMAIN.search(utterance)
    if m:
        return m.group(1).lower()
    m = _SITE_WORD.search(utterance)
    if m:
        word = m.group(1).strip().lower().replace(" dot ", ".").replace(" ", "")
        if word and word not in ("it", "that", "this"):
            return word if "." in word else f"{word}.com"
    return None


@dataclass
class Plan:
    utterance: str
    action: str
    confidence: float
    args: dict[str, Any] = field(default_factory=dict)
    answers: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0

    def __str__(self) -> str:
        a = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.action}({a})  conf={self.confidence:.2f}  {self.latency_ms}ms"


class Brain:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or config.TYPESAFE_API_KEY
        if not self.api_key:
            raise SystemExit("TYPESAFE_API_KEY is not set (put it in .env)")
        self.model = model or config.JEV_MODEL
        self.http = httpx.Client(
            timeout=15.0,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            http2=False,
        )
        # Keep the TCP+TLS connection warm so the first real command is fast.
        try:
            self.http.get("https://api.typesafe.ai/v1/models")
        except Exception:
            pass

    # ------------------------------------------------------------ questions

    def _questions(self, cands: dict[str, str], apps: list[str], tabs: list | None = None, profiles: list | None = None) -> dict[str, Any]:
        tab_criteria = {t.key: t.label for t in (tabs or [])}
        tab_criteria["none"] = "No listed tab matches, or they want a new page rather than an open tab"
        profile_criteria = {p.key: p.label for p in (profiles or [])}
        profile_criteria["other"] = "Another profile, the other profile, or switch profile, with no profile name"
        profile_criteria["none"] = "They did not ask to change profiles, or the browser has only one"
        q: dict[str, Any] = {
            "action": {
                "type": "choice",
                "instructions": "The user is speaking a voice command to their Mac. `utterance` is the transcript. Which single kind of action are they asking the computer to perform right now?",
                "criteria": ACTIONS,
            },
            "addressed": {
                "type": "noul",
                "instructions": "Is `utterance` an instruction spoken to a voice assistant that controls this computer (open, type, search, scroll, press, play, close, and so on), rather than conversation with another person, a phone call, reading aloud, or thinking out loud?",
                "criteria": {"true": "A direct instruction for the computer to do something now", "false": "Not directed at the computer, or not an instruction"},
            },
            "recommend": {
                "type": "noul",
                "instructions": "Does the user want the assistant to pick, choose, or recommend one specific result for them (for example 'recommend me a car', 'which one should I get', 'find me the best deal', 'pick the cheapest')?",
                "criteria": {"true": "They want one result chosen or recommended", "false": "They only want to find, open, or see results"},
            },
            "message_seller": {
                "type": "noul",
                "instructions": "Does the user want the assistant to contact, message, DM, or reach out to the seller or poster of the result it finds (for example 'and message the seller', 'send them a dm', 'ask if it is available')?",
                "criteria": {"true": "They want a message sent to the seller", "false": "No message is requested"},
            },
            "compound": {
                "type": "noul",
                "instructions": "Does `utterance` ask for two or more separate actions to be performed one after another (for example 'open chrome and go to youtube')? A single action with several words is not compound.",
                "criteria": {"true": "Two or more distinct actions are requested", "false": "Exactly one action is requested"},
            },
            "app": {
                "type": "choice",
                "instructions": "Assume the user wants to open or switch to an application. Which installed application in `apps` do they mean? Match on meaning: 'chrome' means Google Chrome, 'settings' means System Settings, 'browser' means the default browser. Choose `none` if no listed app matches.",
                "criteria": {**{a: None for a in apps}, "none": "No listed application matches what the user said"},
            },
            "close_target": {
                "type": "choice",
                "instructions": "Assume the user wants to quit or close an application entirely. Which installed application in `apps` do they mean? Match on meaning: 'chrome' means Google Chrome, 'notes' means Notes. Choose `frontmost` when they mean the current app, this app, or name no application. Choose `none` if no listed app matches. A tab or window is not an application.",
                "criteria": {**{a: None for a in apps}, "frontmost": "The focused app, this app, or no specific application was named", "none": "No listed application matches what the user said"},
            },
            "tab_target": {
                "type": "choice",
                "instructions": "Assume the user wants to switch to a browser tab already open on this desktop. `tabs` lists those tabs. Choose the one they mean. 'the other tab' means the tab that is not marked current, when the list has one other tab. Choose `none` if no listed tab matches or they want a new page. A browser profile is not a tab.",
                "criteria": tab_criteria,
            },
            "profile_target": {
                "type": "choice",
                "instructions": "Assume the user wants to change the browser's user profile. `profiles` lists profiles only for browsers that have more than one. Choose the profile they named. Choose `other` for another profile, the other profile, or switch profile with no name. Choose `none` when they did not ask to change profiles.",
                "criteria": profile_criteria,
            },
            "site": {
                "type": "choice",
                "instructions": "Assume the user wants to open a website. Which site do they mean? Choose `other` if it is not one of the listed sites.",
                "criteria": {**{s: None for s in actions.SITES}, "other": "A site not in this list"},
            },
            "engine": {
                "type": "choice",
                "instructions": "Assume the user wants to search for something. Which site or search engine should the search run on? If the user does not name a site, choose google.",
                "criteria": {e: None for e in actions.SEARCH_ENGINES},
            },
            "text": {
                "type": "choice",
                "instructions": "Assume the user wants some text typed or searched. `candidates` holds possible payloads cut from the utterance. Which candidate is exactly the text to insert, with no command words? 'I want you to write', 'write', and 'type' are the request, not part of the text. Do not include a trailing 'on youtube', 'in notes', or 'and press enter'. If a text area is already focused and they did not use those request words, the whole utterance is the text.",
                "criteria": {k: v for k, v in cands.items()},
            },
            "in_app": {
                "type": "noul",
                "instructions": "Does the user name a specific application that the action should happen inside of (for example 'in the notes app', 'in chrome', 'in cursor')? Naming an app as the thing to open does not count unless the action is something done inside it.",
                "criteria": {"true": "An application is named as the place where the action happens", "false": "No application is named, or the app is only the thing being opened"},
            },
            "new_kind": {
                "type": "choice",
                "instructions": "Assume the user wants to create something new. What kind of thing?",
                "criteria": {"tab": "a new browser tab", "window": "a new window", "item": "a new note, document, file, message, email, page, or anything else created with the app's New command"},
            },
            "has_title": {
                "type": "noul",
                "instructions": "Assume the user is creating a new note, document, or file. Do they give it a title or initial text (for example 'called groceries', 'titled ideas', 'that says hello')?",
                "criteria": {"true": "A title or initial text is given", "false": "No title or text is given"},
            },
            "submit": {
                "type": "noul",
                "instructions": "After typing the text, does the user also want the enter/return key pressed (they say things like 'and hit enter', 'and send it', 'and search')?",
                "criteria": {"true": "The user explicitly asks to submit, send, or press enter afterwards", "false": "They only want the text typed"},
            },
            "shortcut": {
                "type": "choice",
                "instructions": "Assume the user wants a key or keyboard shortcut pressed. Which one?",
                "criteria": SHORTCUT_CRITERIA,
            },
            "scroll_dir": {
                "type": "choice",
                "instructions": "Assume the user wants to scroll. In which direction?",
                "criteria": {"down": "scroll down / further", "up": "scroll up / back up", "top": "jump to the very top", "bottom": "jump to the very bottom"},
            },
            "scroll_amount": {
                "type": "choice",
                "instructions": "Assume the user wants to scroll up or down. How far?",
                "criteria": {"little": "a little / a bit / a few lines", "page": "a normal amount, about one screen; the default when unspecified", "a_lot": "a lot / way down / far"},
            },
            "volume_op": {
                "type": "choice",
                "instructions": "Assume the user wants to change the volume. What change?",
                "criteria": {"up": "louder / turn it up", "down": "quieter / turn it down", "mute": "mute / silence", "unmute": "unmute / sound back on", "max": "maximum / all the way up", "half": "medium / half volume"},
            },
            "media_op": {
                "type": "choice",
                "instructions": "Assume the user wants to control playback. What?",
                "criteria": {"play_pause": "play, pause, resume, or stop the current track or video", "next": "next track / skip this song", "previous": "previous track / go back a song"},
            },
            "folder": {
                "type": "choice",
                "instructions": "Assume the user wants to open a folder in Finder. Which one?",
                "criteria": {f: None for f in actions.FOLDERS},
            },
            "system_op": {
                "type": "choice",
                "instructions": "Assume the user wants a system-level action. Which one?",
                "criteria": {"lock": "lock the screen", "sleep_display": "put the display / screen to sleep", "show_desktop": "show the desktop", "toggle_dark_mode": "switch between dark and light mode", "empty_trash": "empty the trash"},
            },
        }
        return q

    # ------------------------------------------------------------ inference

    def evaluate(self, utterance: str, front: str | None = None) -> Plan:
        from .profiles import load_profiles
        from .tabs import list_tabs

        apps = actions.installed_apps()
        cands = text_candidates(utterance)
        try:
            open_tabs = list_tabs()
        except Exception:
            open_tabs = []
        try:
            profiles = load_profiles()
        except Exception:
            profiles = []
        try:
            focused = actions.focused_text_field()
        except Exception:
            focused = False
        state = {
            "utterance": utterance,
            "frontmost_app": front if front is not None else actions.frontmost_app(),
            "apps": apps,
            "candidates": cands,
            "tabs": [t.label for t in open_tabs],
            "profiles": [p.label for p in profiles],
            "focused_field": focused,
        }
        payload = {"state": state, "model": self.model, "questions": self._questions(cands, apps, open_tabs, profiles)}
        t0 = time.perf_counter()
        r = self.http.post(config.TYPESAFE_URL, json=payload)
        r.raise_for_status()
        data = r.json()
        ms = int((time.perf_counter() - t0) * 1000)
        return self._to_plan(utterance, data["answers"], cands, ms, open_tabs, profiles)

    def _to_plan(self, utterance: str, ans: dict[str, Any], cands: dict[str, str], ms: int,
                 tabs: list | None = None, profiles: list | None = None) -> Plan:
        act = ans["action"]
        action = act["choice"]
        conf = float(act["confidence"])
        args: dict[str, Any] = {}

        def ch(key: str) -> tuple[str, float]:
            a = ans[key]
            return a["choice"], float(a["confidence"])

        if action == "open_app":
            app, c = ch("app")
            args["app"] = app
            conf = min(conf, c)
        elif action == "close_app":
            app, c = ch("close_target")
            args["app"] = app
            conf = min(conf, c)
        elif action == "switch_profile":
            key, c = ch("profile_target")
            args["profile"] = key
            chosen = next((p for p in (profiles or []) if p.key == key), None)
            if key == "other":
                args["other"] = True
            elif chosen is not None:
                args["browser"] = chosen.browser
                args["name"] = chosen.name
                args["directory"] = chosen.directory
            conf = min(conf, c)
        elif action == "switch_tab":
            key, c = ch("tab_target")
            args["tab"] = key
            chosen = next((t for t in (tabs or []) if t.key == key), None)
            if chosen is not None:
                args["browser"] = chosen.browser
                args["title"] = chosen.title
            conf = min(conf, c)
        elif action == "open_website":
            site, c = ch("site")
            if site != "other":
                args["url"] = actions.SITES[site]
                args["site"] = site
                conf = min(conf, c)
            else:
                dom = domain_guess(utterance)
                if dom:
                    args["url"] = f"https://{dom}"
                    args["site"] = dom
                else:  # nothing to navigate to: fall back to a search
                    action = "web_search"
        if action == "web_search":
            engine, c1 = ch("engine")
            tkey, c2 = ch("text")
            args["engine"] = engine
            args["query"] = cands.get(tkey, utterance)
            conf = min(conf, c2)
        elif action == "type_text":
            tkey, c = ch("text")
            args["text"] = cands.get(tkey, utterance)
            args["submit"] = float(ans["submit"]["noul"]) > config.YES
            conf = min(conf, c)
        elif action == "new_item":
            args["kind"], _ = ch("new_kind")
            if float(ans["has_title"]["noul"]) > config.YES:
                tkey, _ = ch("text")
                args["title"] = cands.get(tkey, "")
        elif action == "shortcut":
            s, c = ch("shortcut")
            args["shortcut"] = s
            conf = min(conf, c)
        elif action == "scroll":
            d, c = ch("scroll_dir")
            a, _ = ch("scroll_amount")
            args["direction"], args["amount"] = d, a
            conf = min(conf, c)
        elif action == "volume":
            op, c = ch("volume_op")
            args["op"] = op
            conf = min(conf, c)
        elif action == "media":
            op, c = ch("media_op")
            args["op"] = op
            conf = min(conf, c)
        elif action == "open_folder":
            f, c = ch("folder")
            args["folder"] = f
            conf = min(conf, c)
        elif action == "system":
            op, c = ch("system_op")
            args["op"] = op
            conf = min(conf, c)

        # "in the notes app": focus that app before acting inside it
        if action in ("new_item", "shortcut", "type_text", "scroll", "open_website", "web_search", "switch_tab", "switch_profile") and float(ans["in_app"]["noul"]) > config.YES:
            app, _ = ch("app")
            if app != "none":
                args["in_app"] = app
        args["compound"] = float(ans["compound"]["noul"]) > config.YES
        args["recommend"] = float(ans["recommend"]["noul"]) > config.YES
        args["message_seller"] = float(ans["message_seller"]["noul"]) > config.YES
        args["addressed"] = round(float(ans["addressed"]["noul"]), 2)
        return Plan(utterance, action, conf, args, ans, ms)


_SUBMIT_ONLY = re.compile(r"^(?:then\s+)?(?:hit|press|and)?\s*(?:enter|return|send|submit)(?:\s+it)?$", re.I)


def split_compound(utterance: str) -> list[str]:
    """Split 'A and then B' into parts. A trailing 'hit enter' is not its own step:
    the type_text plan already carries submit=True."""
    parts = [p.strip(" ,.") for p in _SPLIT_COMPOUND.split(utterance)]
    parts = [p for p in parts if len(p) > 1]
    if len(parts) > 1 and _SUBMIT_ONLY.match(parts[-1]):
        parts = parts[:-1]
        if len(parts) == 1:
            return [utterance]
        parts[-1] = parts[-1] + " and hit enter"
    return parts
