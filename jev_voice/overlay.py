"""Dynamic Island for the Mac notch.

The resting shape is the notch itself: same black, same width, flush with the
top of the built-in screen. Speech, transcripts, and results grow it downward
into a rounded island. Borderless and non-activating, so it never steals focus
from the app being controlled. All UI work happens on the main thread; call
`set(...)` from any thread.

States: idle · listening · heard · thinking · done · error
"""
from __future__ import annotations

import threading
import warnings
from typing import Callable

import objc  # type: ignore

warnings.filterwarnings("ignore", category=objc.ObjCPointerWarning)

from AppKit import (  # type: ignore
    NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered, NSColor, NSFont,
    NSFontAttributeName, NSMakeRect, NSPanel, NSScreen, NSTextField, NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary, NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
    NSStatusWindowLevel,
)
from Foundation import NSString  # type: ignore
from Quartz import CABasicAnimation  # type: ignore
from PyObjCTools import AppHelper  # type: ignore

# Bottom corners only, so the shape stays fused to the top edge of the screen.
_BOTTOM_CORNERS = 1 | 2  # minXMinY | maxXMinY in AppKit's bottom-left space

COLORS = {
    "idle": (0.62, 0.62, 0.66),
    "listening": (0.98, 0.32, 0.34),
    "heard": (0.98, 0.78, 0.28),
    "thinking": (0.40, 0.62, 1.00),
    "done": (0.32, 0.86, 0.48),
    "error": (1.00, 0.46, 0.28),
}
IDLE_TEXT = "Listening"
BODY = 46.0
MAX_W = 480.0
_MERGED = {"", "listening"}


def _notch_black():
    """Black that stays black on a transparent panel. Calibrated black is lifted to gray, so it misses the notch."""
    return NSColor.colorWithGenericGamma22White_alpha_(0.0, 1.0)


def island_size(state: str, text: str, text_width: float, notch_w: float, notch_h: float) -> tuple[float, float, bool]:
    """Width, height, and whether the island is merged into the notch."""
    if state == "idle" and text.strip().lower() in _MERGED:
        return notch_w, notch_h, True
    width = min(MAX_W, max(notch_w + 36.0, text_width + 68.0))
    return width, notch_h + BODY, False


def notch_metrics(screen) -> tuple[float, float, float, float]:
    """Notch origin x, screen-top y, width, and height. A screen without a notch gets a compact stand-in."""
    frame = screen.frame()
    top = frame.origin.y + frame.size.height
    left = screen.auxiliaryTopLeftArea()
    right = screen.auxiliaryTopRightArea()
    gap = right.origin.x - (left.origin.x + left.size.width)
    if left.size.width > 1 and right.size.width > 1 and gap > 40:
        return left.origin.x + left.size.width, top, gap, left.size.height
    visible = screen.visibleFrame()
    menu = top - (visible.origin.y + visible.size.height)
    width = 180.0
    return frame.origin.x + (frame.size.width - width) / 2, top, width, max(menu, 32.0)


def _screen():
    for screen in NSScreen.screens() or []:
        left = screen.auxiliaryTopLeftArea()
        right = screen.auxiliaryTopRightArea()
        if right.origin.x - (left.origin.x + left.size.width) > 40:
            return screen
    screens = NSScreen.screens()
    return NSScreen.mainScreen() or (screens[0] if screens else None)


class Overlay:
    def __init__(self) -> None:
        self.panel = None
        self.label = None
        self.dot = None
        self._root = None
        self._gen = 0
        self._lock = threading.Lock()
        self._animate = False
        self._glide_gen = 0

    def _build(self) -> None:
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        screen = _screen()
        nx, top, nw, nh = notch_metrics(screen) if screen is not None else (0.0, 900.0, 180.0, 32.0)
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(nx, top - nh, nw, nh), style, NSBackingStoreBuffered, False)
        p = self.panel
        p.setLevel_(NSStatusWindowLevel)
        p.setOpaque_(False)
        p.setBackgroundColor_(NSColor.clearColor())
        p.setHasShadow_(False)
        p.setAlphaValue_(1.0)
        p.setHidesOnDeactivate_(False)
        p.setIgnoresMouseEvents_(True)
        p.setMovable_(False)
        p.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorIgnoresCycle
        )

        root = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, nw, nh))
        root.setWantsLayer_(True)
        layer = root.layer()
        layer.setBackgroundColor_(_notch_black().CGColor())
        layer.setCornerRadius_(10.0)
        layer.setCornerCurve_("continuous")
        layer.setMaskedCorners_(_BOTTOM_CORNERS)
        layer.setMasksToBounds_(True)
        layer.setAllowsEdgeAntialiasing_(False)
        p.setContentView_(root)
        self._root = root

        self.dot = NSView.alloc().initWithFrame_(NSMakeRect(18, 16, 9, 9))
        self.dot.setWantsLayer_(True)
        self.dot.layer().setCornerRadius_(4.5)
        self.dot.setAlphaValue_(0.0)
        root.addSubview_(self.dot)

        self.label = NSTextField.alloc().initWithFrame_(NSMakeRect(36, 12, nw - 56, 20))
        lb = self.label
        lb.setBezeled_(False); lb.setDrawsBackground_(False); lb.setEditable_(False); lb.setSelectable_(False)
        lb.setFont_(NSFont.systemFontOfSize_weight_(14.0, 0.52))
        lb.setTextColor_(NSColor.whiteColor())
        lb.setLineBreakMode_(4)  # truncate tail
        lb.cell().setUsesSingleLineMode_(True)
        lb.setAlphaValue_(0.0)
        root.addSubview_(lb)

        self._apply("idle", IDLE_TEXT)
        p.orderFrontRegardless()
        self._animate = True

    def _apply(self, state: str, text: str) -> None:
        screen = _screen()
        if screen is None or self.panel is None:
            return
        nx, top, nw, nh = notch_metrics(screen)
        r, g, b = COLORS.get(state, COLORS["idle"])
        self.dot.layer().setBackgroundColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, 1).CGColor())
        self.label.setStringValue_(text)
        text_w = float(NSString.stringWithString_(text).sizeWithAttributes_({NSFontAttributeName: self.label.font()}).width)
        width, height, merged = island_size(state, text, text_w, nw, nh)
        cx = nx + nw / 2
        rect = NSMakeRect(cx - width / 2, top - height, width, height)
        body_y = (BODY - 20.0) / 2
        self.label.setFrame_(NSMakeRect(36, body_y, max(40.0, width - 56), 20))
        self.dot.setFrame_(NSMakeRect(18, (BODY - 9.0) / 2, 9, 9))
        self._pulse(state == "listening" and not merged)

        radius = 10.0 if merged else 20.0
        show = 0.0 if merged else 1.0
        self._root.layer().setBackgroundColor_(_notch_black().CGColor())
        self._root.layer().setShadowOpacity_(0.0)
        self._glide(rect, radius, show)

    def _glide(self, rect, radius: float, show: float) -> None:
        """Grow or fold the window. AppKit's animator proxy never commits this panel's frame."""
        root = self._root
        if not self._animate:
            self.panel.setFrame_display_(rect, True)
            root.layer().setCornerRadius_(radius)
            self.label.setAlphaValue_(show)
            self.dot.setAlphaValue_(show)
            return
        start = self.panel.frame()
        r0 = float(root.layer().cornerRadius())
        a0 = float(self.label.alphaValue())
        steps = 12
        self._glide_gen += 1
        token = self._glide_gen

        def step(i: int) -> None:
            if token != self._glide_gen:
                return
            e = 1 - (1 - i / steps) ** 3
            frame = NSMakeRect(
                start.origin.x + (rect.origin.x - start.origin.x) * e,
                start.origin.y + (rect.origin.y - start.origin.y) * e,
                start.size.width + (rect.size.width - start.size.width) * e,
                start.size.height + (rect.size.height - start.size.height) * e,
            )
            self.panel.setFrame_display_(frame, True)
            root.layer().setCornerRadius_(r0 + (radius - r0) * e)
            alpha = a0 + (show - a0) * e
            self.label.setAlphaValue_(alpha)
            self.dot.setAlphaValue_(alpha)
            if i < steps:
                AppHelper.callLater(0.02, lambda i=i: step(i + 1))

        step(1)

    def _pulse(self, on: bool) -> None:
        layer = self.dot.layer()
        if not on:
            layer.removeAnimationForKey_("pulse")
            layer.setOpacity_(1.0)
            return
        if layer.animationForKey_("pulse") is not None:
            return
        anim = CABasicAnimation.animationWithKeyPath_("opacity")
        anim.setFromValue_(1.0)
        anim.setToValue_(0.25)
        anim.setDuration_(0.7)
        anim.setAutoreverses_(True)
        anim.setRepeatCount_(1e6)
        layer.addAnimation_forKey_(anim, "pulse")

    def set(self, state: str, text: str, revert_after: float | None = None) -> None:
        """Show a state. If revert_after is set, fold back into the notch after that many seconds
        unless something newer was shown in the meantime."""
        with self._lock:
            self._gen += 1
            gen = self._gen

        def _do() -> None:
            self._apply(state, text)
            if revert_after:
                def _revert() -> None:
                    if self._gen == gen:
                        self._apply("idle", IDLE_TEXT)
                AppHelper.callLater(revert_after, _revert)

        AppHelper.callAfter(_do)

    def run(self, worker: Callable[[], None]) -> None:
        """Build the island, run `worker` on a background thread, and pump the Cocoa
        event loop on this (main) thread until the worker finishes."""
        self._build()

        def _thread() -> None:
            try:
                worker()
            finally:
                AppHelper.callAfter(AppHelper.stopEventLoop)

        threading.Thread(target=_thread, daemon=True, name="jev-worker").start()
        AppHelper.runEventLoop(installInterrupt=True)


class NullOverlay:
    def set(self, state: str, text: str, revert_after: float | None = None) -> None:
        pass

    def run(self, worker: Callable[[], None]) -> None:
        worker()
