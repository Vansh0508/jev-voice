"""Microphone capture with energy-based voice activity detection and endpointing."""
from __future__ import annotations

import queue
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from . import config

FRAME_MS = 30
FRAME = config.SAMPLE_RATE * FRAME_MS // 1000
PARTIAL_MS = 320


def should_drop(now: float, paused_until: float, muted: bool) -> bool:
    """Frames are thrown away during a pause window or while the assistant is making sound."""
    return muted or now < paused_until


def should_emit_partial(since_ms: int, speech_ms: int, silence_ms: int, partial_ms: int, min_speech_ms: int) -> bool:
    """A partial goes out once enough new voiced audio has arrived, not during the closing silence."""
    return silence_ms == 0 and since_ms >= partial_ms and speech_ms - silence_ms >= min_speech_ms


@dataclass
class VADConfig:
    start_frames: int = 3          # consecutive loud frames to start
    end_silence_ms: int = 550      # silence that ends an utterance
    min_speech_ms: int = 250
    max_speech_ms: int = 12000
    pre_roll_ms: int = 240         # audio kept from before speech start
    threshold_mult: float = 3.5    # loudness over noise floor
    floor_min: float = 0.004


class Listener:
    """Yields float32 16 kHz mono utterances. Call `pause()` while the assistant speaks."""

    def __init__(self, device: int | str | None = None, vad: VADConfig | None = None) -> None:
        self.vad = vad or VADConfig()
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.paused_until = 0.0
        self.noise = 0.01
        self.on_speech_start = None  # optional callback fired when an utterance begins
        self.on_speech_abort = None  # optional callback when a started utterance is thrown away
        self.mute_when: Callable[[], bool] | None = None
        self._interrupted = False
        self.stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME,
            device=device, callback=self._cb,
        )

    def _cb(self, indata, frames, t, status) -> None:  # noqa: ANN001
        self.q.put(indata[:, 0].copy())

    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        self.stream.stop()
        self.stream.close()

    def pause(self, seconds: float) -> None:
        self.paused_until = max(self.paused_until, time.monotonic() + seconds)

    def interrupt(self) -> None:
        """Make a blocked next_utterance return an empty array."""
        self._interrupted = True

    def _muted(self) -> bool:
        if not self.mute_when:
            return False
        try:
            muted = bool(self.mute_when())
        except Exception:
            return False
        if muted:
            self.pause(0.3)  # the room echo outlasts the playback process
        return muted

    def _abort(self) -> None:
        if self.on_speech_abort:
            try:
                self.on_speech_abort()
            except Exception:
                pass

    def drain(self) -> None:
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def next_utterance(self, on_partial: Callable[[np.ndarray], None] | None = None) -> np.ndarray:
        v = self.vad
        pre_n = v.pre_roll_ms // FRAME_MS
        ring: list[np.ndarray] = []
        speech: list[np.ndarray] = []
        loud_run = 0
        silence_ms = 0
        since_partial = 0
        in_speech = False
        while True:
            if self._interrupted:
                return np.zeros(0, dtype=np.float32)
            try:
                frame = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if should_drop(time.monotonic(), self.paused_until, self._muted()):
                if in_speech:
                    self._abort()
                ring.clear(); speech.clear(); in_speech = False; loud_run = 0; since_partial = 0
                continue
            rms = float(np.sqrt(np.mean(frame * frame)) + 1e-9)
            if not in_speech:
                # adaptive noise floor (slow up, fast down)
                self.noise = self.noise * 0.98 + rms * 0.02 if rms > self.noise else self.noise * 0.9 + rms * 0.1
            thresh = max(v.floor_min, self.noise * v.threshold_mult)
            loud = rms > thresh
            if not in_speech:
                ring.append(frame)
                if len(ring) > pre_n:
                    ring.pop(0)
                loud_run = loud_run + 1 if loud else 0
                if loud_run >= v.start_frames:
                    in_speech = True
                    speech = list(ring)
                    silence_ms = 0
                    if self.on_speech_start:
                        try:
                            self.on_speech_start()
                        except Exception:
                            pass
                continue
            speech.append(frame)
            silence_ms = 0 if loud else silence_ms + FRAME_MS
            since_partial += FRAME_MS
            dur = len(speech) * FRAME_MS
            if on_partial and should_emit_partial(since_partial, dur, silence_ms, PARTIAL_MS, v.min_speech_ms):
                since_partial = 0
                try:
                    on_partial(np.concatenate(speech))
                except Exception:
                    pass
            if silence_ms >= v.end_silence_ms or dur >= v.max_speech_ms:
                if dur - silence_ms >= v.min_speech_ms:
                    return np.concatenate(speech)
                self._abort()
                ring.clear(); speech.clear(); in_speech = False; loud_run = 0; since_partial = 0
