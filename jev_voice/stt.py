"""Speech to text via whisper.cpp's `whisper-server` (Metal accelerated, model stays loaded)."""
from __future__ import annotations

import io
import os
import shutil
import signal
import subprocess
import threading
import time
import wave
from collections.abc import Callable

import httpx
import numpy as np

from . import config

_BLANK = {"", "[BLANK_AUDIO]", "(silence)", "[silence]", "[inaudible]", "[Music]", "[MUSIC]", "(music)"}
# Whisper hallucinates these on near-silent audio. A prompt of example commands is not used:
# whisper repeats that prompt on noise, and the assistant would open Chrome or YouTube unasked.
_HALLUCINATIONS = {"thank you.", "thanks for watching.", "thank you for watching.", "you", "bye.", "thank you"}
_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin"


def _wav_bytes(pcm: np.ndarray, rate: int = config.SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def _prepare(pcm: np.ndarray) -> np.ndarray:
    """Lift quiet speech so a soft utterance stays intelligible."""
    audio = np.asarray(pcm, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if 0.02 < peak < 0.6:
        audio = audio * (0.85 / peak)
    return audio


class WhisperServer:
    def __init__(self, port: int = config.WHISPER_PORT) -> None:
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        self.http = httpx.Client(timeout=30.0)

    def _alive(self) -> bool:
        try:
            r = self.http.get(self.url + "/", timeout=1.0)
            return r.status_code < 500
        except Exception:
            return False

    def _command_matches(self) -> bool:
        """A server already on this port may still be the old base.en model."""
        try:
            pids = subprocess.check_output(
                ["lsof", "-t", f"-iTCP:{self.port}", "-sTCP:LISTEN"], text=True, stderr=subprocess.DEVNULL,
            ).split()
        except Exception:
            return False
        want = str(config.WHISPER_MODEL)
        for pid in pids:
            if not pid.isdigit():
                continue
            try:
                args = subprocess.check_output(["ps", "-p", pid, "-o", "args="], text=True)
            except Exception:
                return False
            if want not in args or f"-bs {config.WHISPER_BEAM}" not in args or "--prompt" in args:
                return False
        return bool(pids)

    def _free_port(self) -> None:
        try:
            pids = subprocess.check_output(
                ["lsof", "-t", f"-iTCP:{self.port}", "-sTCP:LISTEN"], text=True, stderr=subprocess.DEVNULL,
            ).split()
        except Exception:
            return
        for pid in pids:
            if pid.isdigit() and int(pid) != os.getpid():
                os.kill(int(pid), signal.SIGTERM)
        for _ in range(40):
            if not self._alive():
                return
            time.sleep(0.05)

    def start(self) -> None:
        if self._alive() and self._command_matches():
            return
        if self._alive():
            self._free_port()
        exe = shutil.which("whisper-server")
        if not exe:
            raise SystemExit("whisper-server not found: brew install whisper-cpp")
        if not config.WHISPER_MODEL.exists():
            raise SystemExit(f"Whisper model missing: {config.WHISPER_MODEL}\n"
                             f"  curl -L -o models/ggml-small.en.bin {_MODEL_URL}")
        self.proc = subprocess.Popen(
            [exe, "-m", str(config.WHISPER_MODEL), "--host", "127.0.0.1", "--port", str(self.port),
             "-t", str(config.WHISPER_THREADS), "-l", "en", "-nt", "-sns",
             "-bs", str(config.WHISPER_BEAM)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(600):
            if self._alive():
                self._warm_up()
                return
            time.sleep(0.05)
        raise SystemExit("whisper-server failed to start")

    def _warm_up(self) -> None:
        """First inference compiles Metal shaders (~2 s); pay that before the user speaks."""
        tone = 0.05 * np.sin(np.linspace(0, 2 * np.pi * 220 * 0.6, int(config.SAMPLE_RATE * 0.6)))
        try:
            self.transcribe(tone.astype(np.float32))
        except Exception:
            pass

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def transcribe(self, pcm: np.ndarray) -> str:
        files = {"file": ("audio.wav", _wav_bytes(_prepare(pcm)), "audio/wav")}
        data = {"response_format": "json", "temperature": "0.0", "no_timestamps": "true",
                "language": "en"}
        r = self.http.post(self.url + "/inference", files=files, data=data)
        r.raise_for_status()
        text = (r.json().get("text") or "").strip()
        if text in _BLANK or text.lower() in _HALLUCINATIONS or len(text) < 2:
            return ""
        return text


class LiveTranscriber:
    """Transcribe audio while it is still being spoken.

    Each push replaces a clip that whisper has not started yet, so a slow
    inference stays on the newest audio instead of a queue of stale ones.
    """

    def __init__(self, server: WhisperServer, on_text: Callable[[str], None]) -> None:
        self.server = server
        self.on_text = on_text
        self._lock = threading.Lock()
        self._pending: np.ndarray | None = None
        self._text = ""
        self._covered = 0
        self._busy = False
        self._closing = False
        self._event = threading.Event()
        self._thread: threading.Thread | None = None

    def push(self, pcm: np.ndarray) -> None:
        self._ensure()
        with self._lock:
            self._pending = pcm
        self._event.set()

    def _ensure(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="jev-stt", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            self._event.wait()
            with self._lock:
                pcm = self._pending
                self._pending = None
                self._event.clear()
                if pcm is None:
                    if self._closing:
                        return
                    continue
                self._busy = True
            try:
                text = self.server.transcribe(pcm)
            except Exception:
                text = ""
            with self._lock:
                self._busy = False
                self._text = text
                self._covered = len(pcm)
                closing = self._closing
                pending = self._pending is not None
            if text and not closing:
                try:
                    self.on_text(text)
                except Exception:
                    pass
            if closing and not pending:
                return

    def finish(self, pcm: np.ndarray) -> str:
        """Return the transcript of `pcm`, reusing a partial that already covers it."""
        self._ensure()
        with self._lock:
            ready = not self._busy and self._pending is None and self._covered == len(pcm) and bool(self._text)
            if ready:
                self._closing = True
            thread = self._thread
        if ready:
            self._event.set()
            assert thread is not None
            thread.join(timeout=5)
            return self._text
        self.push(pcm)
        with self._lock:
            self._closing = True
        self._event.set()
        assert self._thread is not None
        self._thread.join(timeout=60)
        return self._text

    def cancel(self) -> None:
        with self._lock:
            self._closing = True
            self._pending = None
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=60)
