"""Offline checks for the speech-to-text request. No microphone and no whisper-server."""

import threading
import time
from unittest.mock import Mock

import numpy as np

from jev_voice import stt
from jev_voice.audio import should_emit_partial
from jev_voice.stt import LiveTranscriber, WhisperServer


def test_quiet_speech_is_boosted():
    out = stt._prepare(np.full(1600, 0.05, dtype=np.float32))
    assert float(np.max(np.abs(out))) == np.float32(0.85)


def test_loud_speech_is_left_alone():
    pcm = np.full(1600, 0.9, dtype=np.float32)
    assert float(np.max(np.abs(stt._prepare(pcm)))) == float(pcm[0])


def test_transcribe_does_not_send_a_prompt():
    server = WhisperServer()
    response = Mock()
    response.json.return_value = {"text": "open chrome"}
    response.raise_for_status = Mock()
    server.http = Mock()
    server.http.post.return_value = response
    assert server.transcribe(np.zeros(16000, dtype=np.float32)) == "open chrome"
    data = server.http.post.call_args.kwargs["data"]
    assert "prompt" not in data


def test_partials_wait_for_voiced_audio():
    assert should_emit_partial(320, 320, 0, 320, 250)
    assert not should_emit_partial(300, 300, 0, 320, 250)
    assert not should_emit_partial(320, 400, 90, 320, 250)


def test_live_transcriber_keeps_only_the_newest_clip():
    started = threading.Event()
    release = threading.Event()
    seen: list[int] = []

    def transcribe(pcm: np.ndarray) -> str:
        seen.append(len(pcm))
        if len(pcm) == 1000:
            started.set()
            release.wait(2)
        return f"n{len(pcm)}"

    server = Mock()
    server.transcribe = transcribe
    published: list[str] = []
    live = LiveTranscriber(server, published.append)
    live.push(np.zeros(1000, dtype=np.float32))
    assert started.wait(2)
    live.push(np.zeros(2000, dtype=np.float32))
    live.push(np.zeros(3000, dtype=np.float32))
    release.set()
    deadline = time.monotonic() + 2
    while published != ["n1000", "n3000"] and time.monotonic() < deadline:
        time.sleep(0.01)
    final = live.finish(np.zeros(3000, dtype=np.float32))
    assert seen == [1000, 3000]
    assert published == ["n1000", "n3000"]
    assert final == "n3000"
