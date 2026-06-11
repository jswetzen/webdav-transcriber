from __future__ import annotations

import pytest

from whisperwebdav.client import transcribe_remote
from whisperwebdav.config import Config

SEGMENTS = [{"start": 0.0, "end": 1.0, "text": "hi"}]


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def test_transcribe_remote_returns_segments(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "note.wav"
    audio.write_bytes(b"audio")
    captured = {}

    def fake_post(url, data=None, files=None, headers=None, timeout=None):
        captured.update(url=url, data=data, headers=headers, files=files)
        return _FakeResponse({"segments": SEGMENTS})

    monkeypatch.setattr("whisperwebdav.client.httpx.post", fake_post)

    config = Config(transcribe_backend="http", transcribe_server_url="http://srv:8000/")
    segments = transcribe_remote(str(audio), config)

    assert segments == SEGMENTS
    # trailing slash on the base URL must not double up
    assert captured["url"] == "http://srv:8000/v1/audio/transcriptions"
    assert captured["data"]["response_format"] == "verbose_json"
    assert captured["data"]["language"] == "sv"
    assert "file" in captured["files"]
    assert captured["headers"] == {}  # no api_key configured


def test_transcribe_remote_sends_api_key(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "note.wav"
    audio.write_bytes(b"audio")
    captured = {}

    def fake_post(url, data=None, files=None, headers=None, timeout=None):
        captured.update(headers=headers)
        return _FakeResponse({"segments": SEGMENTS})

    monkeypatch.setattr("whisperwebdav.client.httpx.post", fake_post)

    config = Config(
        transcribe_backend="http",
        transcribe_server_url="http://srv:8000",
        api_key="secret",
    )
    transcribe_remote(str(audio), config)
    assert captured["headers"] == {"Authorization": "Bearer secret"}
