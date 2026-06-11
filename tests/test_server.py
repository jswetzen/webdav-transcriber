from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from whisperwebdav.config import Config
from whisperwebdav.server import create_app

FAKE_SEGMENTS = [
    {"start": 0.0, "end": 3.5, "text": "Hello world"},
    {"start": 4.0, "end": 7.25, "text": "How are you"},
]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # The server needs no WebDAV config; patch out the GPU engine entirely.
    monkeypatch.setattr(
        "whisperwebdav.server.transcribe_one",
        lambda audio_path, config, with_timestamps=True: list(FAKE_SEGMENTS),
    )
    return TestClient(create_app(Config()))


def _post(client: TestClient, response_format: str, **extra):
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("note.wav", b"fake-audio-bytes")},
        data={"model": "kb-whisper-large", "response_format": response_format, **extra},
    )


def test_healthz(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_models(client: TestClient) -> None:
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "KBLab/kb-whisper-large"


def test_transcribe_json_default(client: TestClient) -> None:
    resp = _post(client, "json")
    assert resp.status_code == 200
    assert resp.json() == {"text": "Hello world How are you"}


def test_transcribe_text(client: TestClient) -> None:
    resp = _post(client, "text")
    assert resp.status_code == 200
    assert resp.text == "Hello world How are you"


def test_transcribe_srt(client: TestClient) -> None:
    resp = _post(client, "srt")
    assert resp.status_code == 200
    assert "00:00:00,000 --> 00:00:03,500" in resp.text


def test_transcribe_vtt(client: TestClient) -> None:
    resp = _post(client, "vtt")
    assert resp.status_code == 200
    assert resp.text.startswith("WEBVTT")


def test_transcribe_verbose_json(client: TestClient) -> None:
    resp = _post(client, "verbose_json")
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "Hello world How are you"
    assert body["duration"] == 7.25
    assert len(body["segments"]) == 2
    assert body["segments"][0] == {"id": 0, "start": 0.0, "end": 3.5, "text": "Hello world"}


def test_transcribe_bad_format(client: TestClient) -> None:
    resp = _post(client, "xml")
    assert resp.status_code == 400


def test_language_override_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake(audio_path, config, with_timestamps=True):
        captured["language"] = config.language
        return list(FAKE_SEGMENTS)

    monkeypatch.setattr("whisperwebdav.server.transcribe_one", fake)
    c = TestClient(create_app(Config(language="sv")))
    c.post(
        "/v1/audio/transcriptions",
        files={"file": ("n.wav", b"x")},
        data={"response_format": "json", "language": "en"},
    )
    assert captured["language"] == "en"


# --- auth ---


def test_auth_required_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "whisperwebdav.server.transcribe_one",
        lambda *a, **k: list(FAKE_SEGMENTS),
    )
    c = TestClient(create_app(Config(api_key="secret")))

    assert _post(c, "json").status_code == 401

    ok = c.post(
        "/v1/audio/transcriptions",
        files={"file": ("n.wav", b"x")},
        data={"response_format": "json"},
        headers={"Authorization": "Bearer secret"},
    )
    assert ok.status_code == 200
