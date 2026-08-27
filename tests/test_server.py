from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from whisperwebdav.config import Config
from whisperwebdav.server import create_app
from whisperwebdav.tts import SynthesisResult, TTSError

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


# --- text-to-speech ---


@pytest.fixture
def tts_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(
        "whisperwebdav.server.kokoro_files_present", lambda config: True
    )
    monkeypatch.setattr(
        "whisperwebdav.server.available_voices",
        lambda config: frozenset({"af_sarah", "af_sky"}),
    )
    return TestClient(create_app(Config()))


def test_speech_default_wav(monkeypatch: pytest.MonkeyPatch, tts_client: TestClient) -> None:
    captured = {}

    def fake_synthesize(text, config, *, voice="", response_format="wav"):
        captured["text"] = text
        captured["voice"] = voice
        captured["response_format"] = response_format
        return SynthesisResult(audio=b"RIFF-fake-wav", media_type="audio/wav", duration_seconds=1.5)

    monkeypatch.setattr("whisperwebdav.server.synthesize", fake_synthesize)

    resp = tts_client.post("/v1/audio/speech", json={"input": "Hello there"})
    assert resp.status_code == 200
    assert resp.content == b"RIFF-fake-wav"
    assert resp.headers["content-type"] == "audio/wav"
    assert captured == {"text": "Hello there", "voice": "", "response_format": "wav"}


def test_speech_mp3_and_voice_passed_through(
    monkeypatch: pytest.MonkeyPatch, tts_client: TestClient
) -> None:
    monkeypatch.setattr(
        "whisperwebdav.server.synthesize",
        lambda text, config, *, voice="", response_format="wav": SynthesisResult(
            audio=b"id3-fake-mp3", media_type="audio/mpeg", duration_seconds=1.5
        ),
    )
    resp = tts_client.post(
        "/v1/audio/speech",
        json={"input": "Hej", "voice": "af_sky", "response_format": "mp3"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"


def test_speech_empty_input_rejected(tts_client: TestClient) -> None:
    resp = tts_client.post("/v1/audio/speech", json={"input": "   "})
    assert resp.status_code == 400


def test_speech_input_too_long_rejected(tts_client: TestClient) -> None:
    c = TestClient(create_app(Config(tts_max_input_chars=10)))
    resp = c.post("/v1/audio/speech", json={"input": "x" * 11})
    assert resp.status_code == 400


def test_speech_tts_error_returns_400(
    monkeypatch: pytest.MonkeyPatch, tts_client: TestClient
) -> None:
    def raising(text, config, *, voice="", response_format="wav"):
        raise TTSError("Unknown voice 'bogus'")

    monkeypatch.setattr("whisperwebdav.server.synthesize", raising)
    resp = tts_client.post("/v1/audio/speech", json={"input": "hi", "voice": "bogus"})
    assert resp.status_code == 400
    assert "bogus" in resp.json()["detail"]


def test_speech_auth_required_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "whisperwebdav.server.kokoro_files_present", lambda config: True
    )
    monkeypatch.setattr(
        "whisperwebdav.server.available_voices", lambda config: frozenset({"af_sarah"})
    )
    monkeypatch.setattr(
        "whisperwebdav.server.synthesize",
        lambda text, config, *, voice="", response_format="wav": SynthesisResult(
            audio=b"x", media_type="audio/wav", duration_seconds=0.1
        ),
    )
    c = TestClient(create_app(Config(api_key="secret")))
    assert c.post("/v1/audio/speech", json={"input": "hi"}).status_code == 401
    ok = c.post(
        "/v1/audio/speech",
        json={"input": "hi"},
        headers={"Authorization": "Bearer secret"},
    )
    assert ok.status_code == 200


def test_speech_stream_mp3(monkeypatch: pytest.MonkeyPatch, tts_client: TestClient) -> None:
    async def fake_stream(text, config, *, voice="", response_format="mp3"):
        for chunk in (b"id3-", b"chunk-", b"one"):
            yield chunk

    monkeypatch.setattr("whisperwebdav.server.synthesize_stream", fake_stream)

    resp = tts_client.post("/v1/audio/speech", json={"input": "Hello there", "stream": True})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content == b"id3-chunk-one"


def test_speech_stream_rejects_wav(tts_client: TestClient) -> None:
    resp = tts_client.post(
        "/v1/audio/speech",
        json={"input": "hi", "stream": True, "response_format": "wav"},
    )
    assert resp.status_code == 400
    assert "mp3" in resp.json()["detail"]


def test_speech_stream_tts_error_returns_400(
    monkeypatch: pytest.MonkeyPatch, tts_client: TestClient
) -> None:
    async def raising(text, config, *, voice="", response_format="mp3"):
        raise TTSError("Unknown voice 'bogus'")
        yield b""  # pragma: no cover - never reached; makes this an async generator

    monkeypatch.setattr("whisperwebdav.server.synthesize_stream", raising)

    resp = tts_client.post(
        "/v1/audio/speech", json={"input": "hi", "voice": "bogus", "stream": True}
    )
    assert resp.status_code == 400
    assert "bogus" in resp.json()["detail"]


def test_speech_pcm_streams_without_stream_flag(
    monkeypatch: pytest.MonkeyPatch, tts_client: TestClient
) -> None:
    # Regression case: the customtts Firefox extension (and other OpenAI-compatible clients)
    # never sets `stream`, only response_format="pcm" -- that alone must trigger streaming.
    captured = {}

    async def fake_stream(text, config, *, voice="", response_format="mp3"):
        captured["response_format"] = response_format
        for chunk in (b"\x01\x00", b"\x02\x00"):
            yield chunk

    monkeypatch.setattr("whisperwebdav.server.synthesize_stream", fake_stream)

    resp = tts_client.post(
        "/v1/audio/speech", json={"input": "hi", "response_format": "pcm"}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/pcm"
    assert resp.content == b"\x01\x00\x02\x00"
    assert captured["response_format"] == "pcm"


def test_speech_stream_empty_generator_returns_empty_body(
    monkeypatch: pytest.MonkeyPatch, tts_client: TestClient
) -> None:
    async def empty(text, config, *, voice="", response_format="mp3"):
        return
        yield b""  # pragma: no cover - never reached; makes this an async generator

    monkeypatch.setattr("whisperwebdav.server.synthesize_stream", empty)

    resp = tts_client.post("/v1/audio/speech", json={"input": "hi", "stream": True})
    assert resp.status_code == 200
    assert resp.content == b""


def test_list_models_includes_kokoro_when_files_present(
    monkeypatch: pytest.MonkeyPatch, tts_client: TestClient
) -> None:
    resp = tts_client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert {"id": "kokoro", "object": "model", "owned_by": "kokoro-onnx"} in body["data"]
    assert body["voices"] == ["af_sarah", "af_sky"]


def test_list_models_omits_kokoro_when_files_absent(client: TestClient) -> None:
    # `client` fixture uses a stock Config; kokoro_files_present() will be False since no
    # model files exist at the default paths in the test environment.
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert all(m["id"] != "kokoro" for m in body["data"])
    assert body["voices"] == []
