from __future__ import annotations

import asyncio
import hmac
from tempfile import SpooledTemporaryFile

import pytest
from fastapi.testclient import TestClient

from whisperwebdav.config import Config
from whisperwebdav.server import ApiKeyMiddleware, BodyLimitMiddleware, create_app
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


# --- hardening: auth middleware, upload cap, no docs ---


def _keyed_client(monkeypatch: pytest.MonkeyPatch, **cfg) -> TestClient:
    monkeypatch.setattr("whisperwebdav.server.kokoro_files_present", lambda config: False)
    monkeypatch.setattr(
        "whisperwebdav.server.transcribe_one",
        lambda *a, **k: list(FAKE_SEGMENTS),
    )
    return TestClient(create_app(Config(**cfg)))


def test_models_requires_key_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _keyed_client(monkeypatch, api_key="secret")
    assert c.get("/v1/models").status_code == 401
    ok = c.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
    assert ok.json()["data"][0]["id"] == "KBLab/kb-whisper-large"


def test_models_open_when_no_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _keyed_client(monkeypatch).get("/v1/models").status_code == 200


def test_healthz_open_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _keyed_client(monkeypatch, api_key="secret").get("/healthz")
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": "secret"}]
)
def test_bad_or_missing_key_gets_401_with_challenge(
    monkeypatch: pytest.MonkeyPatch, headers: dict
) -> None:
    c = _keyed_client(monkeypatch, api_key="secret")
    for resp in (
        c.get("/v1/models", headers=headers),
        c.post("/v1/audio/speech", json={"input": "hi"}, headers=headers),
        c.post("/v1/audio/transcriptions", files={"file": ("n.wav", b"x")}, headers=headers),
    ):
        assert resp.status_code == 401
        assert resp.json() == {"detail": "Invalid API key"}
        assert resp.headers["www-authenticate"] == "Bearer"


def test_unauthenticated_upload_rejected_without_reading_body() -> None:
    reached = []

    async def inner(scope, receive, send):  # pragma: no cover - must not be reached
        reached.append(True)

    received = []

    async def receive():  # pragma: no cover - must not be called
        received.append(True)
        return {"type": "http.request", "body": b"x" * 1024, "more_body": True}

    sent = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/audio/transcriptions",
        "headers": [(b"content-type", b"multipart/form-data; boundary=x")],
    }
    asyncio.run(ApiKeyMiddleware(inner, "secret")(scope, receive, send))

    assert not received and not reached
    assert sent[0]["status"] == 401
    assert (b"www-authenticate", b"Bearer") in sent[0]["headers"]


def test_unauthenticated_large_upload_never_reaches_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("transcribe_one reached without auth")

    c = _keyed_client(monkeypatch, api_key="secret")
    monkeypatch.setattr("whisperwebdav.server.transcribe_one", boom)
    resp = c.post(
        "/v1/audio/transcriptions",
        files={"file": ("big.wav", b"\0" * (5 * 1024 * 1024))},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


def test_upload_over_content_length_cap_is_413(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _keyed_client(monkeypatch, max_upload_bytes=1000)
    resp = c.post("/v1/audio/transcriptions", files={"file": ("n.wav", b"\0" * 5000)})
    assert resp.status_code == 413
    assert "detail" in resp.json()


def test_chunked_upload_over_cap_is_413_and_stops_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _keyed_client(monkeypatch, max_upload_bytes=1000)
    boundary = "xBOUNDARYx"

    def body():
        yield (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="n.wav"\r\n\r\n'
        ).encode()
        for _ in range(100):
            yield b"\0" * 500
        yield f"\r\n--{boundary}--\r\n".encode()

    # A generator body makes httpx send Transfer-Encoding: chunked with no Content-Length.
    resp = c.post(
        "/v1/audio/transcriptions",
        content=body(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert resp.status_code == 413
    assert resp.json()["detail"].startswith("Request body exceeds")


def test_upload_under_cap_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _keyed_client(monkeypatch, max_upload_bytes=1000, api_key="secret")
    resp = c.post(
        "/v1/audio/transcriptions",
        files={"file": ("n.wav", b"\0" * 200)},
        headers={"Authorization": "Bearer secret"},
    )
    assert resp.status_code == 200


def test_speech_body_over_cap_is_413(tts_client: TestClient) -> None:
    resp = tts_client.post("/v1/audio/speech", json={"input": "a" * (2 * 1024 * 1024)})
    assert resp.status_code == 413


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_interactive_docs_disabled(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404


def test_streaming_speech_works_through_middleware_with_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("whisperwebdav.server.kokoro_files_present", lambda config: True)
    monkeypatch.setattr(
        "whisperwebdav.server.available_voices", lambda config: frozenset({"af_sarah"})
    )

    async def fake_stream(text, config, *, voice="", response_format="mp3"):
        for chunk in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
            yield chunk

    monkeypatch.setattr("whisperwebdav.server.synthesize_stream", fake_stream)
    c = TestClient(create_app(Config(api_key="secret")))
    with c.stream(
        "POST",
        "/v1/audio/speech",
        json={"input": "hi", "response_format": "pcm"},
        headers={"Authorization": "Bearer secret"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/pcm"
        chunks = list(resp.iter_bytes())
    assert b"".join(chunks) == b"\x01\x00\x02\x00\x03\x00"


# --- hardening, round 2: mutation-proofing and deny-by-default ---


def _asgi_scope(path: str, *, method: str = "POST", root_path: str = "", headers=()) -> dict:
    return {
        "type": "http",
        "method": method,
        "path": path,
        "root_path": root_path,
        "query_string": b"",
        "headers": list(headers),
    }


async def _noop_send(message) -> None:
    pass


def test_deny_by_default_except_healthz(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _keyed_client(monkeypatch, api_key="secret")
    assert c.get("/healthz").status_code == 200
    assert c.get("/anything-else").status_code == 401
    assert c.post("/healthz").status_code == 401 or c.post("/healthz").status_code == 405
    assert c.get("/anything-else", headers={"Authorization": "Bearer secret"}).status_code == 404


def test_auth_matches_routed_path_under_root_path() -> None:
    app = ApiKeyMiddleware(None, "secret")  # type: ignore[arg-type]
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():  # pragma: no cover - must not be called
        raise AssertionError("body read before auth")

    asyncio.run(app(_asgi_scope("/api/v1/models", method="GET", root_path="/api"), receive, send))
    assert sent[0]["status"] == 401


def test_oversize_unauthenticated_upload_under_root_path_is_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _keyed_client(monkeypatch, api_key="secret", max_upload_bytes=1000).app
    sent = []
    received = []

    async def send(message):
        sent.append(message)

    async def receive():  # pragma: no cover - must not be called
        received.append(True)
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = _asgi_scope(
        "/api/v1/audio/transcriptions",
        root_path="/api",
        headers=[(b"content-length", b"999999"), (b"content-type", b"multipart/form-data")],
    )
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 401
    assert not received


def test_body_limit_uses_routed_path_under_root_path() -> None:
    reached = []

    async def inner(scope, receive, send):  # pragma: no cover - must not be reached
        reached.append(True)

    sent = []

    async def send(message):
        sent.append(message)

    scope = _asgi_scope(
        "/api/v1/audio/transcriptions", root_path="/api", headers=[(b"content-length", b"5000")]
    )
    asyncio.run(BodyLimitMiddleware(inner, {"/v1/audio/transcriptions": 1000})(scope, None, send))
    assert sent[0]["status"] == 413 and not reached


def test_content_length_precheck_rejects_before_any_read() -> None:
    # Fails if the Content-Length pre-check is removed: the app would be reached / body read.
    reached, received = [], []

    async def inner(scope, receive, send):  # pragma: no cover - must not be reached
        reached.append(True)

    async def receive():  # pragma: no cover - must not be called
        received.append(True)
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(message):
        sent.append(message)

    scope = _asgi_scope("/v1/audio/transcriptions", headers=[(b"content-length", b"5000")])
    asyncio.run(BodyLimitMiddleware(inner, {"/v1/audio/transcriptions": 1000})(scope, receive, send))
    assert sent[0]["status"] == 413
    assert (b"connection", b"close") in sent[0]["headers"]
    assert not reached and not received


def test_rejections_send_connection_close(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _keyed_client(monkeypatch, api_key="secret")
    assert c.get("/v1/models").headers["connection"] == "close"


def test_auth_runs_before_body_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fails if BodyLimitMiddleware is wrapped outside auth: it would answer 413 first.
    c = _keyed_client(monkeypatch, api_key="secret", max_upload_bytes=1000)
    resp = c.post(
        "/v1/audio/transcriptions",
        files={"file": ("n.wav", b"\0" * 5000)},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


def test_key_compared_with_compare_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    c = _keyed_client(monkeypatch, api_key="secret")
    monkeypatch.setattr("whisperwebdav.server.hmac.compare_digest", spy)
    assert c.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
    assert c.get("/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert calls == [
        (b"Bearer secret", b"Bearer secret"),
        (b"Bearer nope", b"Bearer secret"),
    ]


def test_oversize_body_closes_spooled_upload_files(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fails if _BodyTooLarge stops being a MultiPartException: the multipart parser then no
    # longer closes the temp files it already opened.
    created = []

    class Recording(SpooledTemporaryFile):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    monkeypatch.setattr("starlette.formparsers.SpooledTemporaryFile", Recording)
    app = _keyed_client(monkeypatch, max_upload_bytes=1000).app
    boundary = "xBx"
    # Driven at ASGI level: TestClient delivers the whole body as one message, so the parser
    # would never get to open a temp file before the cap trips.
    chunks = [
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="n.wav"'
            f"\r\n\r\n"
        ).encode()
    ] + [b"\0" * 500 for _ in range(10)]
    messages = [
        {"type": "http.request", "body": chunk, "more_body": True} for chunk in chunks
    ]
    sent = []

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = _asgi_scope(
        "/v1/audio/transcriptions",
        headers=[(b"content-type", f"multipart/form-data; boundary={boundary}".encode())],
    )
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 413
    assert created and all(f.closed for f in created)


def test_handler_rechecks_upload_size(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fails if the handler's own size check is removed (middleware disabled to isolate it).
    class PassThrough:
        def __init__(self, app, limits):
            self.app = app

        async def __call__(self, scope, receive, send):
            await self.app(scope, receive, send)

    monkeypatch.setattr("whisperwebdav.server.BodyLimitMiddleware", PassThrough)
    c = _keyed_client(monkeypatch, max_upload_bytes=1000)
    resp = c.post("/v1/audio/transcriptions", files={"file": ("n.wav", b"\0" * 5000)})
    assert resp.status_code == 413


def test_unauthenticated_upload_never_calls_receive_full_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _keyed_client(monkeypatch, api_key="secret").app
    received = []
    sent = []

    async def receive():
        received.append(True)
        return {"type": "http.request", "body": b"\0" * 4096, "more_body": True}

    async def send(message):
        sent.append(message)

    scope = _asgi_scope(
        "/v1/audio/transcriptions",
        headers=[(b"content-type", b"multipart/form-data; boundary=x")],
    )
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 401
    assert not received
