"""OpenAI-compatible speech server (the model owner).

Exposes POST /v1/audio/transcriptions and POST /v1/audio/speech in the shape of the OpenAI
Audio API, so any OpenAI-speaking client (e.g. a Matrix bot, the poll loop in http mode, or a
TTS client) can talk to the shared KB-Whisper / Kokoro pipelines. Transcription requests
funnel through engine.transcribe_one and synthesis requests through tts.synthesize — both
acquire the same process-global GPU_LOCK (gpu_lock.py), so this remains the single process
that loads the models and the single gate on the GPU.

response_format controls the rendering (and, once supported, the pipeline depth):
  json (default) -> {"text": ...}        text  -> plain text
  verbose_json   -> segments + metadata  srt   -> SubRip      vtt -> WebVTT
"""

from __future__ import annotations

import hmac
import shutil
import tempfile
from pathlib import Path

import structlog
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartException
from starlette.routing import get_route_path
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import Config
from .engine import transcribe_one
from .formatter import full_text, to_srt, to_vtt
from .tts import TTSError, available_voices, kokoro_files_present, synthesize, synthesize_stream

log = structlog.get_logger(__name__)

_TIMESTAMP_FORMATS = frozenset({"srt", "vtt", "verbose_json"})

# /v1/audio/speech takes a small JSON body (TTS_MAX_INPUT_CHARS bounds the text), so it gets a
# fixed, much tighter cap than the multipart upload endpoint.
_SPEECH_MAX_BODY_BYTES = 1024 * 1024


class ApiKeyMiddleware:
    """Bearer-key gate for every HTTP request except an exact /healthz.

    A pure ASGI middleware rather than a FastAPI dependency on purpose: FastAPI parses the
    request body (multipart uploads get spooled to temp files) *before* it resolves
    dependencies, so a dependency would let an unauthenticated client force a full upload to be
    buffered before getting its 401. Here the rejection happens before `receive` is ever called.
    Deny-by-default (only /healthz is exempt) so a new route can't be forgotten, and matched on
    the routed path (root_path stripped) rather than raw scope["path"], which a proxy mounting
    the app under a prefix would otherwise sidestep. No key configured == open.
    """

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        self.app = app
        self.expected = f"Bearer {api_key}".encode() if api_key else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.expected is None or scope["type"] != "http" or get_route_path(scope) == "/healthz":
            await self.app(scope, receive, send)
            return
        supplied = b""
        for name, value in scope["headers"]:
            if name == b"authorization":
                supplied = value
                break
        # Constant-time comparison so response timing can't be used to guess the key.
        if not hmac.compare_digest(supplied, self.expected):
            response = JSONResponse(
                {"detail": "Invalid API key"},
                status_code=401,
                # Connection: close makes uvicorn hang up instead of draining the unread body.
                headers={"WWW-Authenticate": "Bearer", "Connection": "close"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class _BodyTooLarge(MultiPartException):
    """Raised from the wrapped `receive` once the body exceeds its cap.

    Subclasses MultiPartException so starlette's multipart parser closes the spooled upload
    temp files it has already opened (it only cleans up on that exception type and OSError).
    """

    def __init__(self) -> None:
        super().__init__("Request body too large")


class BodyLimitMiddleware:
    """Reject over-sized request bodies for the POST endpoints named in `limits`.

    Content-Length is checked up front; chunked uploads (no Content-Length) are caught by
    counting bytes as they arrive through a wrapped `receive`. FastAPI turns any exception from
    body parsing into its own 400, so once the cap trips we swallow whatever the app sends
    and emit the 413 ourselves.
    """

    def __init__(self, app: ASGIApp, limits: dict[str, int]) -> None:
        self.app = app
        self.limits = limits

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit = self.limits.get(get_route_path(scope)) if scope["type"] == "http" else None
        if limit is None or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return

        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    too_big = int(value) > limit
                except ValueError:
                    too_big = False  # malformed; the byte counter below still applies
                if too_big:
                    await self._reject(scope, receive, send, limit)
                    return
                break

        received = 0
        exceeded = False
        responded = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            if exceeded:
                raise _BodyTooLarge
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal responded
            if exceeded:
                return
            responded = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded and not responded:
            await self._reject(scope, receive, send, limit)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send, limit: int) -> None:
        response = JSONResponse(
            {"detail": f"Request body exceeds the {limit} byte limit"},
            status_code=413,
            headers={"Connection": "close"},
        )
        await response(scope, receive, send)


class SpeechRequest(BaseModel):
    """POST /v1/audio/speech body, matching OpenAI's JSON (not multipart) request shape."""

    model: str = ""
    input: str
    voice: str = ""
    response_format: str = "wav"
    # Not part of OpenAI's JSON body (their client picks streaming vs. buffered by how it reads
    # the HTTP response, not via a request field) -- added explicitly here so existing wav
    # callers keep their current buffered behavior by default. Only mp3/pcm support it; see
    # tts.synthesize_stream's docstring for why. response_format="pcm" implies streaming on its
    # own too (see the speech() handler) since some OpenAI-compatible clients (e.g. customtts)
    # signal streaming that way instead of setting this field.
    stream: bool = False


def create_app(config: Config) -> FastAPI:
    # The interactive docs / schema are off: the server is exposed publicly and they would
    # advertise the API surface to unauthenticated visitors.
    app = FastAPI(
        title="whisperwebdav", version="1", docs_url=None, redoc_url=None, openapi_url=None
    )
    # add_middleware wraps outward, so the last one added runs first: auth must precede the
    # body cap so unauthenticated clients learn nothing (not even a 413) about the limits.
    app.add_middleware(
        BodyLimitMiddleware,
        limits={
            "/v1/audio/transcriptions": config.max_upload_bytes,
            "/v1/audio/speech": _SPEECH_MAX_BODY_BYTES,
        },
    )
    app.add_middleware(ApiKeyMiddleware, api_key=config.api_key)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict:
        # `voices` is additive beyond the strict OpenAI /v1/models shape (that endpoint has no
        # concept of voices) — harmless for OpenAI clients, which only read `data`, and lets
        # non-OpenAI clients (the Firefox extension, Open WebUI) discover voices without a
        # second bespoke endpoint.
        data = [{"id": config.transcription_model, "object": "model", "owned_by": "kblab"}]
        voices: list[str] = []
        if kokoro_files_present(config):
            data.append({"id": "kokoro", "object": "model", "owned_by": "kokoro-onnx"})
            try:
                voices = sorted(await run_in_threadpool(available_voices, config))
            except TTSError:
                log.warning("Kokoro model files present but failed to load; omitting voices")
        return {"object": "list", "data": data, "voices": voices}

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(default=""),
        language: str = Form(default=""),
        response_format: str = Form(default="json"),
        temperature: float = Form(default=0.0),
    ):
        if response_format not in {"json", "text", "verbose_json", "srt", "vtt"}:
            raise HTTPException(
                status_code=400, detail=f"Unsupported response_format '{response_format}'"
            )

        # easytranscriber loads audio by path; preserve the extension so ffmpeg picks the
        # right demuxer.
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        # Defensive: BodyLimitMiddleware already caps the whole request body.
        if file.size is not None and file.size > config.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file too large")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            # Chunked copy in a worker thread: no single bytes object for the whole upload and
            # no blocking of the event loop on disk writes.
            await run_in_threadpool(shutil.copyfileobj, file.file, tmp)
            audio_path = tmp.name

        # Per-request language override (falls back to the server's configured language).
        req_config = config.model_copy(update={"language": language}) if language else config
        with_timestamps = response_format in _TIMESTAMP_FORMATS

        try:
            segments = await run_in_threadpool(
                transcribe_one, audio_path, req_config, with_timestamps=with_timestamps
            )
        except Exception as exc:
            log.exception("Transcription failed", filename=file.filename)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            Path(audio_path).unlink(missing_ok=True)

        text = full_text(segments)
        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "srt":
            return PlainTextResponse(to_srt(segments))
        if response_format == "vtt":
            return PlainTextResponse(to_vtt(segments))
        if response_format == "verbose_json":
            duration = max((s["end"] for s in segments), default=0.0)
            return JSONResponse(
                {
                    "task": "transcribe",
                    "language": req_config.language,
                    "duration": duration,
                    "text": text,
                    "segments": [
                        {"id": i, "start": s["start"], "end": s["end"], "text": s["text"]}
                        for i, s in enumerate(segments)
                    ],
                }
            )
        return JSONResponse({"text": text})

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        if not req.input.strip():
            raise HTTPException(status_code=400, detail="input must not be empty")
        if len(req.input) > config.tts_max_input_chars:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"input is {len(req.input)} chars, exceeding TTS_MAX_INPUT_CHARS "
                    f"({config.tts_max_input_chars}). Split long articles client-side."
                ),
            )

        # response_format="pcm" always implies streaming, `stream` field or not: pcm is a
        # headerless raw-sample format that only makes sense read incrementally, and it's what
        # OpenAI-compatible streaming TTS clients request instead of setting a stream flag --
        # e.g. the customtts Firefox extension never sends `stream`, only response_format=pcm,
        # per its background.js (streaming mode) vs. response_format=mp3 (download mode).
        if req.stream or req.response_format == "pcm":
            # response_format's Pydantic default is "wav" (the non-streaming default), which is
            # indistinguishable from a client explicitly asking for wav. Use model_fields_set
            # to tell "left unset" (silently take mp3) apart from "explicitly asked for wav" (a
            # real 400 -- streaming a wav is impossible, not just unspecified).
            fmt = req.response_format if "response_format" in req.model_fields_set else "mp3"
            if fmt not in ("mp3", "pcm"):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"stream=true only supports response_format='mp3' or 'pcm' (wav needs "
                        f"a known length up front); got '{fmt}'."
                    ),
                )

            gen = synthesize_stream(req.input, config, voice=req.voice, response_format=fmt)
            # Pull the first chunk before returning a StreamingResponse: kokoro-onnx's
            # create_stream() is a plain async generator, so the model-load/voice-validation
            # code inside synthesize_stream runs on this first anext() and raises TTSError
            # here (still a normal 400) rather than after headers are already sent. See that
            # function's docstring for the streaming-vs-error-handling tradeoff this implies.
            try:
                first_chunk = await gen.__anext__()
            except StopAsyncIteration:
                first_chunk = b""
            except TTSError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                log.exception("Speech streaming failed")
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            async def body():
                if first_chunk:
                    yield first_chunk
                async for chunk in gen:
                    yield chunk

            log.info(
                "Streaming speech",
                chars=len(req.input),
                voice=req.voice or config.tts_default_voice,
                format=fmt,
            )
            media_type = "audio/mpeg" if fmt == "mp3" else "audio/pcm"
            return StreamingResponse(body(), media_type=media_type)

        try:
            result = await run_in_threadpool(
                synthesize,
                req.input,
                config,
                voice=req.voice,
                response_format=req.response_format,
            )
        except TTSError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("Speech synthesis failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        log.info(
            "Synthesized speech",
            chars=len(req.input),
            voice=req.voice or config.tts_default_voice,
            format=req.response_format,
            duration_s=round(result.duration_seconds, 2),
        )
        return Response(content=result.audio, media_type=result.media_type)

    return app


def main() -> None:
    import uvicorn

    from .watcher import _configure_logging

    config = Config()
    _configure_logging(config)
    log.info(
        "Starting whisperwebdav server",
        host=config.server_host,
        port=config.server_port,
        model=config.transcription_model,
        device=config.device,
        auth=bool(config.api_key),
    )
    uvicorn.run(create_app(config), host=config.server_host, port=config.server_port)
