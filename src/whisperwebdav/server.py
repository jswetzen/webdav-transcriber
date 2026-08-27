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

import tempfile
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .config import Config
from .engine import transcribe_one
from .formatter import full_text, to_srt, to_vtt
from .tts import TTSError, available_voices, kokoro_files_present, synthesize, synthesize_stream

log = structlog.get_logger(__name__)

_TIMESTAMP_FORMATS = frozenset({"srt", "vtt", "verbose_json"})


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
    app = FastAPI(title="whisperwebdav", version="1")

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        """Enforce a bearer key iff config.api_key is set. No key configured == open."""
        if not config.api_key:
            return
        expected = f"Bearer {config.api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Invalid API key")

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

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(require_auth)])
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
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
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

    @app.post("/v1/audio/speech", dependencies=[Depends(require_auth)])
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
