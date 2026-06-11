"""OpenAI-compatible transcription server (the model owner).

Exposes POST /v1/audio/transcriptions in the shape of the OpenAI Audio API, so any
OpenAI-speaking client (e.g. a Matrix bot, or the poll loop in http mode) can transcribe
against the shared KB-Whisper pipeline. All requests funnel through engine.transcribe_one,
which holds a process-global lock — this is the single process that loads the model and the
single gate on the GPU.

response_format controls the rendering (and, once supported, the pipeline depth):
  json (default) -> {"text": ...}        text  -> plain text
  verbose_json   -> segments + metadata  srt   -> SubRip      vtt -> WebVTT
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from .config import Config
from .engine import transcribe_one
from .formatter import full_text, to_srt, to_vtt

log = structlog.get_logger(__name__)

_TIMESTAMP_FORMATS = frozenset({"srt", "vtt", "verbose_json"})


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
        return {
            "object": "list",
            "data": [
                {"id": config.transcription_model, "object": "model", "owned_by": "kblab"}
            ],
        }

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
