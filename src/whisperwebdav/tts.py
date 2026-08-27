"""In-process Kokoro TTS engine — the synthesis counterpart to engine.py's transcription.

kokoro-onnx (https://github.com/thewh1teagle/kokoro-onnx) wraps a single ONNX Runtime
session; constructing `Kokoro(...)` loads the model weights and the voice bank into memory.
Unlike easytranscriber's per-call pipeline() (see engine.py), that construction is cheap
enough (~300 MB) to do once and hold warm for the process lifetime, so we lazily build one
module-global instance on first use and reuse it for every request after.

GPU selection is automatic and NOT configured here: ONNX Runtime's default session picks
whichever execution providers are registered in the environment (CUDAExecutionProvider when
the `onnxruntime-gpu` wheel is installed and CUDA libs are visible — see Dockerfile's
ldconfig step reused from the Whisper/CTranslate2 path — else it falls back to
CPUExecutionProvider on its own). Synthesis still acquires the process-global GPU_LOCK
(gpu_lock.py) so it never overlaps a Whisper transcription on the same card, even though
Kokoro itself doesn't know about Whisper.
"""

from __future__ import annotations

import asyncio
import io
import shutil
import subprocess
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
from starlette.concurrency import run_in_threadpool

from .config import Config
from .gpu_lock import GPU_LOCK

log = structlog.get_logger(__name__)

_SUPPORTED_FORMATS = frozenset({"wav", "mp3"})

# Lazily-built singleton. Guarded by _init_lock rather than GPU_LOCK: construction itself
# needs the GPU gate (it touches the GPU the moment ONNX Runtime picks a CUDA provider), but
# the emptiness check must not race two concurrent first requests into double-constructing.
_kokoro = None
_init_lock = threading.Lock()
_voices_cache: frozenset[str] | None = None


class TTSError(RuntimeError):
    """User-facing synthesis failure: bad request, missing model files, encoding failure."""


def kokoro_files_present(config: Config) -> bool:
    """Cheap existence check, used by /v1/models to decide whether to advertise TTS at all
    without forcing a model load on a GET request."""
    return Path(config.kokoro_model_path).is_file() and Path(config.kokoro_voices_path).is_file()


def _load_kokoro(config: Config):
    global _kokoro
    if _kokoro is not None:
        return _kokoro
    with _init_lock:
        if _kokoro is not None:
            return _kokoro
        if not kokoro_files_present(config):
            raise TTSError(
                f"Kokoro model files not found at KOKORO_MODEL_PATH={config.kokoro_model_path} "
                f"/ KOKORO_VOICES_PATH={config.kokoro_voices_path}. Download kokoro-v1.0.onnx "
                "and voices-v1.0.bin from https://github.com/thewh1teagle/kokoro-onnx/releases "
                "and place them there (see README)."
            )
        from kokoro_onnx import Kokoro  # heavy import; deferred like transcriber.py's easytranscriber

        log.info("Loading Kokoro TTS model", model_path=config.kokoro_model_path)
        instance = Kokoro(config.kokoro_model_path, config.kokoro_voices_path)
        try:
            import onnxruntime as ort

            log.info("ONNX Runtime providers available", providers=ort.get_available_providers())
        except ImportError:
            pass
        _kokoro = instance
        return _kokoro


def available_voices(config: Config) -> frozenset[str]:
    global _voices_cache
    if _voices_cache is None:
        _voices_cache = frozenset(_load_kokoro(config).get_voices())
    return _voices_cache


@dataclass
class SynthesisResult:
    audio: bytes
    media_type: str
    duration_seconds: float


def synthesize(
    text: str, config: Config, *, voice: str = "", response_format: str = "wav"
) -> SynthesisResult:
    """Synthesize `text` to audio bytes. Runs under GPU_LOCK end to end (model load included,
    on first call) so it never overlaps a Whisper transcription — see module docstring."""
    response_format = response_format or "wav"
    if response_format not in _SUPPORTED_FORMATS:
        raise TTSError(
            f"Unsupported response_format '{response_format}'; "
            f"supported: {', '.join(sorted(_SUPPORTED_FORMATS))}"
        )

    voice = voice or config.tts_default_voice

    with GPU_LOCK:
        kokoro = _load_kokoro(config)
        voices = available_voices(config)
        if voice not in voices:
            raise TTSError(f"Unknown voice '{voice}'. Available: {', '.join(sorted(voices))}")

        log.info("Synthesizing speech", chars=len(text), voice=voice, format=response_format)
        samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang=config.tts_lang)

    wav_bytes = _encode_wav(samples, sample_rate)
    duration = len(samples) / float(sample_rate)

    if response_format == "wav":
        return SynthesisResult(audio=wav_bytes, media_type="audio/wav", duration_seconds=duration)
    return SynthesisResult(
        audio=_wav_to_mp3(wav_bytes), media_type="audio/mpeg", duration_seconds=duration
    )


async def synthesize_stream(
    text: str, config: Config, *, voice: str = "", response_format: str = "mp3"
) -> AsyncGenerator[bytes, None]:
    """Async counterpart to synthesize(): yields mp3 bytes as kokoro-onnx's create_stream()
    produces them, sentence/clause-chunked internally (see its docstring), instead of
    buffering the whole utterance like synthesize() does. wav is deliberately unsupported here
    — a valid WAV header wants a known total sample count up front, which a live stream can't
    supply — so callers must request response_format="mp3"; server.py enforces that before
    calling in.

    Validation (bad format, missing model files, unknown voice) all happens as plain sync code
    before the first `yield` below, so it raises TTSError out of the FIRST `anext()` on this
    generator rather than mid-stream. server.py relies on that: it pulls one chunk before
    committing to a 200 response, so a bad voice still comes back as a normal 400 instead of a
    truncated stream. A failure *after* that point (e.g. ffmpeg dying mid-encode) can't be
    turned into an HTTP error any more — the client already has a 200 and partial bytes — the
    stream just ends early; that's an inherent limit of HTTP streaming, not special-cased here.

    GPU_LOCK handling: unlike synthesize(), this holds the lock across a whole async generator
    lifetime (a full utterance's worth of chunks, however long the client takes to consume
    them), so acquiring it can't happen with a plain blocking `with GPU_LOCK:` on the event
    loop thread — that would stall every other request on this process for the acquire wait,
    not just this one. The acquire is done in a threadpool worker instead (release() itself
    never blocks, so it's called directly). Net effect: a streaming request still monopolizes
    the single GPU for its whole duration, same as a non-streaming one would for its shorter
    one — deliberate given this is a single-GPU, single-tenant service (see module docstring),
    not an accident of this implementation.
    """
    response_format = response_format or "mp3"
    if response_format != "mp3":
        raise TTSError(
            f"Streaming only supports response_format='mp3' (wav needs a known length "
            f"up front); got '{response_format}'."
        )

    voice = voice or config.tts_default_voice

    await run_in_threadpool(GPU_LOCK.acquire)
    try:
        kokoro = _load_kokoro(config)
        voices = available_voices(config)
        if voice not in voices:
            raise TTSError(f"Unknown voice '{voice}'. Available: {', '.join(sorted(voices))}")

        log.info("Streaming speech", chars=len(text), voice=voice, format=response_format)

        chunks = kokoro.create_stream(text, voice=voice, speed=1.0, lang=config.tts_lang)
        async for mp3_bytes in _stream_pcm_to_mp3(chunks):
            yield mp3_bytes
    finally:
        GPU_LOCK.release()


async def _stream_pcm_to_mp3(
    pcm_chunks: AsyncGenerator[tuple, None],
) -> AsyncGenerator[bytes, None]:
    """Pipe float32 PCM chunks through one persistent ffmpeg process, yielding mp3 bytes as
    they become available, rather than _wav_to_mp3's one subprocess.run() per whole buffer.
    ffmpeg is started lazily on the first chunk (it needs -ar up front, which we only learn
    from kokoro's first (samples, sample_rate) tuple — every chunk shares one rate in practice)
    -- also means text that produces zero chunks (e.g. trims to nothing) never touches ffmpeg
    at all, so its absence isn't an error for that case.
    """
    proc: asyncio.subprocess.Process | None = None
    stdout_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def pump_stdout(stdout: asyncio.StreamReader) -> None:
        while True:
            chunk = await stdout.read(4096)
            if not chunk:
                break
            await stdout_queue.put(chunk)
        await stdout_queue.put(None)  # EOF sentinel

    pump_task: asyncio.Task | None = None
    try:
        async for samples, sample_rate in pcm_chunks:
            if proc is None:
                if shutil.which("ffmpeg") is None:
                    raise TTSError(
                        "mp3 output requires ffmpeg, which is not installed in this image"
                    )
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "f32le",
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    "1",
                    "-i",
                    "pipe:0",
                    "-f",
                    "mp3",
                    "pipe:1",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                assert proc.stdout is not None
                pump_task = asyncio.create_task(pump_stdout(proc.stdout))

            assert proc.stdin is not None
            proc.stdin.write(np.asarray(samples, dtype=np.float32).tobytes())
            await proc.stdin.drain()

            # Drain whatever mp3 bytes ffmpeg has already flushed without waiting for more --
            # it emits frames as soon as it has enough encoded audio, not lockstep with our
            # writes, so pulling eagerly here (rather than only after EOF) is what actually
            # makes this stream instead of buffer.
            while not stdout_queue.empty():
                chunk = stdout_queue.get_nowait()
                if chunk is None:
                    break
                yield chunk

        if proc is None:
            return  # kokoro produced no chunks at all (e.g. text trimmed to nothing)

        assert proc.stdin is not None
        proc.stdin.write_eof()
        while True:
            chunk = await stdout_queue.get()
            if chunk is None:
                break
            yield chunk

        assert pump_task is not None
        await pump_task
        returncode = await proc.wait()
        if returncode != 0:
            stderr = await proc.stderr.read() if proc.stderr else b""
            raise TTSError(f"ffmpeg mp3 encoding failed: {stderr.decode(errors='replace')}")
    finally:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()


def _encode_wav(samples, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


def _wav_to_mp3(wav_bytes: bytes) -> bytes:
    """Shell out to ffmpeg rather than adding another audio-encoding dependency — it's
    already a required runtime package for Whisper's input demuxing (see Dockerfile)."""
    if shutil.which("ffmpeg") is None:
        raise TTSError("mp3 output requires ffmpeg, which is not installed in this image")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "mp3", "pipe:1"],
        input=wav_bytes,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise TTSError(f"ffmpeg mp3 encoding failed: {proc.stderr.decode(errors='replace')}")
    return proc.stdout
