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
import gc
import io
import shutil
import subprocess
import threading
import time
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

# Idle release. The ONNX Runtime session is NOT covered by transcriber.release_gpu_memory():
# that calls torch.cuda.empty_cache(), which only touches PyTorch's allocator, and Kokoro runs on
# onnxruntime-gpu, whose CUDA arena lives and dies with the InferenceSession. Observed 2026-09-29:
# ~2.3 GiB of VRAM still held by whisper-server more than a day after its last /v1/audio/speech
# request, which left no room for bonsai (llama-server on the ollama CT) on the shared 12 GB card
# and put it in a crash loop on cudaMalloc OOM. The only way to give that memory back is to drop
# the session, so a watchdog thread does exactly that after tts_idle_release_seconds without a
# request and the next request pays the (cheap, see module docstring) reload. _voices_cache is
# deliberately kept across a release: it's a plain frozenset with no GPU cost, and keeping it
# means a /v1/models listing after a release doesn't reload the model just to name the voices.
_last_used = time.monotonic()
_watchdog: threading.Thread | None = None
_WATCHDOG_MAX_POLL_SECONDS = 30.0


class TTSError(RuntimeError):
    """User-facing synthesis failure: bad request, missing model files, encoding failure."""


def kokoro_files_present(config: Config) -> bool:
    """Cheap existence check, used by /v1/models to decide whether to advertise TTS at all
    without forcing a model load on a GET request."""
    return Path(config.kokoro_model_path).is_file() and Path(config.kokoro_voices_path).is_file()


def _touch() -> None:
    global _last_used
    _last_used = time.monotonic()


def _release_if_idle(idle_seconds: float) -> bool:
    """Drop the Kokoro session if it has sat unused for `idle_seconds`. Returns True when the
    watchdog is done (released, or already gone) and False when it should keep waiting.

    GPU_LOCK is only try-acquired: a synthesis (or a streaming response, which holds it for the
    whole utterance) means the model is in use right now, so we back off and look again next
    tick instead of blocking behind it. Lock order is _init_lock -> GPU_LOCK here but
    GPU_LOCK -> _init_lock in _load_kokoro's callers; that can't deadlock because this side never
    waits on GPU_LOCK.
    """
    global _kokoro, _watchdog
    with _init_lock:
        if _kokoro is None:
            _watchdog = None
            return True
        if time.monotonic() - _last_used < idle_seconds:
            return False
        if not GPU_LOCK.acquire(blocking=False):
            return False
        try:
            _kokoro = None
            _watchdog = None
        finally:
            GPU_LOCK.release()
    # The session is destroyed (and its CUDA arena freed) once the last reference goes; collect
    # now rather than whenever the GC next gets around to it, since the point is to free VRAM.
    gc.collect()
    log.info("Released idle Kokoro TTS model", idle_seconds=idle_seconds)
    return True


def _watch_idle(idle_seconds: float) -> None:
    # Poll at most every 30s so a long idle threshold still releases reasonably close to it,
    # without a busy loop for a short one.
    poll = min(idle_seconds, _WATCHDOG_MAX_POLL_SECONDS)
    while True:
        time.sleep(poll)
        if _release_if_idle(idle_seconds):
            return


def _load_kokoro(config: Config):
    global _kokoro, _watchdog
    if _kokoro is not None:
        _touch()
        return _kokoro
    with _init_lock:
        if _kokoro is not None:
            _touch()
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
        _touch()
        # One watchdog per loaded instance: it exits after releasing, and the next load (the
        # first request after a release) starts a fresh one.
        idle = config.tts_idle_release_seconds
        if idle > 0 and _watchdog is None:
            _watchdog = threading.Thread(
                target=_watch_idle, args=(idle,), name="kokoro-idle-release", daemon=True
            )
            _watchdog.start()
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
        _touch()

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
    """Async counterpart to synthesize(): yields audio bytes as kokoro-onnx's create_stream()
    produces them, sentence/clause-chunked internally (see its docstring), instead of
    buffering the whole utterance like synthesize() does. Supports "mp3" and "pcm"; wav is
    deliberately unsupported — a valid WAV header wants a known total sample count up front,
    which a live stream can't supply. "pcm" means raw s16le mono samples at kokoro's own
    output rate (24kHz), no container at all — the format OpenAI-compatible streaming TTS
    clients (e.g. the customtts Firefox extension) request for low-latency playback via
    Web Audio, and the cheaper of the two here since it skips ffmpeg entirely.

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
    if response_format not in ("mp3", "pcm"):
        raise TTSError(
            f"Streaming only supports response_format='mp3' or 'pcm' (wav needs a known "
            f"length up front); got '{response_format}'."
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
        encode = _stream_pcm_to_mp3 if response_format == "mp3" else _stream_pcm_to_s16le
        async for audio_bytes in encode(chunks):
            yield audio_bytes
    finally:
        # Touch before releasing the lock so idle time counts from the END of a long stream,
        # not from when it started (_load_kokoro's touch) -- otherwise a multi-minute stream
        # could be released moments after it finishes.
        _touch()
        GPU_LOCK.release()


async def _stream_pcm_to_s16le(
    pcm_chunks: AsyncGenerator[tuple, None],
) -> AsyncGenerator[bytes, None]:
    """Convert each (float32 samples, sample_rate) chunk straight to raw little-endian int16
    bytes and yield it -- no subprocess, no container framing, so there's nothing to buffer:
    every chunk kokoro produces goes out as soon as it's converted. sample_rate is passed
    through unused (kokoro-onnx's own SAMPLE_RATE is fixed at 24kHz; a caller expecting raw
    PCM is expected to already know the rate out of band, same as the customtts extension's
    hardcoded PCM_SAMPLE_RATE — there's no room in a headerless format to say it inline).
    """
    async for samples, _sample_rate in pcm_chunks:
        clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        yield (clipped * 32767.0).astype("<i2").tobytes()


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
