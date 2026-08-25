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

import io
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import structlog

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
