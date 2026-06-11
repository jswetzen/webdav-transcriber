"""In-process transcription engine: the single serialization point for the GPU.

The model owner (server.py) and the standalone poll loop both go through transcribe_one()
under a process-global lock, so only one transcription touches the GPU at a time. With a
single 12 GB card shared by other consumers (e.g. an LLM), serializing here is what prevents
two concurrent jobs from each loading the model and exhausting VRAM.

Models are loaded per call today (easytranscriber's pipeline() isn't structured to hold
weights warm); release_gpu_memory() runs after each job. A warm-model fast path and a
whisper-only (skip wav2vec2/alignment) path for timestamp-free requests are future
optimizations — see with_timestamps below.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

import structlog

from .config import Config
from .formatter import normalize_segments
from .transcriber import transcribe_batch

log = structlog.get_logger(__name__)

# Process-global GPU gate. Held for the duration of a transcription so concurrent HTTP
# requests (and the in-process poll loop, if any) serialize rather than contend for VRAM.
_gpu_lock = threading.Lock()


def transcribe_one(
    audio_path: str, config: Config, *, with_timestamps: bool = True
) -> list[dict]:
    """Transcribe a single audio file, returning normalized {start, end, text} segments.

    Acquires the process-global GPU lock for the whole run. with_timestamps is accepted now
    so callers (the server, keyed off response_format) can request a cheaper text-only result;
    until easytranscriber exposes a whisper-only path it always runs the full aligned pipeline.
    """
    with _gpu_lock:
        log.info(
            "Transcribing", file=Path(audio_path).name, with_timestamps=with_timestamps
        )
        result = transcribe_batch([audio_path], config)
        try:
            segments = result.segments_by_path.get(audio_path, [])
            return normalize_segments(segments)
        finally:
            # transcribe_batch already calls release_gpu_memory() in its finally; we own the
            # workspace cleanup (see its docstring).
            shutil.rmtree(result.workspace, ignore_errors=True)
