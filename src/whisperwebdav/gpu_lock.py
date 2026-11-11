"""Process-global GPU gate shared by every GPU-bound engine (transcription, synthesis).

engine.py (Whisper) and tts.py (Kokoro) can both run on the GPU when GPU_ENABLED=true. With a
single card shared by other consumers too (see engine.py's docstring — e.g. a co-located LLM),
letting a transcription and a synthesis job run concurrently risks two model loads exhausting
VRAM at once. Both engines acquire this same lock for the duration of a job, so at most one
GPU-bound operation runs at a time across the whole process, transcription and synthesis alike.

Split into its own module (rather than living in engine.py, where it started) purely so tts.py
doesn't have to reach into another module's "private" underscore-prefixed name to share it.
"""

from __future__ import annotations

import threading

GPU_LOCK = threading.Lock()

# Set when a killed GPU job would not exit (a D-state process in the driver). GPU_LOCK is then
# never released, so the device is unusable until restart; callers check this before waiting.
_wedged = False


class GpuUnavailable(RuntimeError):
    """The GPU is wedged and the service must be restarted."""


def mark_wedged() -> None:
    global _wedged
    _wedged = True


def is_wedged() -> bool:
    return _wedged
