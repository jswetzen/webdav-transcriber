"""Bounded, progress-monitored, process-isolated transcription for the HTTP server.

Every request runs in a child process (worker.py) under GPU_LOCK, so:
  - only one transcription touches the GPU at a time (GPU_LOCK is shared with TTS);
  - a request waiting for that slot is capped (max_waiting) and times out, instead of
    piling up threads and re-loading the model once per retry;
  - a job is killed only when it stops making progress (no new output or scratch files, and
    no CPU time) for transcribe_stall_seconds. Long recordings are never cut off for length.
    A stall is killed with the whole process group, which releases the model and any worker
    processes it spawned. Python can't kill a thread, which is why the old lock was held
    forever after one hung job.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import structlog

from .config import Config
from .gpu_lock import GPU_LOCK

log = structlog.get_logger(__name__)

# How often the monitor samples progress. Module-level so tests can shorten it.
_POLL_SECONDS = 5.0

_state_lock = threading.Lock()
_waiting = 0


class QueueFull(RuntimeError):
    """Too many requests are already waiting for the GPU. Maps to HTTP 503."""


class QueueTimeout(RuntimeError):
    """Waited too long for the GPU slot. Maps to HTTP 503."""


class JobStalled(RuntimeError):
    """The transcription child made no progress for transcribe_stall_seconds and was killed.

    Maps to HTTP 504.
    """


class JobFailed(RuntimeError):
    """The transcription child exited non-zero. HTTP 500."""


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _group_cpu_ticks(pgid: int) -> int:
    """Total utime+stime of every process in the group, from /proc. Rises while work happens."""
    total = 0
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            # Fields after the ")" that closes the comm name: state, ppid, pgrp, ..., utime (11),
            # stime (12). Index 2 is pgrp.
            fields = stat.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid:
                total += int(fields[11]) + int(fields[12])
        except (OSError, IndexError, ValueError):
            continue
    return total


def _tree_bytes(path: Path) -> int:
    """Bytes under the job's scratch dir (its log and easytranscriber's intermediate files)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _tail(path: Path, lines: int = 5) -> str:
    """Last few lines of the child's log, read from the end so a chatty child can't OOM us."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 8192))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return " | ".join(text.strip().splitlines()[-lines:])[-2000:]


def transcribe_isolated(
    audio_path: str,
    config: Config,
    *,
    argv_prefix: list[str] | None = None,
) -> list[dict]:
    """Run one transcription in a child process and return its normalized segments.

    argv_prefix is the command to run; the default is this interpreter running worker.py.
    The indirection exists so tests can substitute a fake child.
    """
    global _waiting
    acquired = GPU_LOCK.acquire(blocking=False)
    if not acquired:
        with _state_lock:
            if _waiting >= config.transcribe_max_waiting:
                raise QueueFull(f"{_waiting} transcription(s) already waiting for the GPU")
            _waiting += 1
        try:
            acquired = GPU_LOCK.acquire(timeout=config.transcribe_queue_wait_seconds)
        finally:
            with _state_lock:
                _waiting -= 1
    if not acquired:
        raise QueueTimeout(
            f"no GPU slot within {config.transcribe_queue_wait_seconds}s; try again later"
        )

    prefix = argv_prefix or [sys.executable, "-m", "whisperwebdav.worker"]
    try:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            result_path = tmpdir / "segments.json"
            log_path = tmpdir / "child.log"
            argv = [*prefix, audio_path, str(result_path), config.language]
            # TMPDIR points the child's own scratch space (easytranscriber's workspace) into
            # tmpdir, so the progress monitor sees its files grow too.
            env = {**os.environ, "TMPDIR": tmpdir_str}
            with open(log_path, "w", encoding="utf-8") as log_fh:
                # start_new_session puts the child (and anything it forks) in its own process
                # group, so the kill below reaches easyaligner's worker processes too.
                proc = subprocess.Popen(
                    argv, start_new_session=True, stdout=log_fh, stderr=subprocess.STDOUT, env=env
                )
                try:
                    _wait_while_progressing(proc, tmpdir, config, Path(audio_path).name)
                finally:
                    # Covers normal exit as well: no grandchild may outlive the job.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            if proc.returncode != 0:
                raise JobFailed(f"worker exited {proc.returncode}: {_tail(log_path)}")
            with open(result_path, encoding="utf-8") as fh:
                return json.load(fh)
    finally:
        GPU_LOCK.release()


def _wait_while_progressing(
    proc: subprocess.Popen, tmpdir: Path, config: Config, name: str
) -> None:
    """Block until proc exits; kill it if its progress signature stops changing for too long."""
    last_sig: tuple[int, int] | None = None
    last_progress = time.monotonic()
    while True:
        try:
            proc.wait(timeout=_POLL_SECONDS)
            return
        except subprocess.TimeoutExpired:
            pass
        sig = (_tree_bytes(tmpdir), _group_cpu_ticks(proc.pid))
        now = time.monotonic()
        if sig != last_sig:
            last_sig, last_progress = sig, now
        elif now - last_progress > config.transcribe_stall_seconds:
            _kill_group(proc)
            log.error(
                "Transcription stalled; child process group killed",
                file=name,
                stall_s=config.transcribe_stall_seconds,
            )
            raise JobStalled(
                f"no progress for {config.transcribe_stall_seconds}s; transcription was killed"
            )
