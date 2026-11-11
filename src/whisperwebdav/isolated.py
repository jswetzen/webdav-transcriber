"""Process-isolated transcription: one child process per job, under GPU_LOCK.

Exiting the child frees its model and VRAM; killing its group also reaps easyaligner's workers.
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
from .gpu_lock import GPU_LOCK, GpuUnavailable, is_wedged, mark_wedged

log = structlog.get_logger(__name__)

# Module-level so tests can shorten them.
_POLL_SECONDS = 5.0  # how often progress is sampled
_TICK_SECONDS = 0.5  # how often exit is checked, so a finished job isn't held for a sample
_REAP_SECONDS = 30.0

_state_lock = threading.Lock()
_waiting = 0


class QueueFull(RuntimeError):
    """Too many requests are already waiting for the GPU. Maps to HTTP 503."""


class QueueTimeout(RuntimeError):
    """Waited too long for the GPU slot. Maps to HTTP 503."""


class JobStalled(RuntimeError):
    """The child stopped making progress, or ran past transcribe_max_job_seconds, and was killed.

    Maps to HTTP 504.
    """


class JobFailed(RuntimeError):
    """The transcription child exited non-zero. HTTP 500."""


def _has_exited(proc: subprocess.Popen) -> bool:
    # WNOWAIT leaves the zombie in place, so its pid can't be recycled before we killpg.
    return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _terminate_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _reap(proc: subprocess.Popen) -> None:
    try:
        proc.wait(timeout=_REAP_SECONDS)
    except subprocess.TimeoutExpired:
        mark_wedged()
        log.error("Killed child did not exit; GPU unusable until restart", pid=proc.pid)


def _group_cpu_ticks(pgid: int) -> int:
    """Total utime+stime of every process in the group, from /proc."""
    total = 0
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            # Fields after the ")" of the comm name: pgrp=2, utime=11, stime=12.
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
    """
    global _waiting
    if is_wedged():
        raise GpuUnavailable("a previous transcription could not be killed; restart the service")

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
        if is_wedged():
            raise GpuUnavailable("a previous job could not be killed; restart the service")
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
            # Scratch files land under tmpdir, which is what the progress monitor measures.
            env = {**os.environ, "TMPDIR": tmpdir_str}
            with open(log_path, "w", encoding="utf-8") as log_fh:
                # Own process group, so the kill also reaches easyaligner's workers.
                proc = subprocess.Popen(
                    argv, start_new_session=True, stdout=log_fh, stderr=subprocess.STDOUT, env=env
                )
                try:
                    _wait_while_progressing(proc, tmpdir, config, Path(audio_path).name)
                finally:
                    _terminate_group(proc)
                    _reap(proc)
            if is_wedged():
                raise JobStalled("transcription child could not be killed; restart the service")
            if proc.returncode != 0:
                raise JobFailed(f"worker exited {proc.returncode}: {_tail(log_path)}")
            with open(result_path, encoding="utf-8") as fh:
                return json.load(fh)
    finally:
        # A wedged device keeps the lock: nothing else may touch it.
        if not is_wedged():
            GPU_LOCK.release()


def _wait_while_progressing(
    proc: subprocess.Popen, tmpdir: Path, config: Config, name: str
) -> None:
    """Return once proc exits; kill it if progress stalls or the job runs past its ceiling."""
    started = time.monotonic()
    last_sig: tuple[int, int] | None = None
    last_progress = last_sample = started
    while not _has_exited(proc):
        time.sleep(_TICK_SECONDS)
        now = time.monotonic()
        if now - started > config.transcribe_max_job_seconds:
            _terminate_group(proc)
            log.error("Transcription exceeded max job time; killed", file=name,
                      max_s=config.transcribe_max_job_seconds)
            raise JobStalled(
                f"ran longer than {config.transcribe_max_job_seconds}s; transcription was killed"
            )
        if now - last_sample < _POLL_SECONDS:
            continue
        last_sample = now
        sig = (_tree_bytes(tmpdir), _group_cpu_ticks(proc.pid))
        if sig != last_sig:
            last_sig, last_progress = sig, now
        elif now - last_progress > config.transcribe_stall_seconds:
            _terminate_group(proc)
            log.error("Transcription stalled; killed", file=name,
                      stall_s=config.transcribe_stall_seconds)
            raise JobStalled(
                f"no progress for {config.transcribe_stall_seconds}s; transcription was killed"
            )
