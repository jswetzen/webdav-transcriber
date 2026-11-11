"""transcribe_isolated: one job at a time, bounded wait, and kill-only-when-progress-stops."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from whisperwebdav import isolated
from whisperwebdav.config import Config
from whisperwebdav.gpu_lock import GPU_LOCK
from whisperwebdav.isolated import (
    JobFailed,
    JobStalled,
    QueueFull,
    QueueTimeout,
    transcribe_isolated,
)

OK_SCRIPT = (
    "import json,sys; "
    "json.dump([{'start': 0.0, 'end': 1.0, 'text': 'hej'}], open(sys.argv[2], 'w'))"
)


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(isolated, "_POLL_SECONDS", 0.2)


def _cfg(**kw) -> Config:
    return Config(**kw)


def _proc_dead(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return True
    return state == "Z"


def _lock_is_free() -> bool:
    if GPU_LOCK.acquire(blocking=False):
        GPU_LOCK.release()
        return True
    return False


def test_success_returns_segments_and_releases_lock() -> None:
    segments = transcribe_isolated(
        "/tmp/a.wav", _cfg(), argv_prefix=[sys.executable, "-c", OK_SCRIPT]
    )
    assert segments == [{"start": 0.0, "end": 1.0, "text": "hej"}]
    assert _lock_is_free()


def test_silent_job_is_killed_once_progress_stops() -> None:
    started = time.monotonic()
    with pytest.raises(JobStalled):
        transcribe_isolated(
            "/tmp/a.wav",
            _cfg(transcribe_stall_seconds=1),
            argv_prefix=[sys.executable, "-c", "import time; time.sleep(60)"],
        )
    assert time.monotonic() - started < 15
    assert _lock_is_free()


def test_slow_job_that_keeps_making_progress_is_not_killed() -> None:
    # Runs for ~4.5s, well past the 2s stall window, but writes output every 0.3s.
    script = (
        "import json,sys,time\n"
        "for i in range(15):\n"
        "    print('step', i, flush=True)\n"
        "    time.sleep(0.3)\n"
        "json.dump([{'start': 0.0, 'end': 1.0, 'text': 'lång'}], open(sys.argv[2], 'w'))\n"
    )
    segments = transcribe_isolated(
        "/tmp/a.wav",
        _cfg(transcribe_stall_seconds=2),
        argv_prefix=[sys.executable, "-c", script],
    )
    assert segments[0]["text"] == "lång"


def test_cpu_bound_job_without_output_is_not_killed() -> None:
    # No output and no files, but CPU time keeps rising: that's work, not a stall.
    script = (
        "import json,sys,time\n"
        "end = time.monotonic() + 4.5\n"
        "while time.monotonic() < end:\n"
        "    sum(range(10000))\n"
        "json.dump([], open(sys.argv[2], 'w'))\n"
    )
    assert (
        transcribe_isolated(
            "/tmp/a.wav",
            _cfg(transcribe_stall_seconds=2),
            argv_prefix=[sys.executable, "-c", script],
        )
        == []
    )


def test_stall_also_kills_grandchildren(tmp_path: Path) -> None:
    # Models easyaligner's worker processes: a child that forks a helper and then hangs.
    pid_file = tmp_path / "pid"
    script = (
        "import subprocess,sys,time; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"open({str(pid_file)!r}, 'w').write(str(p.pid)); "
        "time.sleep(60)"
    )
    with pytest.raises(JobStalled):
        transcribe_isolated(
            "/tmp/a.wav",
            _cfg(transcribe_stall_seconds=2),
            argv_prefix=[sys.executable, "-c", script],
        )
    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while not _proc_dead(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _proc_dead(grandchild)


def test_nonzero_exit_is_a_job_failure() -> None:
    with pytest.raises(JobFailed, match="boom"):
        transcribe_isolated(
            "/tmp/a.wav",
            _cfg(),
            argv_prefix=[
                sys.executable,
                "-c",
                "import sys; print('boom', file=sys.stderr); sys.exit(3)",
            ],
        )
    assert _lock_is_free()


def test_busy_gpu_with_no_waiting_room_fails_fast() -> None:
    GPU_LOCK.acquire()
    try:
        with pytest.raises(QueueFull):
            transcribe_isolated(
                "/tmp/a.wav",
                _cfg(transcribe_max_waiting=0),
                argv_prefix=[sys.executable, "-c", OK_SCRIPT],
            )
    finally:
        GPU_LOCK.release()


def test_waiting_for_busy_gpu_times_out() -> None:
    GPU_LOCK.acquire()
    try:
        with pytest.raises(QueueTimeout):
            transcribe_isolated(
                "/tmp/a.wav",
                _cfg(transcribe_queue_wait_seconds=1),
                argv_prefix=[sys.executable, "-c", OK_SCRIPT],
            )
    finally:
        GPU_LOCK.release()
