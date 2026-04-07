"""End-to-end integration test.

Spins up a real WsgiDAV server on localhost, downloads the Axel Pettersson
recording from Wikimedia (cached in tests/fixtures/), runs the full poll()
pipeline with KBLab/kb-whisper-tiny, and asserts that output files and a
.done marker appear on the WebDAV share.

Run with:
    uv run pytest tests/test_integration.py -v -s

Skipped automatically when the SKIP_INTEGRATION env var is set, or when
the audio fixture cannot be fetched (e.g. no network in CI).
"""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIO_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/0/0f/"
    "Axel_Pettersson_r%C3%B6stinspelning.ogg"
)
FIXTURE_DIR = Path(__file__).parent / "fixtures"
AUDIO_FIXTURE = FIXTURE_DIR / "axel_pettersson.ogg"

TINY_MODEL = "KBLab/kb-whisper-tiny"
TINY_EMISSIONS = "KBLab/wav2vec2-large-voxrex-swedish"  # correct model ID

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fetch_audio_fixture() -> Path:
    """Download the test clip once and cache it in tests/fixtures/."""
    FIXTURE_DIR.mkdir(exist_ok=True)
    if not AUDIO_FIXTURE.exists():
        urllib.request.urlretrieve(AUDIO_URL, AUDIO_FIXTURE)  # noqa: S310
    return AUDIO_FIXTURE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def audio_fixture() -> Path:
    """Return path to the cached Axel Pettersson .ogg clip."""
    try:
        return _fetch_audio_fixture()
    except Exception as exc:
        pytest.skip(f"Could not fetch audio fixture: {exc}")


@pytest.fixture(scope="module")
def webdav_server(tmp_path_factory):
    """Start a WsgiDAV server backed by a temp directory. Yields (url, root)."""
    from cheroot import wsgi as cheroot_wsgi
    from wsgidav.app import WsgiDAVApp

    root = tmp_path_factory.mktemp("webdav_root")
    port = _free_port()

    app = WsgiDAVApp(
        {
            "host": "127.0.0.1",
            "port": port,
            "root_path": str(root),
            "provider_mapping": {"/": str(root)},
            "simple_dc": {"user_mapping": {"*": {"testuser": {"password": "testpass"}}}},
            "verbose": 0,
            "logging": {"enable_loggers": []},
        }
    )

    server = cheroot_wsgi.Server(("127.0.0.1", port), app)
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()

    # Wait until port is accepting connections
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)

    yield f"http://127.0.0.1:{port}", root

    server.stop()


@pytest.fixture
def integration_config(monkeypatch, webdav_server):
    """Config pointed at the local WsgiDAV server with kb-whisper-tiny."""
    from whisperwebdav.config import Config

    url, _ = webdav_server
    monkeypatch.setenv("WEBDAV_URL", url)
    monkeypatch.setenv("WEBDAV_USERNAME", "testuser")
    monkeypatch.setenv("WEBDAV_PASSWORD", "testpass")
    monkeypatch.setenv("WEBDAV_WATCH_PATH", "/")
    monkeypatch.setenv("TRANSCRIPTION_MODEL", TINY_MODEL)
    monkeypatch.setenv("EMISSIONS_MODEL", TINY_EMISSIONS)
    monkeypatch.setenv("VAD_MODEL", "silero")
    monkeypatch.setenv("LANGUAGE", "sv")
    monkeypatch.setenv("OUTPUT_FORMATS", "txt,srt,timestamps")
    monkeypatch.setenv("GPU_ENABLED", "false")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    return Config()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("SKIP_INTEGRATION") == "1",
    reason="SKIP_INTEGRATION=1",
)
class TestEndToEnd:
    def test_full_poll_cycle(self, audio_fixture, webdav_server, integration_config):
        """Upload audio to WebDAV, run poll(), verify outputs + .done marker."""
        from whisperwebdav.notifier import Notifier
        from whisperwebdav.webdav import WebDAVClient
        from whisperwebdav.watcher import poll

        _, root = webdav_server

        # Place the audio file in the WebDAV root
        dest = root / audio_fixture.name
        shutil.copy(audio_fixture, dest)

        webdav = WebDAVClient(integration_config)
        notifier = Notifier(integration_config)

        poll(webdav, integration_config, notifier)

        stem = audio_fixture.stem
        files = {f.name for f in root.iterdir()}

        # .done marker must exist
        assert f"{stem}.done" in files, f".done marker missing; files: {files}"

        # At least txt output must exist
        assert f"{stem}.txt" in files, f".txt output missing; files: {files}"

        # srt output
        assert f"{stem}.srt" in files, f".srt output missing; files: {files}"

        # timestamps output (suffix _timestamps.txt)
        assert f"{stem}_timestamps.txt" in files, (
            f"_timestamps.txt output missing; files: {files}"
        )

        # txt output must be non-empty
        txt_content = (root / f"{stem}.txt").read_text(encoding="utf-8")
        assert txt_content.strip(), "Transcription output is empty"

    def test_done_file_skipped_on_second_poll(
        self, audio_fixture, webdav_server, integration_config
    ):
        """A second poll() must not re-process a file that already has a .done marker."""
        from unittest.mock import patch

        from whisperwebdav.notifier import Notifier
        from whisperwebdav.webdav import WebDAVClient
        from whisperwebdav.watcher import poll

        _, root = webdav_server

        # .done already exists from the previous test (same module-scoped server)
        webdav = WebDAVClient(integration_config)
        notifier = Notifier(integration_config)

        with patch("whisperwebdav.watcher.process_file") as mock_pf:
            poll(webdav, integration_config, notifier)
            mock_pf.assert_not_called()
