from __future__ import annotations

import os
import tempfile
from pathlib import Path

import structlog
from webdav3.client import Client

from .config import AUDIO_EXTENSIONS, Config

log = structlog.get_logger(__name__)


class WebDAVClient:
    def __init__(self, config: Config) -> None:
        self._watch_path = config.webdav_watch_path

        options: dict[str, str] = {
            "webdav_hostname": config.webdav_url,
        }

        if config.webdav_token:
            options["webdav_token"] = config.webdav_token
        else:
            options["webdav_login"] = config.webdav_username
            options["webdav_password"] = config.webdav_password

        self._client = Client(options)

    def check_connection(self) -> None:
        """List the watch path, raising on any failure (auth, network, missing path).

        Unlike list_audio_files (which swallows and returns [] to keep the poll loop
        alive through transient blips), this propagates: it backs the healthcheck
        probe, where a raised exception is the unhealthy signal.
        """
        self._client.list(self._watch_path)

    def list_audio_files(self) -> list[str]:
        """List audio files in the watch path. Returns [] on error."""
        try:
            entries = self._client.list(self._watch_path)
            results = []
            for entry in entries:
                # Skip directory entries (end with /) and the directory itself
                if entry.endswith("/"):
                    continue
                ext = Path(entry).suffix.lower()
                if ext in AUDIO_EXTENSIONS:
                    results.append(entry)
            return results
        except Exception:
            log.exception("Failed to list WebDAV directory", path=self._watch_path)
            return []

    def done_marker_exists(self, audio_filename: str) -> bool:
        """Check if a .done sidecar exists for the given audio file. Returns False on error."""
        try:
            stem = Path(audio_filename).stem
            marker = f"{stem}.done"
            remote_path = str(Path(self._watch_path) / marker)
            return self._client.check(remote_path)
        except Exception:
            log.exception("Failed to check done marker", filename=audio_filename)
            return False

    def download(self, remote_filename: str, local_path: str) -> None:
        """Download a file from the watch path."""
        remote_path = str(Path(self._watch_path) / remote_filename)
        self._client.download_sync(remote_path=remote_path, local_path=local_path)

    def upload(self, local_path: str, remote_filename: str) -> None:
        """Upload a file to the watch path."""
        remote_path = str(Path(self._watch_path) / remote_filename)
        self._client.upload_sync(remote_path=remote_path, local_path=local_path)

    def upload_string(self, content: str, remote_filename: str) -> None:
        """Write content to a temp file and upload it."""
        fd, tmp_path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            self.upload(tmp_path, remote_filename)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def create_done_marker(self, audio_filename: str) -> None:
        """Create a .done sidecar for the given audio file."""
        stem = Path(audio_filename).stem
        marker = f"{stem}.done"
        self.upload_string("", marker)

    def quarantined_marker_exists(self, audio_filename: str) -> bool:
        """Check if a .quarantined sidecar exists (given up after too many failures)."""
        try:
            stem = Path(audio_filename).stem
            marker = f"{stem}.quarantined"
            remote_path = str(Path(self._watch_path) / marker)
            return self._client.check(remote_path)
        except Exception:
            log.exception("Failed to check quarantine marker", filename=audio_filename)
            return False

    def create_quarantine_marker(self, audio_filename: str, reason: str) -> None:
        """Create a .quarantined sidecar recording why the file was given up on.

        Written once a file's failure count sidecar (see record_failure) crosses
        config.max_retries -- stops the poll loop from ever re-attempting it, so a file
        that cannot succeed doesn't re-queue forever (was ~1000x/day/file in production
        before this existed, see the 2026-09-12 incident notes in transcriber.py).
        """
        stem = Path(audio_filename).stem
        self.upload_string(reason, f"{stem}.quarantined")

    def get_failure_count(self, audio_filename: str) -> int:
        """Read the persisted failure count sidecar. Returns 0 if missing or unreadable.

        Persisted to WebDAV (not kept in-process) so the count survives a poller restart
        (redeploy, pull, crash) -- otherwise every restart would reset every file's count
        to 0 and a permanently-broken file would never actually reach quarantine.
        """
        stem = Path(audio_filename).stem
        remote_path = str(Path(self._watch_path) / f"{stem}.failcount")
        if not self._client.check(remote_path):
            return 0
        fd, tmp_path = tempfile.mkstemp()
        os.close(fd)
        try:
            self._client.download_sync(remote_path=remote_path, local_path=tmp_path)
            return int(Path(tmp_path).read_text().strip())
        except (OSError, ValueError):
            log.exception(
                "Failed to read failure count sidecar; treating as 0", filename=audio_filename
            )
            return 0
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def record_failure(self, audio_filename: str) -> int:
        """Increment and persist the failure count sidecar for a file. Returns the new count."""
        count = self.get_failure_count(audio_filename) + 1
        stem = Path(audio_filename).stem
        self.upload_string(str(count), f"{stem}.failcount")
        return count

    def clear_failure_count(self, audio_filename: str) -> None:
        """Remove the failure count sidecar, e.g. after an eventual success."""
        stem = Path(audio_filename).stem
        remote_path = str(Path(self._watch_path) / f"{stem}.failcount")
        try:
            if self._client.check(remote_path):
                self._client.clean(remote_path)
        except Exception:
            log.exception("Failed to clear failure count sidecar", filename=audio_filename)
