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
