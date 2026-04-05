from __future__ import annotations

import structlog

from .config import Config

log = structlog.get_logger(__name__)


class Notifier:
    def __init__(self, config: Config) -> None:
        import apprise  # type: ignore[import]

        self._apprise = apprise.Apprise()
        urls = config.apprise_urls_list
        for url in urls:
            self._apprise.add(url)
        self._enabled = bool(urls)

    def notify_success(self, filename: str, formats: list[str]) -> None:
        """Send a success notification. No-op if no URLs are configured."""
        if not self._enabled:
            return
        fmt_list = ", ".join(formats)
        title = "Transcription complete"
        body = f"File '{filename}' transcribed successfully.\nFormats: {fmt_list}"
        self._apprise.notify(title=title, body=body)

    def notify_failure(self, filename: str, error: Exception) -> None:
        """Send a failure notification. No-op if no URLs are configured."""
        if not self._enabled:
            return
        title = "Transcription failed"
        body = f"Failed to transcribe '{filename}'.\nError: {error}"
        self._apprise.notify(title=title, body=body)
