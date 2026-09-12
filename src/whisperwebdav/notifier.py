from __future__ import annotations

import structlog

from .config import Config

log = structlog.get_logger(__name__)


class Notifier:
    def __init__(self, config: Config) -> None:
        import apprise  # type: ignore[import]

        self._apprise = apprise.Apprise()
        urls = config.apprise_urls_list
        added = 0
        for url in urls:
            if self._apprise.add(url):
                added += 1
            else:
                log.error("Failed to register Apprise URL", url=url)
        self._enabled = added > 0
        if urls and not self._enabled:
            log.error("No Apprise URLs registered successfully", configured=len(urls))

    def _send(self, title: str, body: str) -> None:
        ok = self._apprise.notify(title=title, body=body)
        if not ok:
            log.error("Apprise notify() reported failure", title=title)

    def notify_success(self, filename: str, formats: list[str]) -> None:
        """Send a success notification. No-op if no URLs are configured."""
        if not self._enabled:
            return
        fmt_list = ", ".join(formats)
        self._send(
            "Transcription complete",
            f"File '{filename}' transcribed successfully.\nFormats: {fmt_list}",
        )

    def notify_failure(self, filename: str, error: Exception) -> None:
        """Send a failure notification. No-op if no URLs are configured."""
        if not self._enabled:
            return
        self._send(
            "Transcription failed",
            f"Failed to transcribe '{filename}'.\nError: {error}",
        )

    def notify_quarantined(self, filename: str, attempts: int, error: Exception) -> None:
        """Send a one-time quarantine notification. No-op if no URLs are configured.

        Unlike notify_failure (sent on every failed attempt), this fires exactly once, when
        a file crosses max_retries and is given up on for good -- so a human still finds out,
        even though the poll loop will now stay silent about that file forever.
        """
        if not self._enabled:
            return
        self._send(
            "Transcription permanently failed — quarantined",
            f"'{filename}' failed {attempts} times and will not be retried again.\n"
            f"Last error: {error}\n"
            "Delete its .quarantined and .failcount sidecars on the share to retry.",
        )
