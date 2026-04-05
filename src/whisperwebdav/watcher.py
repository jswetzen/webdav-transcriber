from __future__ import annotations

import shutil
import signal
import tempfile
import time
from pathlib import Path

import structlog

from .config import Config
from .formatter import FORMAT_EXTENSIONS, format_output
from .notifier import Notifier
from .transcriber import TranscriptionResult, transcribe
from .webdav import WebDAVClient

log = structlog.get_logger(__name__)


def _configure_logging(config: Config) -> None:
    import logging

    import structlog

    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if config.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def process_file(
    filename: str,
    webdav: WebDAVClient,
    config: Config,
    notifier: Notifier,
    local_audio_dir: str,
) -> None:
    """Download, transcribe, and upload results for a single audio file."""
    local_path = str(Path(local_audio_dir) / filename)
    result: TranscriptionResult | None = None

    try:
        webdav.download(filename, local_path)
        result = transcribe(local_path, config)

        stem = Path(filename).stem
        formats = config.output_formats_list

        for fmt in formats:
            content = format_output(result.segments, fmt)
            ext = FORMAT_EXTENSIONS.get(fmt, f".{fmt}")
            if fmt == "timestamps":
                output_name = f"{stem}_timestamps{ext}"
            else:
                output_name = f"{stem}{ext}"

            if config.output_subdir:
                remote_path = f"{config.output_subdir.rstrip('/')}/{output_name}"
            else:
                remote_path = output_name

            webdav.upload_string(content, remote_path)

        webdav.create_done_marker(filename)
        notifier.notify_success(filename, formats)
        log.info("File processed successfully", filename=filename)

    except Exception as exc:
        log.exception("Failed to process file", filename=filename)
        notifier.notify_failure(filename, exc)
        raise

    finally:
        if result is not None:
            shutil.rmtree(result.workspace, ignore_errors=True)
        try:
            Path(local_path).unlink(missing_ok=True)
        except OSError:
            pass


def poll(webdav: WebDAVClient, config: Config, notifier: Notifier) -> None:
    """Poll the WebDAV share for new audio files and process them."""
    audio_files = webdav.list_audio_files()
    log.debug("Poll cycle", file_count=len(audio_files))

    for filename in audio_files:
        if webdav.done_marker_exists(filename):
            log.debug("Skipping already-done file", filename=filename)
            continue

        download_dir = tempfile.mkdtemp()
        try:
            process_file(filename, webdav, config, notifier, download_dir)
        except Exception:
            log.exception("Error processing file, continuing", filename=filename)
        finally:
            shutil.rmtree(download_dir, ignore_errors=True)


def main() -> None:
    config = Config()
    _configure_logging(config)

    log.info(
        "Starting WhisperWebDAV",
        webdav_url=config.webdav_url,
        watch_path=config.webdav_watch_path,
        poll_interval=config.poll_interval_seconds,
        model=config.transcription_model,
    )

    webdav = WebDAVClient(config)
    notifier = Notifier(config)

    running = True

    def _stop(signum: int, frame: object) -> None:
        nonlocal running
        log.info("Shutdown signal received", signal=signum)
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while running:
        try:
            poll(webdav, config, notifier)
        except Exception:
            log.exception("Unexpected error in poll loop")

        # Sliced sleep for fast SIGTERM response
        for _ in range(config.poll_interval_seconds):
            if not running:
                break
            time.sleep(1)

    log.info("WhisperWebDAV stopped")
