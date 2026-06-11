from __future__ import annotations

import shutil
import signal
import tempfile
import time
from pathlib import Path

import structlog

from .client import transcribe_remote
from .config import Config
from .formatter import FORMAT_EXTENSIONS, format_output
from .notifier import Notifier
from .transcriber import BatchTranscriptionResult, release_gpu_memory, transcribe_batch
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


def _publish_results(
    filename: str,
    segments: list,
    webdav: WebDAVClient,
    config: Config,
    notifier: Notifier,
) -> None:
    """Format, upload, and mark done for a single transcribed file."""
    stem = Path(filename).stem
    formats = config.output_formats_list

    for fmt in formats:
        content = format_output(segments, fmt)
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


def process_batch(
    filenames: list[str],
    webdav: WebDAVClient,
    config: Config,
    notifier: Notifier,
) -> None:
    """Download, transcribe, and publish a batch of audio files.

    Errors are isolated per file: a single download or upload failure does not
    block the rest of the batch. If transcription itself fails, every file in
    the batch is reported as failed (no .done markers written, retried next poll).
    """
    if not filenames:
        return

    download_dir = tempfile.mkdtemp()
    local_paths: list[str] = []
    filename_by_local: dict[str, str] = {}
    result: BatchTranscriptionResult | None = None

    try:
        for filename in filenames:
            local_path = str(Path(download_dir) / filename)
            try:
                webdav.download(filename, local_path)
            except Exception as exc:
                log.exception("Download failed", filename=filename)
                notifier.notify_failure(filename, exc)
                continue
            local_paths.append(local_path)
            filename_by_local[local_path] = filename

        if not local_paths:
            return

        segments_by_path: dict[str, list] = {}
        if config.transcribe_backend == "http":
            # Thin-client mode: offload each file to the model-owner server. Per-file isolation
            # — one failed request doesn't sink the rest of the batch.
            for local_path in local_paths:
                filename = filename_by_local[local_path]
                try:
                    segments_by_path[local_path] = transcribe_remote(local_path, config)
                except Exception as exc:
                    log.exception("Remote transcription failed", filename=filename)
                    notifier.notify_failure(filename, exc)
        else:
            try:
                result = transcribe_batch(local_paths, config)
            except Exception as exc:
                log.exception(
                    "Batch transcription failed",
                    batch_size=len(local_paths),
                    filenames=list(filename_by_local.values()),
                )
                for filename in filename_by_local.values():
                    notifier.notify_failure(filename, exc)
                return
            segments_by_path = result.segments_by_path

        for local_path, segments in segments_by_path.items():
            filename = filename_by_local[local_path]
            try:
                _publish_results(filename, segments, webdav, config, notifier)
                log.info("File processed successfully", filename=filename)
            except Exception as exc:
                log.exception("Failed to publish results", filename=filename)
                notifier.notify_failure(filename, exc)

    finally:
        if result is not None:
            shutil.rmtree(result.workspace, ignore_errors=True)
        shutil.rmtree(download_dir, ignore_errors=True)


def poll(webdav: WebDAVClient, config: Config, notifier: Notifier) -> bool:
    """Poll the WebDAV share for new audio files and process them in batches.

    Returns True if any work was done this cycle, False otherwise.
    """
    audio_files = webdav.list_audio_files()
    log.debug("Poll cycle", file_count=len(audio_files))

    pending: list[str] = []
    for filename in audio_files:
        if webdav.done_marker_exists(filename):
            log.debug("Skipping already-done file", filename=filename)
            continue
        pending.append(filename)

    if not pending:
        return False

    batch_size = config.max_batch_size
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        try:
            process_batch(chunk, webdav, config, notifier)
        except Exception:
            log.exception("Error processing batch, continuing", batch=chunk)
    return True


def main() -> None:
    config = Config()
    _configure_logging(config)

    if not config.webdav_url:
        raise SystemExit("WEBDAV_URL is required to run the poll loop")

    log.info(
        "Starting WhisperWebDAV",
        webdav_url=config.webdav_url,
        watch_path=config.webdav_watch_path,
        poll_interval=config.poll_interval_seconds,
        max_batch_size=config.max_batch_size,
        backend=config.transcribe_backend,
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

    last_work_at = time.monotonic()
    gpu_released = False
    idle_threshold = config.gpu_idle_release_seconds

    while running:
        did_work = False
        try:
            did_work = poll(webdav, config, notifier)
        except Exception:
            log.exception("Unexpected error in poll loop")

        if did_work:
            last_work_at = time.monotonic()
            gpu_released = False
        elif (
            config.gpu_enabled
            and not gpu_released
            and idle_threshold > 0
            and time.monotonic() - last_work_at >= idle_threshold
        ):
            log.info("Releasing idle GPU memory", idle_seconds=idle_threshold)
            release_gpu_memory()
            gpu_released = True

        # Sliced sleep for fast SIGTERM response
        for _ in range(config.poll_interval_seconds):
            if not running:
                break
            time.sleep(1)

    log.info("WhisperWebDAV stopped")
