from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whisperwebdav.watcher import poll, process_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_webdav(audio_files=None, done_files=None):
    """Create a mock WebDAVClient."""
    webdav = MagicMock()
    webdav.list_audio_files.return_value = audio_files or []
    done_files = done_files or set()
    webdav.done_marker_exists.side_effect = lambda f: f in done_files
    return webdav


# ---------------------------------------------------------------------------
# TestPoll
# ---------------------------------------------------------------------------


class TestPoll:
    def test_skips_done_files(self, minimal_config, mock_segments):
        webdav = make_mock_webdav(
            audio_files=["a.mp3", "b.mp3"],
            done_files={"a.mp3", "b.mp3"},
        )
        notifier = MagicMock()

        poll(webdav, minimal_config, notifier)

        webdav.download.assert_not_called()

    def test_processes_new_files(self, minimal_config, mock_segments):
        webdav = make_mock_webdav(audio_files=["new.mp3"])
        notifier = MagicMock()

        fake_result = MagicMock()
        fake_result.segments = mock_segments
        fake_result.workspace = Path(tempfile.mkdtemp())

        with patch("whisperwebdav.watcher.process_file") as mock_process:
            poll(webdav, minimal_config, notifier)
            mock_process.assert_called_once()
            call_args = mock_process.call_args
            assert call_args[0][0] == "new.mp3"

        # Clean up the fake workspace
        shutil.rmtree(fake_result.workspace, ignore_errors=True)

    def test_continues_after_per_file_error(self, minimal_config):
        webdav = make_mock_webdav(audio_files=["bad.mp3", "good.mp3"])
        notifier = MagicMock()

        call_count = 0

        def side_effect(filename, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if filename == "bad.mp3":
                raise RuntimeError("Transcription failed")

        with patch("whisperwebdav.watcher.process_file", side_effect=side_effect):
            poll(webdav, minimal_config, notifier)

        assert call_count == 2

    def test_handles_empty_directory(self, minimal_config):
        webdav = make_mock_webdav(audio_files=[])
        notifier = MagicMock()

        poll(webdav, minimal_config, notifier)

        webdav.done_marker_exists.assert_not_called()
        webdav.download.assert_not_called()


# ---------------------------------------------------------------------------
# TestProcessFile
# ---------------------------------------------------------------------------


class TestProcessFile:
    def _run_process(self, filename, config, webdav, notifier, segments=None):
        """Helper to run process_file with a mocked transcribe."""
        if segments is None:
            from tests.conftest import mock_segments as _ms
            segments = [
                MagicMock(start=0.0, end=1.0, text="Hello"),
                MagicMock(start=1.0, end=2.0, text="World"),
            ]

        fake_workspace = Path(tempfile.mkdtemp())
        fake_result = MagicMock()
        fake_result.segments = segments
        fake_result.workspace = fake_workspace

        local_audio_dir = tempfile.mkdtemp()
        try:
            with patch("whisperwebdav.watcher.transcribe", return_value=fake_result):
                yield local_audio_dir
        finally:
            shutil.rmtree(local_audio_dir, ignore_errors=True)
            shutil.rmtree(fake_workspace, ignore_errors=True)

    def test_uploads_each_format(self, minimal_config, mock_segments):
        minimal_config.__dict__["output_formats"] = "txt,srt"
        webdav = MagicMock()
        notifier = MagicMock()

        fake_workspace = Path(tempfile.mkdtemp())
        fake_result = MagicMock()
        fake_result.segments = mock_segments
        fake_result.workspace = fake_workspace

        local_dir = tempfile.mkdtemp()
        try:
            with patch("whisperwebdav.watcher.transcribe", return_value=fake_result):
                process_file("test.mp3", webdav, minimal_config, notifier, local_dir)
        finally:
            shutil.rmtree(local_dir, ignore_errors=True)
            shutil.rmtree(fake_workspace, ignore_errors=True)

        upload_calls = [c[0][1] for c in webdav.upload_string.call_args_list]
        assert any(n.endswith(".txt") for n in upload_calls)
        assert any(n.endswith(".srt") for n in upload_calls)

    def test_notifies_success(self, minimal_config, mock_segments):
        webdav = MagicMock()
        notifier = MagicMock()

        fake_workspace = Path(tempfile.mkdtemp())
        fake_result = MagicMock()
        fake_result.segments = mock_segments
        fake_result.workspace = fake_workspace

        local_dir = tempfile.mkdtemp()
        try:
            with patch("whisperwebdav.watcher.transcribe", return_value=fake_result):
                process_file("audio.mp3", webdav, minimal_config, notifier, local_dir)
        finally:
            shutil.rmtree(local_dir, ignore_errors=True)
            shutil.rmtree(fake_workspace, ignore_errors=True)

        notifier.notify_success.assert_called_once()
        call_args = notifier.notify_success.call_args[0]
        assert call_args[0] == "audio.mp3"

    def test_notifies_and_raises_on_upload_failure(self, minimal_config, mock_segments):
        webdav = MagicMock()
        webdav.upload_string.side_effect = OSError("Upload failed")
        notifier = MagicMock()

        fake_workspace = Path(tempfile.mkdtemp())
        fake_result = MagicMock()
        fake_result.segments = mock_segments
        fake_result.workspace = fake_workspace

        local_dir = tempfile.mkdtemp()
        try:
            with patch("whisperwebdav.watcher.transcribe", return_value=fake_result):
                with pytest.raises(OSError, match="Upload failed"):
                    process_file("fail.mp3", webdav, minimal_config, notifier, local_dir)
        finally:
            shutil.rmtree(local_dir, ignore_errors=True)
            shutil.rmtree(fake_workspace, ignore_errors=True)

        notifier.notify_failure.assert_called_once()
        assert notifier.notify_success.call_count == 0

    def test_cleans_workspace_on_error(self, minimal_config, mock_segments):
        webdav = MagicMock()
        webdav.upload_string.side_effect = RuntimeError("boom")
        notifier = MagicMock()

        fake_workspace = Path(tempfile.mkdtemp())
        assert fake_workspace.exists()

        fake_result = MagicMock()
        fake_result.segments = mock_segments
        fake_result.workspace = fake_workspace

        local_dir = tempfile.mkdtemp()
        try:
            with patch("whisperwebdav.watcher.transcribe", return_value=fake_result):
                with pytest.raises(RuntimeError):
                    process_file("err.mp3", webdav, minimal_config, notifier, local_dir)
        finally:
            shutil.rmtree(local_dir, ignore_errors=True)

        assert not fake_workspace.exists()

    def test_prefixes_output_subdir(self, minimal_config, mock_segments):
        minimal_config.__dict__["output_subdir"] = "transcripts"
        webdav = MagicMock()
        notifier = MagicMock()

        fake_workspace = Path(tempfile.mkdtemp())
        fake_result = MagicMock()
        fake_result.segments = mock_segments
        fake_result.workspace = fake_workspace

        local_dir = tempfile.mkdtemp()
        try:
            with patch("whisperwebdav.watcher.transcribe", return_value=fake_result):
                process_file("sub.mp3", webdav, minimal_config, notifier, local_dir)
        finally:
            shutil.rmtree(local_dir, ignore_errors=True)
            shutil.rmtree(fake_workspace, ignore_errors=True)

        upload_calls = [c[0][1] for c in webdav.upload_string.call_args_list]
        assert all(name.startswith("transcripts/") for name in upload_calls)
