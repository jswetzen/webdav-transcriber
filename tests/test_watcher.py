from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whisperwebdav.transcriber import BatchTranscriptionResult
from whisperwebdav.watcher import poll, process_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_webdav(audio_files=None, done_files=None):
    webdav = MagicMock()
    webdav.list_audio_files.return_value = audio_files or []
    done_files = done_files or set()
    webdav.done_marker_exists.side_effect = lambda f: f in done_files
    return webdav


def fake_batch_result(local_paths, segments_per_file):
    workspace = Path(tempfile.mkdtemp())
    return BatchTranscriptionResult(
        segments_by_path={p: segments_per_file for p in local_paths},
        workspace=workspace,
    )


# ---------------------------------------------------------------------------
# TestPoll
# ---------------------------------------------------------------------------


class TestPoll:
    def test_skips_done_files(self, minimal_config):
        webdav = make_mock_webdav(
            audio_files=["a.mp3", "b.mp3"],
            done_files={"a.mp3", "b.mp3"},
        )
        notifier = MagicMock()

        with patch("whisperwebdav.watcher.process_batch") as mock_process:
            poll(webdav, minimal_config, notifier)
            mock_process.assert_not_called()

    def test_processes_new_files(self, minimal_config):
        webdav = make_mock_webdav(audio_files=["new.mp3"])
        notifier = MagicMock()

        with patch("whisperwebdav.watcher.process_batch") as mock_process:
            poll(webdav, minimal_config, notifier)
            mock_process.assert_called_once()
            chunk = mock_process.call_args[0][0]
            assert chunk == ["new.mp3"]

    def test_continues_after_per_batch_error(self, minimal_config):
        webdav = make_mock_webdav(audio_files=["a.mp3", "b.mp3"])
        notifier = MagicMock()
        minimal_config.__dict__["max_batch_size"] = 1

        call_count = 0

        def side_effect(chunk, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if chunk == ["a.mp3"]:
                raise RuntimeError("boom")

        with patch("whisperwebdav.watcher.process_batch", side_effect=side_effect):
            poll(webdav, minimal_config, notifier)

        assert call_count == 2

    def test_handles_empty_directory(self, minimal_config):
        webdav = make_mock_webdav(audio_files=[])
        notifier = MagicMock()

        with patch("whisperwebdav.watcher.process_batch") as mock_process:
            poll(webdav, minimal_config, notifier)
            mock_process.assert_not_called()

        webdav.done_marker_exists.assert_not_called()

    def test_chunks_by_max_batch_size(self, minimal_config):
        files = [f"f{i}.mp3" for i in range(17)]
        webdav = make_mock_webdav(audio_files=files)
        notifier = MagicMock()
        minimal_config.__dict__["max_batch_size"] = 8

        with patch("whisperwebdav.watcher.process_batch") as mock_process:
            poll(webdav, minimal_config, notifier)

        chunk_sizes = [len(c[0][0]) for c in mock_process.call_args_list]
        assert chunk_sizes == [8, 8, 1]


# ---------------------------------------------------------------------------
# TestProcessBatch
# ---------------------------------------------------------------------------


class TestProcessBatch:
    def test_uploads_each_format(self, minimal_config, mock_segments):
        minimal_config.__dict__["output_formats"] = "txt,srt"
        webdav = MagicMock()
        notifier = MagicMock()

        def fake_transcribe(local_paths, config):
            return fake_batch_result(local_paths, mock_segments)

        with patch(
            "whisperwebdav.watcher.transcribe_batch", side_effect=fake_transcribe
        ):
            process_batch(["test.mp3"], webdav, minimal_config, notifier)

        upload_names = [c[0][1] for c in webdav.upload_string.call_args_list]
        assert any(n.endswith(".txt") for n in upload_names)
        assert any(n.endswith(".srt") for n in upload_names)

    def test_marks_done_and_notifies_for_each_file(
        self, minimal_config, mock_segments
    ):
        webdav = MagicMock()
        notifier = MagicMock()

        def fake_transcribe(local_paths, config):
            return fake_batch_result(local_paths, mock_segments)

        with patch(
            "whisperwebdav.watcher.transcribe_batch", side_effect=fake_transcribe
        ):
            process_batch(
                ["a.mp3", "b.mp3", "c.mp3"], webdav, minimal_config, notifier
            )

        done_calls = [c[0][0] for c in webdav.create_done_marker.call_args_list]
        assert sorted(done_calls) == ["a.mp3", "b.mp3", "c.mp3"]
        assert notifier.notify_success.call_count == 3

    def test_transcribe_failure_notifies_all_no_done(
        self, minimal_config, mock_segments
    ):
        webdav = MagicMock()
        notifier = MagicMock()

        with patch(
            "whisperwebdav.watcher.transcribe_batch",
            side_effect=RuntimeError("transcribe boom"),
        ):
            process_batch(
                ["a.mp3", "b.mp3", "c.mp3"], webdav, minimal_config, notifier
            )

        webdav.create_done_marker.assert_not_called()
        assert notifier.notify_failure.call_count == 3
        assert notifier.notify_success.call_count == 0

    def test_per_file_upload_isolation(self, minimal_config, mock_segments):
        webdav = MagicMock()
        notifier = MagicMock()

        # Fail upload only for the 'bad.mp3' stem
        def upload_side_effect(content, remote_path):
            if remote_path.startswith("bad"):
                raise OSError("upload boom")

        webdav.upload_string.side_effect = upload_side_effect

        def fake_transcribe(local_paths, config):
            return fake_batch_result(local_paths, mock_segments)

        with patch(
            "whisperwebdav.watcher.transcribe_batch", side_effect=fake_transcribe
        ):
            process_batch(
                ["good1.mp3", "bad.mp3", "good2.mp3"],
                webdav,
                minimal_config,
                notifier,
            )

        done_calls = [c[0][0] for c in webdav.create_done_marker.call_args_list]
        assert sorted(done_calls) == ["good1.mp3", "good2.mp3"]
        assert notifier.notify_success.call_count == 2
        assert notifier.notify_failure.call_count == 1
        assert notifier.notify_failure.call_args[0][0] == "bad.mp3"

    def test_download_failure_skips_only_that_file(
        self, minimal_config, mock_segments
    ):
        webdav = MagicMock()
        notifier = MagicMock()

        def download_side_effect(filename, local_path):
            if filename == "bad.mp3":
                raise OSError("download boom")

        webdav.download.side_effect = download_side_effect

        def fake_transcribe(local_paths, config):
            return fake_batch_result(local_paths, mock_segments)

        with patch(
            "whisperwebdav.watcher.transcribe_batch", side_effect=fake_transcribe
        ) as mock_transcribe:
            process_batch(
                ["good.mp3", "bad.mp3"], webdav, minimal_config, notifier
            )

        # Only good.mp3 should reach transcribe
        passed_paths = mock_transcribe.call_args[0][0]
        assert len(passed_paths) == 1
        assert passed_paths[0].endswith("good.mp3")
        # bad.mp3 reported as failure, good.mp3 as success
        assert notifier.notify_failure.call_count == 1
        assert notifier.notify_failure.call_args[0][0] == "bad.mp3"
        assert notifier.notify_success.call_count == 1

    def test_cleans_workspace(self, minimal_config, mock_segments):
        webdav = MagicMock()
        notifier = MagicMock()

        captured = {}

        def fake_transcribe(local_paths, config):
            res = fake_batch_result(local_paths, mock_segments)
            captured["workspace"] = res.workspace
            return res

        with patch(
            "whisperwebdav.watcher.transcribe_batch", side_effect=fake_transcribe
        ):
            process_batch(["x.mp3"], webdav, minimal_config, notifier)

        assert not captured["workspace"].exists()

    def test_prefixes_output_subdir(self, minimal_config, mock_segments):
        minimal_config.__dict__["output_subdir"] = "transcripts"
        webdav = MagicMock()
        notifier = MagicMock()

        def fake_transcribe(local_paths, config):
            return fake_batch_result(local_paths, mock_segments)

        with patch(
            "whisperwebdav.watcher.transcribe_batch", side_effect=fake_transcribe
        ):
            process_batch(["sub.mp3"], webdav, minimal_config, notifier)

        upload_names = [c[0][1] for c in webdav.upload_string.call_args_list]
        assert all(n.startswith("transcripts/") for n in upload_names)

    def test_empty_batch_is_noop(self, minimal_config):
        webdav = MagicMock()
        notifier = MagicMock()

        with patch("whisperwebdav.watcher.transcribe_batch") as mock_transcribe:
            process_batch([], webdav, minimal_config, notifier)
            mock_transcribe.assert_not_called()

        webdav.download.assert_not_called()
