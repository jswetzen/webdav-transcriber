from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from whisperwebdav.config import Config
from whisperwebdav.transcriber import (
    BatchTranscriptionResult,
    _is_empty_vad_indexerror,
    transcribe_batch,
)


# ---------------------------------------------------------------------------
# _is_empty_vad_indexerror
# ---------------------------------------------------------------------------
#
# Incident 2026-09-12: four whisper-server files 500ed forever (~1000x/day/file) because
# easyaligner 0.2.3's Silero VAD wrapper indexes into an empty segment list (merge_chunks,
# easyaligner/vad/silero.py:45) when VAD finds zero speech in a file (music, silence, a
# near-inaudible recording). transcribe_batch works around it for the single-file batch
# case; these tests pin the detector against the REAL installed dependency rather than a
# fabricated traceback, so an easyaligner upgrade that fixes the underlying bug fails
# test_matches_real_easyaligner_empty_vad_crash instead of silently leaving a dead
# workaround in place.


def test_matches_real_easyaligner_empty_vad_crash():
    from easyaligner.vad.silero import merge_chunks

    try:
        merge_chunks([])
    except IndexError as exc:
        assert _is_empty_vad_indexerror(exc)
    else:
        pytest.fail(
            "merge_chunks([]) no longer raises IndexError -- easyaligner may have fixed "
            "the empty-VAD-segments bug; the transcribe_batch workaround can likely be "
            "removed (see _is_empty_vad_indexerror's docstring)"
        )


def test_does_not_match_unrelated_indexerror():
    try:
        [][0]
    except IndexError as exc:
        assert not _is_empty_vad_indexerror(exc)


def test_does_not_match_non_indexerror():
    assert not _is_empty_vad_indexerror(ValueError("not an IndexError at all"))


# ---------------------------------------------------------------------------
# transcribe_batch's handling of the VAD-empty crash
# ---------------------------------------------------------------------------


def _config(**overrides) -> Config:
    return Config(webdav_url="", **overrides)


def _touch(path: Path) -> str:
    path.write_bytes(b"fake audio")
    return str(path)


class TestTranscribeBatchEmptyVad:
    def test_single_file_returns_empty_segments_instead_of_raising(self, tmp_path):
        """The one shape that actually happens in production: the server calls
        transcribe_batch with exactly one file (TRANSCRIBE_BACKEND=http is always a
        singleton batch). A VAD-empty crash there unambiguously means *this* file had no
        detectable speech -- a valid empty-transcript outcome, not a server error."""
        audio_path = _touch(tmp_path / "silent.m4a")

        from easyaligner.vad.silero import merge_chunks

        def fake_pipeline(*args, **kwargs):
            merge_chunks([])  # raises the real IndexError, from the real file

        with (
            patch("easytranscriber.pipelines.pipeline", side_effect=fake_pipeline),
            patch("easyaligner.text.load_tokenizer", return_value=object()),
            patch("easyaligner.text.text_normalizer", return_value=None),
        ):
            result = transcribe_batch([audio_path], _config())

        assert isinstance(result, BatchTranscriptionResult)
        assert result.segments_by_path == {audio_path: []}

    def test_multi_file_batch_still_raises(self, tmp_path):
        """With more than one file we can't tell which one caused a batch-wide VAD crash
        without re-running per file, so the existing all-fail-together behavior stands."""
        a = _touch(tmp_path / "a.m4a")
        b = _touch(tmp_path / "b.m4a")

        from easyaligner.vad.silero import merge_chunks

        def fake_pipeline(*args, **kwargs):
            merge_chunks([])

        with (
            patch("easytranscriber.pipelines.pipeline", side_effect=fake_pipeline),
            patch("easyaligner.text.load_tokenizer", return_value=object()),
            patch("easyaligner.text.text_normalizer", return_value=None),
        ):
            with pytest.raises(IndexError):
                transcribe_batch([a, b], _config())

    def test_unrelated_indexerror_still_raises(self, tmp_path):
        """Only the specific known easyaligner VAD-empty crash is swallowed -- any other
        IndexError (a real bug, elsewhere in the pipeline) must still propagate."""
        audio_path = _touch(tmp_path / "x.m4a")

        def fake_pipeline(*args, **kwargs):
            raise IndexError("some unrelated indexing bug")

        with (
            patch("easytranscriber.pipelines.pipeline", side_effect=fake_pipeline),
            patch("easyaligner.text.load_tokenizer", return_value=object()),
            patch("easyaligner.text.text_normalizer", return_value=None),
        ):
            with pytest.raises(IndexError):
                transcribe_batch([audio_path], _config())

    def test_workspace_cleaned_up_on_unhandled_error(self, tmp_path):
        audio_path = _touch(tmp_path / "x.m4a")
        captured = {}

        real_mkdtemp = tempfile.mkdtemp

        def spy_mkdtemp(*args, **kwargs):
            d = real_mkdtemp(*args, **kwargs)
            captured["workspace"] = Path(d)
            return d

        def fake_pipeline(*args, **kwargs):
            raise IndexError("unrelated")

        with (
            patch("tempfile.mkdtemp", side_effect=spy_mkdtemp),
            patch("easytranscriber.pipelines.pipeline", side_effect=fake_pipeline),
            patch("easyaligner.text.load_tokenizer", return_value=object()),
            patch("easyaligner.text.text_normalizer", return_value=None),
        ):
            with pytest.raises(IndexError):
                transcribe_batch([audio_path], _config())

        assert not captured["workspace"].exists()
