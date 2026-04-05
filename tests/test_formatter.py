from __future__ import annotations

import json

import pytest

from whisperwebdav.formatter import (
    _seconds_to_srt_time,
    format_output,
    to_json,
    to_srt,
    to_timestamps,
    to_txt,
)


# --- _seconds_to_srt_time ---


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0.0, "00:00:00,000"),
        (1.5, "00:00:01,500"),
        (61.0, "00:01:01,000"),
        (3661.001, "01:01:01,001"),
        (3599.999, "00:59:59,999"),
        (7322.123, "02:02:02,123"),
        # Floating point tricky case — use integer arithmetic
        (0.1, "00:00:00,100"),
        (0.999, "00:00:00,999"),
    ],
)
def test_seconds_to_srt_time(seconds: float, expected: str) -> None:
    assert _seconds_to_srt_time(seconds) == expected


# --- to_txt ---


def test_to_txt_attribute_style(mock_segments):
    result = to_txt(mock_segments)
    assert result == "Hello world\nHow are you"


def test_to_txt_dict_style():
    segs = [{"start": 0.0, "end": 1.0, "text": "First"}, {"start": 1.0, "end": 2.0, "text": "Second"}]
    assert to_txt(segs) == "First\nSecond"


def test_to_txt_empty():
    assert to_txt([]) == ""


# --- to_srt ---


def test_to_srt_attribute_style(mock_segments):
    result = to_srt(mock_segments)
    lines = result.split("\n\n")
    assert len(lines) == 2
    assert lines[0].startswith("1\n")
    assert "00:00:00,000 --> 00:00:03,500" in lines[0]
    assert "Hello world" in lines[0]
    assert lines[1].startswith("2\n")
    assert "00:00:04,000 --> 00:00:07,250" in lines[1]
    assert "How are you" in lines[1]


def test_to_srt_dict_style():
    segs = [{"start": 0.0, "end": 1.0, "text": "Line one"}]
    result = to_srt(segs)
    assert "1\n" in result
    assert "00:00:00,000 --> 00:00:01,000" in result
    assert "Line one" in result


def test_to_srt_empty():
    assert to_srt([]) == ""


# --- to_json ---


def test_to_json_attribute_style(mock_segments):
    result = to_json(mock_segments)
    data = json.loads(result)
    assert len(data) == 2
    assert data[0]["text"] == "Hello world"
    assert data[0]["start"] == 0.0
    assert data[0]["end"] == 3.5


def test_to_json_dict_style():
    segs = [{"start": 1.0, "end": 2.0, "text": "word", "extra": "data"}]
    result = to_json(segs)
    data = json.loads(result)
    assert data[0]["text"] == "word"
    assert data[0]["extra"] == "data"


# --- to_timestamps ---


def test_to_timestamps_attribute_style(mock_segments):
    result = to_timestamps(mock_segments)
    lines = result.split("\n")
    assert lines[0] == "[00:00:00] Hello world"
    assert lines[1] == "[00:00:04] How are you"


def test_to_timestamps_dict_style():
    segs = [{"start": 3661.0, "end": 3662.0, "text": "Late segment"}]
    result = to_timestamps(segs)
    assert result == "[01:01:01] Late segment"


def test_to_timestamps_empty():
    assert to_timestamps([]) == ""


# --- format_output dispatch ---


def test_format_output_txt(mock_segments):
    result = format_output(mock_segments, "txt")
    assert "Hello world" in result
    assert "How are you" in result


def test_format_output_srt(mock_segments):
    result = format_output(mock_segments, "srt")
    assert "-->" in result


def test_format_output_json(mock_segments):
    result = format_output(mock_segments, "json")
    data = json.loads(result)
    assert isinstance(data, list)


def test_format_output_timestamps(mock_segments):
    result = format_output(mock_segments, "timestamps")
    assert result.startswith("[")


def test_format_output_unknown_raises():
    with pytest.raises(ValueError, match="Unknown format"):
        format_output([], "xml")
