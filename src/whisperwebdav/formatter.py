from __future__ import annotations

import json
from typing import Any


def _get(seg: Any, key: str) -> Any:
    """Get attribute from segment using attribute-style or dict-style access."""
    if isinstance(seg, dict):
        return seg[key]
    return getattr(seg, key)


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm using integer arithmetic."""
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format (no milliseconds) using integer arithmetic."""
    total_s = int(seconds)
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def to_txt(segments: Any) -> str:
    """One segment text per line."""
    lines = [_get(seg, "text") for seg in segments]
    return "\n".join(lines)


def to_srt(segments: Any) -> str:
    """Numbered SRT blocks with HH:MM:SS,mmm --> HH:MM:SS,mmm timestamps."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = _seconds_to_srt_time(_get(seg, "start"))
        end = _seconds_to_srt_time(_get(seg, "end"))
        text = _get(seg, "text")
        blocks.append(f"{i}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks)


def to_json(segments: Any) -> str:
    """Raw alignment JSON (word-level timestamps)."""
    data = []
    for seg in segments:
        if isinstance(seg, dict):
            data.append(seg)
        else:
            data.append(
                {
                    "start": _get(seg, "start"),
                    "end": _get(seg, "end"),
                    "text": _get(seg, "text"),
                }
            )
    return json.dumps(data, ensure_ascii=False, indent=2)


def to_timestamps(segments: Any) -> str:
    """[HH:MM:SS] text per segment (no milliseconds)."""
    lines = []
    for seg in segments:
        ts = _seconds_to_timestamp(_get(seg, "start"))
        text = _get(seg, "text")
        lines.append(f"[{ts}] {text}")
    return "\n".join(lines)


FORMAT_FUNCTIONS: dict[str, Any] = {
    "txt": to_txt,
    "srt": to_srt,
    "json": to_json,
    "timestamps": to_timestamps,
}

FORMAT_EXTENSIONS: dict[str, str] = {
    "txt": ".txt",
    "srt": ".srt",
    "json": ".json",
    "timestamps": ".txt",
}


def format_output(segments: Any, fmt: str) -> str:
    """Dispatch to the appropriate format function."""
    if fmt not in FORMAT_FUNCTIONS:
        supported = ", ".join(sorted(FORMAT_FUNCTIONS.keys()))
        raise ValueError(f"Unknown format '{fmt}'. Supported formats: {supported}")
    return FORMAT_FUNCTIONS[fmt](segments)
