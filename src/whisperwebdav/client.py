"""HTTP transcription client — the thin-client side of TRANSCRIBE_BACKEND=http.

Lets the poll loop offload inference to a whisperwebdav-server instead of loading the model
itself (so the poller needs no torch/GPU). Requests verbose_json and returns the segment
dicts, which the watcher then renders into its configured OUTPUT_FORMATS locally.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from .config import Config


def transcribe_remote(audio_path: str, config: Config) -> list[dict]:
    """POST one audio file to the configured server, returning {start, end, text} segments."""
    url = config.transcribe_server_url.rstrip("/") + "/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    data = {
        "model": config.transcription_model,
        "language": config.language,
        "response_format": "verbose_json",
    }
    with open(audio_path, "rb") as fh:
        files = {"file": (Path(audio_path).name, fh)}
        # No total timeout: a long recording can take minutes; rely on the connection drop.
        resp = httpx.post(
            url, data=data, files=files, headers=headers, timeout=httpx.Timeout(None, connect=10.0)
        )
    resp.raise_for_status()
    return resp.json()["segments"]
