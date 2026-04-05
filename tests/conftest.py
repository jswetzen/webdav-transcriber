from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from whisperwebdav.config import Config


@pytest.fixture
def minimal_config(monkeypatch: pytest.MonkeyPatch) -> Config:
    """Provide a minimal valid Config with environment variables set."""
    monkeypatch.setenv("WEBDAV_URL", "http://webdav.example.com")
    monkeypatch.setenv("WEBDAV_USERNAME", "user")
    monkeypatch.setenv("WEBDAV_PASSWORD", "pass")
    return Config()


@pytest.fixture
def mock_segments() -> list[MagicMock]:
    """Two mock segments with start, end, and text attributes."""
    seg1 = MagicMock()
    seg1.start = 0.0
    seg1.end = 3.5
    seg1.text = "Hello world"

    seg2 = MagicMock()
    seg2.start = 4.0
    seg2.end = 7.25
    seg2.text = "How are you"

    return [seg1, seg2]
