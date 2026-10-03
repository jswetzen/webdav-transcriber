from __future__ import annotations

import pytest
from pydantic import ValidationError

from whisperwebdav.config import Config

# Explicit kwargs override env, so these don't depend on the ambient shell environment.


def test_no_webdav_is_valid_for_server() -> None:
    # The server process runs with no WebDAV config at all.
    cfg = Config(webdav_url="")
    assert cfg.webdav_url == ""
    assert cfg.transcribe_backend == "local"


def test_webdav_url_without_auth_raises() -> None:
    with pytest.raises(ValidationError, match="webdav_username"):
        Config(webdav_url="http://x", webdav_username="", webdav_password="", webdav_token="")


def test_webdav_url_with_token_ok() -> None:
    cfg = Config(webdav_url="http://x", webdav_token="t")
    assert cfg.webdav_token == "t"


def test_http_backend_requires_server_url() -> None:
    with pytest.raises(ValidationError, match="transcribe_server_url"):
        Config(transcribe_backend="http", transcribe_server_url="")


def test_http_backend_with_server_url_ok() -> None:
    cfg = Config(transcribe_backend="http", transcribe_server_url="http://srv:8000")
    assert cfg.transcribe_server_url == "http://srv:8000"


@pytest.mark.parametrize("value", [0, -1])
def test_max_upload_bytes_must_be_positive(value: int) -> None:
    with pytest.raises(ValidationError):
        Config(max_upload_bytes=value)


def test_max_upload_bytes_default_is_2_gib() -> None:
    assert Config().max_upload_bytes == 2 * 1024**3
