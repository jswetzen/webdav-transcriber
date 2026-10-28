from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from whisperwebdav.config import Config
from whisperwebdav.webdav import WebDAVClient


@pytest.fixture
def webdav(minimal_config: Config) -> WebDAVClient:
    client = WebDAVClient(minimal_config)
    client._client = MagicMock()
    return client


def _tree(webdav: WebDAVClient, tree: dict[str, list[str]]) -> None:
    webdav._client.list.side_effect = lambda path: tree[path]


def test_lists_audio_in_subfolders(webdav: WebDAVClient) -> None:
    _tree(
        webdav,
        {
            "/": ["a.mp3", "notes.txt", "Call recordings/"],
            "/Call recordings": ["b.amr", "b.done", "2026/"],
            "/Call recordings/2026": ["c.AWB"],
        },
    )
    assert webdav.list_audio_files() == ["a.mp3", "Call recordings/b.amr", "Call recordings/2026/c.AWB"]


def test_listing_error_returns_empty(webdav: WebDAVClient) -> None:
    webdav._client.list.side_effect = RuntimeError("boom")
    assert webdav.list_audio_files() == []


def test_sidecars_live_beside_the_audio_file(webdav: WebDAVClient) -> None:
    webdav.create_done_marker("Call recordings/x.amr")
    webdav.create_quarantine_marker("Call recordings/x.amr", reason="bad")
    remotes = [c[1]["remote_path"] for c in webdav._client.upload_sync.call_args_list]
    assert remotes == ["/Call recordings/x.done", "/Call recordings/x.quarantined"]


def test_done_marker_checked_in_the_files_folder(webdav: WebDAVClient) -> None:
    webdav.done_marker_exists("Call recordings/x.amr")
    webdav._client.check.assert_called_once_with("/Call recordings/x.done")
