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


def test_skips_self_entry_echoed_by_owncloud(webdav: WebDAVClient) -> None:
    # ownCloud public links list the directory itself as a child ("webdav/" for the root,
    # the folder's own name inside a subfolder); recursing into it would 404.
    webdav._root_name = "webdav"
    _tree(
        webdav,
        {
            "/": ["webdav/", "a.mp3", "Call recordings/"],
            "/Call recordings": ["Call recordings/", "b.amr"],
        },
    )
    assert webdav.list_audio_files() == ["a.mp3", "Call recordings/b.amr"]


def test_unreadable_subfolder_does_not_hide_the_rest(webdav: WebDAVClient) -> None:
    def ls(path: str) -> list[str]:
        if path == "/bad":
            raise RuntimeError("403")
        return {"/": ["bad/", "a.mp3"]}[path]

    webdav._client.list.side_effect = ls
    assert webdav.list_audio_files() == ["a.mp3"]


def test_skips_hidden_folders(webdav: WebDAVClient) -> None:
    # .trash/ holds notes the user deleted; transcribing them would resurrect them.
    _tree(webdav, {"/": [".trash/", "a.mp3"], "/.trash": ["gone.m4a"]})
    assert webdav.list_audio_files() == ["a.mp3"]
