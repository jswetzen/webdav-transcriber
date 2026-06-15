from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from whisperwebdav import healthcheck


def _set_share_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBDAV_URL", "https://owncloud.example.com/public.php/webdav")
    monkeypatch.setenv("WEBDAV_USERNAME", "share-token")
    monkeypatch.setenv("WEBDAV_PASSWORD", "x")


class TestHealthcheck:
    def test_healthy_share_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_share_env(monkeypatch)
        client = MagicMock()
        with patch("whisperwebdav.healthcheck.WebDAVClient", return_value=client):
            with pytest.raises(SystemExit) as exc:
                healthcheck.main()
        assert exc.value.code == 0
        client.check_connection.assert_called_once()

    def test_unauthorized_share_exits_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_share_env(monkeypatch)
        client = MagicMock()
        client.check_connection.side_effect = RuntimeError("401 Unauthorized")
        with patch("whisperwebdav.healthcheck.WebDAVClient", return_value=client):
            with pytest.raises(SystemExit) as exc:
                healthcheck.main()
        assert exc.value.code == 1

    def test_no_share_configured_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Server-only deployment: no WEBDAV_URL -> nothing to be unhealthy about.
        monkeypatch.delenv("WEBDAV_URL", raising=False)
        with patch("whisperwebdav.healthcheck.WebDAVClient") as client_cls:
            with pytest.raises(SystemExit) as exc:
                healthcheck.main()
        assert exc.value.code == 0
        client_cls.assert_not_called()
