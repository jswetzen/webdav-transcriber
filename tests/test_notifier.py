from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from whisperwebdav.config import Config
from whisperwebdav.notifier import Notifier


@pytest.fixture
def config_no_urls(minimal_config: Config) -> Config:
    return minimal_config


@pytest.fixture
def config_with_urls(monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("WEBDAV_URL", "http://webdav.example.com")
    monkeypatch.setenv("WEBDAV_USERNAME", "user")
    monkeypatch.setenv("WEBDAV_PASSWORD", "pass")
    monkeypatch.setenv("APPRISE_URLS", "mailto://a:b@c.com,json://example.com")
    return Config()


class TestNotifier:
    def test_disabled_when_no_urls(self, config_no_urls: Config) -> None:
        n = Notifier(config_no_urls)
        assert n._enabled is False

    def test_notify_noop_when_disabled(self, config_no_urls: Config) -> None:
        n = Notifier(config_no_urls)
        n._apprise = MagicMock()
        n.notify_success("f.mp3", ["txt"])
        n.notify_failure("f.mp3", RuntimeError("x"))
        n._apprise.notify.assert_not_called()

    def test_enabled_when_urls_added(self, config_with_urls: Config) -> None:
        with patch("apprise.Apprise") as ap_cls:
            instance = ap_cls.return_value
            instance.add.return_value = True
            instance.notify.return_value = True
            n = Notifier(config_with_urls)
        assert n._enabled is True
        assert instance.add.call_count == 2

    def test_disabled_when_all_urls_invalid(
        self, config_with_urls: Config, caplog: pytest.LogCaptureFixture
    ) -> None:
        with patch("apprise.Apprise") as ap_cls:
            instance = ap_cls.return_value
            instance.add.return_value = False
            n = Notifier(config_with_urls)
        assert n._enabled is False

    def test_partial_invalid_urls_still_enabled(
        self, config_with_urls: Config
    ) -> None:
        with patch("apprise.Apprise") as ap_cls:
            instance = ap_cls.return_value
            instance.add.side_effect = [True, False]
            n = Notifier(config_with_urls)
        assert n._enabled is True

    def test_notify_success_calls_apprise(self, config_with_urls: Config) -> None:
        with patch("apprise.Apprise") as ap_cls:
            instance = ap_cls.return_value
            instance.add.return_value = True
            instance.notify.return_value = True
            n = Notifier(config_with_urls)
            n.notify_success("foo.mp3", ["txt", "srt"])
        instance.notify.assert_called_once()
        kwargs = instance.notify.call_args.kwargs
        assert "foo.mp3" in kwargs["body"]
        assert "txt, srt" in kwargs["body"]
        assert kwargs["title"] == "Transcription complete"

    def test_notify_failure_calls_apprise(self, config_with_urls: Config) -> None:
        with patch("apprise.Apprise") as ap_cls:
            instance = ap_cls.return_value
            instance.add.return_value = True
            instance.notify.return_value = True
            n = Notifier(config_with_urls)
            n.notify_failure("bad.mp3", RuntimeError("boom"))
        kwargs = instance.notify.call_args.kwargs
        assert "bad.mp3" in kwargs["body"]
        assert "boom" in kwargs["body"]
        assert kwargs["title"] == "Transcription failed"

    def test_notify_logs_when_apprise_returns_false(
        self, config_with_urls: Config
    ) -> None:
        with patch("apprise.Apprise") as ap_cls:
            instance = ap_cls.return_value
            instance.add.return_value = True
            instance.notify.return_value = False
            n = Notifier(config_with_urls)
            with patch("whisperwebdav.notifier.log") as mock_log:
                n.notify_success("f.mp3", ["txt"])
                mock_log.error.assert_called_once()
