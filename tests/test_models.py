"""Startup model check: detect missing files, fetch/convert them in children, then pin offline."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from whisperwebdav import models
from whisperwebdav.config import Config
from whisperwebdav.models import ModelsUnavailable, ensure_models, missing_models


@pytest.fixture(autouse=True)
def clean_offline_env(monkeypatch: pytest.MonkeyPatch):
    # pin_offline() mutates os.environ for real, so clear it before and after every test.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    yield
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(key, None)


def _cfg(cache: Path) -> Config:
    return Config(cache_dir=str(cache))


def _hub_file(cache: Path, repo: str, name: str) -> None:
    snap = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc123"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / name).write_text("x")


def _ct2(cache: Path, cfg: Config) -> None:
    d = Path(cache) / "ct2" / Path(cfg.transcription_model).stem
    d.mkdir(parents=True, exist_ok=True)
    for name in ("model.bin", "config.json", "vocabulary.json", "preprocessor_config.json"):
        (d / name).write_text("x")


def _populate_everything(cache: Path, cfg: Config) -> None:
    for name in models._TRANSCRIPTION_REQUIRED:
        _hub_file(cache, cfg.transcription_model, name)
    for name in models._EMISSIONS_REQUIRED:
        _hub_file(cache, cfg.emissions_model, name)
    _ct2(cache, cfg)


def test_empty_cache_reports_everything_missing(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    missing = missing_models(cfg)
    assert any(m.startswith("ct2:") for m in missing)
    assert any(cfg.emissions_model in m for m in missing)
    assert any(cfg.transcription_model in m for m in missing)


def test_complete_cache_reports_nothing_missing(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _populate_everything(tmp_path, cfg)
    assert missing_models(cfg) == []


def test_truncated_ct2_directory_counts_as_missing(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _populate_everything(tmp_path, cfg)
    (tmp_path / "ct2" / "kb-whisper-large" / "model.bin").unlink()
    assert missing_models(cfg) == [f"ct2:kb-whisper-large/model.bin"]


def test_complete_cache_goes_offline_without_any_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    _populate_everything(tmp_path, cfg)

    def no_children(*a, **k):
        raise AssertionError("no child should run when the cache is complete")

    monkeypatch.setattr(models.subprocess, "run", no_children)
    ensure_models(cfg)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_missing_models_are_fetched_then_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    calls: list[list[str]] = []

    def fake_child(args: list[str]) -> None:
        calls.append(args)
        if args[0] == "download":
            for name in args[3].split(","):
                _hub_file(tmp_path, args[1], name)
        elif args[0] == "convert":
            _ct2(tmp_path, cfg)

    monkeypatch.setattr(models, "_run_child", fake_child)
    ensure_models(cfg)

    assert [c[0] for c in calls] == ["download", "download", "convert"]
    # Conversion reads the HF weights, so the transcription download must include them.
    assert "model.safetensors" in calls[0][3].split(",")
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_fetch_that_leaves_gaps_refuses_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(models, "_run_child", lambda args: None)
    with pytest.raises(ModelsUnavailable, match="still missing"):
        ensure_models(cfg)
    assert "HF_HUB_OFFLINE" not in os.environ


def test_failed_child_is_surfaced_as_models_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(
        models.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1)
    )
    with pytest.raises(ModelsUnavailable, match="exit 1"):
        ensure_models(cfg)


def test_child_environment_reenables_network_only_for_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    env = models._network_env()
    assert env["HF_HUB_OFFLINE"] == "0"
    assert "TRANSFORMERS_OFFLINE" not in env
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_partial_ct2_directory_is_cleared_before_reconverting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A killed conversion leaves a half-written dir; the converter would reuse it as-is.
    cfg = _cfg(tmp_path)
    _populate_everything(tmp_path, cfg)
    ct2 = tmp_path / "ct2" / "kb-whisper-large"
    (ct2 / "model.bin").unlink()
    seen: list[bool] = []

    def fake_child(args: list[str]) -> None:
        if args[0] == "convert":
            seen.append(ct2.exists())
            _ct2(tmp_path, cfg)

    monkeypatch.setattr(models, "_run_child", fake_child)
    ensure_models(cfg)
    assert seen == [False]


def test_empty_transcription_model_is_refused_before_touching_the_cache(tmp_path: Path) -> None:
    cfg = Config(cache_dir=str(tmp_path), transcription_model="")
    with pytest.raises(ModelsUnavailable, match="empty"):
        ensure_models(cfg)
    assert not (tmp_path / "ct2").exists()
