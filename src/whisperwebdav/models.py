"""Startup model check: make sure everything the server needs is on disk, then go offline.

Runs once in server main(), before uvicorn binds. Anything missing is fetched (or converted to
CTranslate2) in a child process, so the download and conversion memory is released afterwards.
Then HF_HUB_OFFLINE=1 is set in this process's environment. Every later model load, and every
child spawned from here, inherits it, so a stalled network can't hang a transcription. A broken
or incomplete cache fails here at startup, not on the first request.

This module must not import huggingface_hub or transformers at top level: they read
HF_HUB_OFFLINE once, when first imported, and the parent must set it before that happens.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import structlog

from .config import Config

log = structlog.get_logger(__name__)

# Files the transcription processor needs (tokenizer + feature extractor), and the ones the
# CTranslate2 converter reads. Named exactly so a download never pulls the ONNX variants.
_TRANSCRIPTION_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "normalizer.json",
)
_TRANSCRIPTION_WEIGHTS = ("model.safetensors",)  # only needed to convert to CTranslate2
_TRANSCRIPTION_REQUIRED = ("preprocessor_config.json", "vocab.json", "merges.txt")

_EMISSIONS_FILES = (
    "config.json",
    "preprocessor_config.json",
    "vocab.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
)
_EMISSIONS_REQUIRED = ("config.json", "preprocessor_config.json", "vocab.json", "model.safetensors")

# Ensures (and converts) an HF repo's files are available without network access.
_CT2_FILES = ("model.bin", "config.json", "vocabulary.json", "preprocessor_config.json")


class ModelsUnavailable(RuntimeError):
    """A required model is missing and could not be fetched or converted. Startup must abort."""


def _hub_dir(cache_dir: str, repo: str) -> Path:
    return Path(cache_dir) / ("models--" + repo.replace("/", "--"))


def _snapshot_has(cache_dir: str, repo: str, filename: str) -> bool:
    return any(p.exists() for p in _hub_dir(cache_dir, repo).glob(f"snapshots/*/{filename}"))


def _ct2_dir(config: Config) -> Path:
    # Same naming as easytranscriber.utils.hf_to_ct2_converter: the repo's last path component.
    return Path(config.cache_dir) / "ct2" / Path(config.transcription_model).stem


def missing_models(config: Config) -> list[str]:
    """Human-readable list of what isn't on disk yet. Empty means the server can start offline."""
    missing: list[str] = []
    ct2 = _ct2_dir(config)
    for name in _CT2_FILES:
        if not (ct2 / name).exists():
            missing.append(f"ct2:{ct2.name}/{name}")
    for name in _TRANSCRIPTION_REQUIRED:
        if not _snapshot_has(config.cache_dir, config.transcription_model, name):
            missing.append(f"hub:{config.transcription_model}/{name}")
    for name in _EMISSIONS_REQUIRED:
        if not _snapshot_has(config.cache_dir, config.emissions_model, name):
            missing.append(f"hub:{config.emissions_model}/{name}")
    return missing


def _network_env() -> dict[str, str]:
    """Environment for a child that is allowed to reach the Hub. HF_TOKEN passes through."""
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    env["HF_HUB_OFFLINE"] = "0"
    return env


def _run_child(args: list[str]) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "whisperwebdav.models", *args], env=_network_env(), check=False
    )
    if proc.returncode != 0:
        raise ModelsUnavailable(f"model step {args[0]!r} failed with exit {proc.returncode}")


def pin_offline() -> None:
    """Set offline mode for this process and every child it spawns from now on."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def ensure_models(config: Config) -> None:
    """Fetch/convert whatever is missing, verify, then pin offline mode. Raises ModelsUnavailable."""
    missing = missing_models(config)
    if missing:
        log.warning("Models missing from cache; fetching before startup", missing=missing)
        ct2_missing = any(m.startswith("ct2:") for m in missing)
        hub_missing = any(m.startswith("hub:") for m in missing)
        if hub_missing or ct2_missing:
            # Conversion reads the HF weights, so fetch them whenever the CT2 copy is absent.
            _run_child([
                "download", config.transcription_model, config.cache_dir,
                ",".join(_TRANSCRIPTION_FILES + (_TRANSCRIPTION_WEIGHTS if ct2_missing else ())),
            ])
            _run_child([
                "download", config.emissions_model, config.cache_dir,
                ",".join(_EMISSIONS_FILES),
            ])
        if ct2_missing:
            _run_child(["convert", config.transcription_model, config.cache_dir])
        still = missing_models(config)
        if still:
            raise ModelsUnavailable(f"still missing after fetch: {still}")
    pin_offline()
    log.info("Models present; running offline", cache_dir=config.cache_dir)


def _child_main(argv: list[str]) -> int:
    """`python -m whisperwebdav.models download REPO CACHE FILES` / `convert MODEL CACHE`."""
    if argv[0] == "download":
        from huggingface_hub import snapshot_download  # type: ignore[import]

        _, repo, cache_dir, files = argv
        snapshot_download(
            repo_id=repo,
            cache_dir=cache_dir,
            allow_patterns=files.split(","),
            token=os.environ.get("HF_TOKEN") or None,
        )
        return 0
    if argv[0] == "convert":
        from easytranscriber.utils import hf_to_ct2_converter  # type: ignore[import]

        _, model, cache_dir = argv
        hf_to_ct2_converter(model, cache_dir=cache_dir)
        return 0
    raise SystemExit(f"unknown model step: {argv[:1]}")


if __name__ == "__main__":
    sys.exit(_child_main(sys.argv[1:]))
