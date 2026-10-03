from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus", ".aac"}
)

LANGUAGE_TOKENIZER_MAP: dict[str, str] = {
    "sv": "swedish",
    "en": "english",
    "de": "german",
    "fr": "french",
    "fi": "finnish",
    "no": "norwegian",
    "da": "danish",
    "nl": "dutch",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "pl": "polish",
    "ru": "russian",
}


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # WebDAV connection. Optional: the server process (whisperwebdav-server) needs no WebDAV
    # config at all, and the poll loop only requires it when TRANSCRIBE_BACKEND polls a share.
    # Auth is enforced in validate_auth only when webdav_url is set.
    webdav_url: str = ""
    webdav_username: str = ""
    webdav_password: str = ""
    webdav_token: str = ""
    webdav_watch_path: str = "/"

    # Transcription backend: "local" runs the model in-process (standalone use); "http" turns
    # the poll loop into a thin client that POSTs files to a whisperwebdav-server instance
    # (the model owner), so the poller needs no torch/GPU. See server.py / client.py.
    transcribe_backend: Literal["local", "http"] = "local"
    transcribe_server_url: str = ""  # required when transcribe_backend == "http"
    # Optional bearer key. The server requires it on requests when set; the http client sends it.
    api_key: str = ""

    # OpenAI-compatible server bind (whisperwebdav-server only).
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    # Upper bound in bytes on a POST /v1/audio/transcriptions request body (enforced before and
    # while the upload is buffered, see server.BodyLimitMiddleware). 2 GiB by default because the
    # poll loop in http mode uploads whole WAV/FLAC files (~11 MiB/min for 48 kHz stereo WAV): a
    # lower default would 413 long recordings, which the poller then quarantines. Operators
    # exposing the server publicly may want a smaller value.
    max_upload_bytes: int = Field(default=2 * 1024**3, gt=0)

    # Polling
    poll_interval_seconds: int = 60
    max_batch_size: int = 8
    # After this many consecutive failures a file is quarantined (a .quarantined sidecar is
    # written and it's never retried again) instead of re-queued forever. Was unbounded before
    # 2026-09-12: four permanently-failing files each re-queued ~1000x/day, 70% of the fleet's
    # journal volume for this CT, until this incident.
    max_retries: int = 5

    # Transcription models
    transcription_model: str = "KBLab/kb-whisper-large"
    emissions_model: str = "KBLab/wav2vec2-large-voxrex-swedish"
    vad_model: Literal["silero", "pyannote"] = "silero"
    hf_token: str = ""
    language: str = "sv"
    cache_dir: str = "/app/models"
    gpu_enabled: bool = False
    gpu_idle_release_seconds: int = 120

    # Text-to-speech (Kokoro), see tts.py. Model files aren't published on an index kokoro-onnx
    # can fetch itself — download kokoro-v1.0.onnx / voices-v1.0.bin from
    # https://github.com/thewh1teagle/kokoro-onnx/releases and place them at these paths
    # (cache_dir by default, so they share the model-cache volume with Whisper).
    kokoro_model_path: str = "/app/models/kokoro-v1.0.onnx"
    kokoro_voices_path: str = "/app/models/voices-v1.0.bin"
    tts_default_voice: str = "af_sarah"
    tts_lang: str = "en-us"
    tts_max_input_chars: int = 5000
    # Seconds without a TTS request before the Kokoro ONNX session is dropped to hand its VRAM
    # back to co-located consumers (see tts.py). 0 keeps it resident for the process lifetime.
    # Separate from gpu_idle_release_seconds on purpose: that one is only read by watcher.py's
    # poll loop (a thin HTTP client with no model of its own) and drives a torch cache flush,
    # which does nothing for onnxruntime's CUDA arena anyway.
    tts_idle_release_seconds: int = 300

    # Output
    output_formats: str = "txt"
    output_subdir: str = ""

    # Notifications
    apprise_urls: str = ""

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "plain"] = "plain"

    @field_validator("max_batch_size")
    @classmethod
    def validate_max_batch_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_batch_size must be >= 1")
        return v

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_retries must be >= 1")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if v not in LANGUAGE_TOKENIZER_MAP:
            supported = ", ".join(sorted(LANGUAGE_TOKENIZER_MAP.keys()))
            raise ValueError(
                f"Unsupported language '{v}'. Supported languages: {supported}"
            )
        return v

    @model_validator(mode="after")
    def validate_auth(self) -> "Config":
        # Only the poll loop talks to WebDAV; the server leaves webdav_url empty. Enforce auth
        # only once a share is configured.
        if self.webdav_url:
            has_userpass = bool(self.webdav_username and self.webdav_password)
            has_token = bool(self.webdav_token)
            if not has_userpass and not has_token:
                raise ValueError(
                    "Either webdav_username+webdav_password or webdav_token must be provided"
                )
        return self

    @model_validator(mode="after")
    def validate_backend(self) -> "Config":
        if self.transcribe_backend == "http" and not self.transcribe_server_url:
            raise ValueError(
                "transcribe_server_url is required when transcribe_backend is 'http'"
            )
        return self

    @property
    def output_formats_list(self) -> list[str]:
        return [fmt.strip() for fmt in self.output_formats.split(",") if fmt.strip()]

    @property
    def apprise_urls_list(self) -> list[str]:
        return [url.strip() for url in self.apprise_urls.split(",") if url.strip()]

    @property
    def tokenizer_name(self) -> str:
        return LANGUAGE_TOKENIZER_MAP[self.language]

    @property
    def device(self) -> str:
        return "cuda" if self.gpu_enabled else "cpu"
