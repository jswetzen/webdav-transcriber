from __future__ import annotations

from typing import Literal

from pydantic import field_validator, model_validator
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

    # WebDAV connection
    webdav_url: str
    webdav_username: str = ""
    webdav_password: str = ""
    webdav_token: str = ""
    webdav_watch_path: str = "/"

    # Polling
    poll_interval_seconds: int = 60

    # Transcription models
    transcription_model: str = "KBLab/kb-whisper-large"
    emissions_model: str = "KBLab/kb-wav2vec2-large"
    vad_model: Literal["silero", "pyannote"] = "silero"
    hf_token: str = ""
    language: str = "sv"
    cache_dir: str = "/app/models"
    gpu_enabled: bool = False

    # Output
    output_formats: str = "txt"
    output_subdir: str = ""

    # Notifications
    apprise_urls: str = ""

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "plain"] = "plain"

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
        has_userpass = bool(self.webdav_username and self.webdav_password)
        has_token = bool(self.webdav_token)
        if not has_userpass and not has_token:
            raise ValueError(
                "Either webdav_username+webdav_password or webdav_token must be provided"
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
