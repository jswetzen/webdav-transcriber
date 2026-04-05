# WhisperWebDAV

A containerized daemon that polls a WebDAV share, transcribes audio files using [KBLab KB-Whisper](https://huggingface.co/KBLab) models via `easytranscriber`, uploads the results back to WebDAV, and sends notifications via [Apprise](https://github.com/caronc/apprise).

## Quick Start

```bash
# Copy and edit the environment file
cp .env.example .env

# Run with Docker Compose
docker compose up -d
```

## Configuration

All configuration is done via environment variables (or a `.env` file).

| Variable | Default | Description |
|---|---|---|
| `WEBDAV_URL` | *(required)* | Base URL of the WebDAV server |
| `WEBDAV_USERNAME` | `""` | WebDAV username (use with `WEBDAV_PASSWORD`) |
| `WEBDAV_PASSWORD` | `""` | WebDAV password |
| `WEBDAV_TOKEN` | `""` | Bearer token (alternative to username/password) |
| `WEBDAV_WATCH_PATH` | `"/"` | Path on WebDAV to watch for audio files |
| `POLL_INTERVAL_SECONDS` | `60` | How often to poll the WebDAV share |
| `TRANSCRIPTION_MODEL` | `"KBLab/kb-whisper-large"` | HuggingFace model ID for transcription |
| `EMISSIONS_MODEL` | `"KBLab/kb-wav2vec2-large"` | HuggingFace model ID for emission |
| `VAD_MODEL` | `"silero"` | Voice activity detection: `silero` or `pyannote` |
| `HF_TOKEN` | `""` | HuggingFace token (required for pyannote or gated models) |
| `LANGUAGE` | `"sv"` | BCP-47 language code (see supported languages below) |
| `CACHE_DIR` | `"/app/models"` | Directory to cache downloaded models |
| `GPU_ENABLED` | `false` | Set to `true` to use CUDA GPU |
| `OUTPUT_FORMATS` | `"txt"` | Comma-separated output formats (see below) |
| `OUTPUT_SUBDIR` | `""` | Optional subdirectory on WebDAV for output files |
| `APPRISE_URLS` | `""` | Comma-separated Apprise notification URLs |
| `LOG_LEVEL` | `"INFO"` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FORMAT` | `"plain"` | Log format: `plain` or `json` |

### Supported Languages

`sv` (Swedish), `en` (English), `de` (German), `fr` (French), `fi` (Finnish), `no` (Norwegian), `da` (Danish), `nl` (Dutch), `es` (Spanish), `it` (Italian), `pt` (Portuguese), `pl` (Polish), `ru` (Russian)

## Output Formats

| Format | Extension | Description |
|---|---|---|
| `txt` | `.txt` | Plain text, one segment per line |
| `srt` | `.srt` | SubRip subtitle format with timestamps |
| `json` | `.json` | Raw alignment JSON with word-level timestamps |
| `timestamps` | `.txt` | `[HH:MM:SS] text` per segment |

Set `OUTPUT_FORMATS=txt,srt` for multiple formats. Output files are named `<stem>.<ext>` and uploaded to `WEBDAV_WATCH_PATH` (or `WEBDAV_WATCH_PATH/OUTPUT_SUBDIR` if set).

Files are marked as processed by creating a `<stem>.done` sidecar file on the WebDAV share. If processing fails, no `.done` file is created and the file will be retried on the next poll cycle.

## Docker Compose

```yaml
services:
  whisperwebdav:
    image: ghcr.io/your-org/whisperwebdav:latest
    restart: unless-stopped
    environment:
      WEBDAV_URL: "https://nextcloud.example.com/remote.php/dav/files/user"
      WEBDAV_USERNAME: "user"
      WEBDAV_PASSWORD: "password"
      WEBDAV_WATCH_PATH: "/recordings"
      LANGUAGE: "sv"
      OUTPUT_FORMATS: "txt,srt"
    volumes:
      - model-cache:/app/models

volumes:
  model-cache:
```

## GPU Support

Set `GPU_ENABLED=true` and enable the NVIDIA runtime in Docker Compose:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Notification URLs (Apprise)

Set `APPRISE_URLS` to one or more [Apprise-compatible URLs](https://github.com/caronc/apprise/wiki) separated by commas:

```env
APPRISE_URLS=slack://TokenA/TokenB/TokenC,mailto://user:pass@gmail.com
```

Notifications are sent on transcription success and failure.

## Development

### Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager

### Running Tests

```bash
uv sync --frozen
uv run pytest
```

### Local Build

```bash
docker build -t whisperwebdav:local .
```

### Running Locally

```bash
cp .env.example .env
# Edit .env with your settings
uv run whisperwebdav
```

## Architecture

```
poll loop (watcher.py)
  └─ list WebDAV files (webdav.py)
       └─ for each new file:
            ├─ download audio
            ├─ transcribe (transcriber.py)
            │    └─ easytranscriber pipeline
            ├─ format output (formatter.py)
            ├─ upload results (webdav.py)
            ├─ create .done marker (webdav.py)
            └─ send notification (notifier.py)
```

Config is loaded at startup from environment variables via `pydantic-settings`. Structured logging uses `structlog` with optional JSON output for log aggregation.

## License

Apache 2.0 — see [LICENSE](LICENSE).
