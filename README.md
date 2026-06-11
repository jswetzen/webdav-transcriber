# WhisperWebDAV

Transcribes audio with [KBLab KB-Whisper](https://huggingface.co/KBLab) models via `easytranscriber`. It runs in two complementary modes from one image:

- **`whisperwebdav-server`** — an OpenAI-compatible HTTP server (`POST /v1/audio/transcriptions`). It owns the model and is the single gate on the GPU (requests serialize through one in-process lock). Point any OpenAI-speaking client at it. This is the image's default command.
- **`whisperwebdav`** — the WebDAV poll loop: watches a share, transcribes new audio, uploads results, and notifies via [Apprise](https://github.com/caronc/apprise). With `TRANSCRIBE_BACKEND=http` it becomes a thin client that offloads transcription to a server instance (so it needs no GPU); with the default `local` it transcribes in-process for standalone use.

Co-locating the two as separate processes/containers means **one model load** shared by both the poll loop and any HTTP caller — see [`docker-compose.yaml`](docker-compose.yaml).

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
| `TRANSCRIBE_BACKEND` | `"local"` | Poll loop only: `local` transcribes in-process, `http` offloads to a server |
| `TRANSCRIBE_SERVER_URL` | `""` | Required when `TRANSCRIBE_BACKEND=http` (e.g. `http://whisper-server:8000`) |
| `API_KEY` | `""` | Optional bearer key. Server requires it on requests when set; http client sends it |
| `SERVER_HOST` | `"0.0.0.0"` | Server bind host (`whisperwebdav-server` only) |
| `SERVER_PORT` | `8000` | Server bind port (`whisperwebdav-server` only) |
| `WEBDAV_URL` | *(required for poll loop)* | Base URL of the WebDAV server (unused by the server) |
| `WEBDAV_USERNAME` | `""` | WebDAV username (use with `WEBDAV_PASSWORD`) |
| `WEBDAV_PASSWORD` | `""` | WebDAV password |
| `WEBDAV_TOKEN` | `""` | Bearer token (alternative to username/password) |
| `WEBDAV_WATCH_PATH` | `"/"` | Path on WebDAV to watch for audio files |
| `POLL_INTERVAL_SECONDS` | `60` | How often to poll the WebDAV share |
| `MAX_BATCH_SIZE` | `8` | Max files per `easytranscriber` pipeline call (see Batching below) |
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
| `vtt` | `.vtt` | WebVTT subtitle format with timestamps |
| `json` | `.json` | Raw alignment JSON with word-level timestamps |
| `timestamps` | `.txt` | `[HH:MM:SS] text` per segment |

Set `OUTPUT_FORMATS=txt,srt` for multiple formats. Output files are named `<stem>.<ext>` and uploaded to `WEBDAV_WATCH_PATH` (or `WEBDAV_WATCH_PATH/OUTPUT_SUBDIR` if set).

Files are marked as processed by creating a `<stem>.done` sidecar file on the WebDAV share. If processing fails, no `.done` file is created and the file will be retried on the next poll cycle.

## OpenAI-compatible server

`whisperwebdav-server` exposes the [OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio/createTranscription) shape:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \   # only if API_KEY is set
  -F file=@note.m4a \
  -F response_format=srt
```

Endpoints: `POST /v1/audio/transcriptions`, `GET /v1/models`, `GET /healthz`.

`response_format` selects the rendering:

| Value | Returns |
|---|---|
| `json` *(default)* | `{"text": "..."}` |
| `text` | plain transcript text |
| `verbose_json` | `{task, language, duration, text, segments[]}` with per-segment timestamps |
| `srt` / `vtt` | subtitle text with timestamps |

The form fields `model` and `temperature` are accepted for client compatibility; `language` overrides the server's configured language per request.

## Docker Compose

See [`docker-compose.yaml`](docker-compose.yaml) for a two-service setup: `whisper-server` (the model owner / OpenAI endpoint) and an optional `whisper-poller` (WebDAV poll loop in `http` mode). Run only the server if you just want the endpoint.

## GPU Support

Set `GPU_ENABLED=true` and pass the GPU through. Whichever path you take, also
bump the container's shared-memory size — PyTorch DataLoader workers serialize
tensors via `/dev/shm` and the 64 MB container default trips
`RuntimeError: unable to allocate shared memory(shm)` partway through transcription:

```yaml
shm_size: "2gb"
```

### Path A: Docker + NVIDIA Container Toolkit

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Path B: Podman / bare LXC (no nvidia-container-toolkit)

When the NVIDIA Container Toolkit isn't available (e.g. inside a Proxmox LXC),
pass the device nodes directly and bind-mount the host's userspace driver
libraries. The bind-mounted libs must match the host driver version exactly,
so the host needs `libcuda.so.1` / `libnvidia-ml.so.1` /
`libnvidia-ptxjitcompiler.so.1` SONAME symlinks pointing at the versioned files.

```yaml
devices:
  - /dev/nvidia0
  - /dev/nvidiactl
  - /dev/nvidia-uvm
  # Optional — only needed for profiling tools / display modesetting:
  # - /dev/nvidia-uvm-tools
  # - /dev/nvidia-modeset
volumes:
  - /usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro
  - /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro
  - /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro
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
uv run whisperwebdav-server   # OpenAI endpoint on :8000
uv run whisperwebdav          # WebDAV poll loop
```

## Architecture

```
poll loop (watcher.py)
  └─ list WebDAV files (webdav.py)
       └─ chunk new files into batches of MAX_BATCH_SIZE
            └─ for each batch:
                 ├─ download audio (per file)
                 ├─ transcribe_batch (transcriber.py)
                 │    └─ single easytranscriber pipeline call
                 └─ for each transcribed file:
                      ├─ format output (formatter.py)
                      ├─ upload results (webdav.py)
                      ├─ create .done marker (webdav.py)
                      └─ send notification (notifier.py)
```

Config is loaded at startup from environment variables via `pydantic-settings`. Structured logging uses `structlog` with optional JSON output for log aggregation.

### Batching

When a poll cycle finds multiple unprocessed files, they are transcribed together in a single `easytranscriber` pipeline call (up to `MAX_BATCH_SIZE` per call). This amortizes model warm-up and lets `easytranscriber` parallel-prefetch audio across the batch.

Failures are isolated per file: a download or upload error for one file does not block the rest of the batch. If the transcription call itself fails, every file in that batch is reported as failed and retried on the next poll (no `.done` marker is written).

## License

Apache 2.0 — see [LICENSE](LICENSE).
