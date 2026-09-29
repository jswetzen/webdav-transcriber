# WhisperWebDAV

Transcribes audio with [KBLab KB-Whisper](https://huggingface.co/KBLab) models via `easytranscriber`. It runs in two complementary modes from one image:

- **`whisperwebdav-server`** — an OpenAI-compatible HTTP server: `POST /v1/audio/transcriptions` (KB-Whisper) and `POST /v1/audio/speech` (Kokoro TTS, see [below](#text-to-speech-kokoro)). It owns both models and is the single gate on the GPU (requests serialize through one in-process lock, shared by both). Point any OpenAI-speaking client at it. This is the image's default command.
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
| `KOKORO_MODEL_PATH` | `"/app/models/kokoro-v1.0.onnx"` | Path to the Kokoro ONNX model (see [Text-to-speech](#text-to-speech-kokoro)) |
| `KOKORO_VOICES_PATH` | `"/app/models/voices-v1.0.bin"` | Path to the Kokoro voice bank |
| `TTS_DEFAULT_VOICE` | `"af_sarah"` | Voice used when a request omits `voice` |
| `TTS_LANG` | `"en-us"` | Kokoro phonemizer language (not exposed via the API — OpenAI's TTS request has no `lang` field) |
| `TTS_MAX_INPUT_CHARS` | `5000` | Reject `/v1/audio/speech` requests with longer `input` |
| `TTS_IDLE_RELEASE_SECONDS` | `300` | Drop the Kokoro ONNX session (and its VRAM) after this long without a TTS request; the next request reloads it. `0` keeps it resident |

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

Endpoints: `POST /v1/audio/transcriptions`, `POST /v1/audio/speech`, `GET /v1/models`, `GET /healthz`.

`response_format` selects the rendering:

| Value | Returns |
|---|---|
| `json` *(default)* | `{"text": "..."}` |
| `text` | plain transcript text |
| `verbose_json` | `{task, language, duration, text, segments[]}` with per-segment timestamps |
| `srt` / `vtt` | subtitle text with timestamps |

The form fields `model` and `temperature` are accepted for client compatibility; `language` overrides the server's configured language per request.

## Text-to-speech (Kokoro)

`whisperwebdav-server` also exposes the [OpenAI TTS API](https://platform.openai.com/docs/api-reference/audio/createSpeech) shape, backed by [`kokoro-onnx`](https://github.com/thewh1teagle/kokoro-onnx):

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer $API_KEY" \   # only if API_KEY is set
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from Kokoro", "voice": "af_sarah", "response_format": "wav"}' \
  -o speech.wav
```

Unlike `/v1/audio/transcriptions`, this endpoint takes a JSON body (`model`, `input`, `voice`, `response_format`), matching OpenAI's own request shape. `response_format` is `wav` (default) or `mp3` (transcoded through the `ffmpeg` already bundled for Whisper's input demuxing). `GET /v1/models` lists available voices under a `voices` key once the model files below are present, and warms the model on first call.

Streaming is supported too: send `"stream": true` with `response_format` `mp3` or `pcm`, and audio is sent as soon as kokoro-onnx's `create_stream()` produces each sentence or clause, instead of after the whole utterance. `pcm` is raw mono s16le at 24 kHz with no container. It *always* streams, whether `stream` is set or not, because that is how OpenAI-compatible streaming clients (e.g. the customtts Firefox extension) request low-latency playback. `wav` can't be streamed: a valid WAV header needs the total length up front, so `stream: true` with `wav` returns `400`.

### Model files

kokoro-onnx doesn't fetch its own weights from a package index — download the two release assets once and place them where `KOKORO_MODEL_PATH` / `KOKORO_VOICES_PATH` point (the model-cache volume by default, alongside the Whisper cache):

```bash
wget -P ./models https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx
wget -P ./models https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin
```

Until both files exist, `/v1/audio/speech` returns `400` and `/v1/models` simply omits the `kokoro` entry — the rest of the server (transcription) is unaffected. Voice names come from the voice bank itself (`kokoro.get_voices()`); see [Kokoro-82M/VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) for what's in the v1.0 release.

GPU inference is automatic when `GPU_ENABLED=true`: ONNX Runtime picks up `CUDAExecutionProvider` on its own once `onnxruntime-gpu` is installed (the Docker image does this for x86_64 builds, reusing the CUDA 12 / cuDNN 9 libraries already staged for Whisper — see the Dockerfile). No separate synthesis queue exists: Kokoro synthesis and Whisper transcription share the same process-global GPU lock used for transcription, so the two never run concurrently on one card.

**Not implemented (deliberately out of scope for this pass):** response caching, Prometheus metrics (no metrics infra exists in this service yet), and a generic per-request timeout middleware. `TTS_MAX_INPUT_CHARS` is the only guard against oversized/slow requests today.

For a survey of alternative TTS models (Qwen3-TTS, Qwen-Audio 3.1, Chatterbox, Piper, …) and why Kokoro stays for now, see [docs/tts-alternatives.md](docs/tts-alternatives.md).

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
