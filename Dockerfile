# Stage 1: builder
FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.30 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Two-pass sync for better layer caching:
# Pass 1: install dependencies only (cached unless pyproject.toml/uv.lock changes)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Pass 2: install the project itself
COPY README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Extra CUDA libs the wheels don't otherwise pull (x86_64 only; arm64 is CPU-only):
#   - CTranslate2 dlopens libcublas.so.12 / libcudnn.so.9 (CUDA 12).
#   - torchcodec (torchaudio's decode backend) NEEDs CUDA-13 NPP (libnppicc.so.13 et al.),
#     which is not a torch dependency, so it must be installed explicitly.
# libcuda.so.1 is still injected at runtime (NVIDIA Container Toolkit or bind-mount).
# NOTE: discovery (ldconfig) is done in the FINAL stage — an ld.so.conf written here would be
# dropped by the `COPY --from=builder /app/.venv` and never reach the runtime image.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
    uv pip install --python /app/.venv/bin/python --no-deps \
        nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-npp-cu13; fi


# Stage 2: final runtime image
FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system appuser && useradd --system --gid appuser appuser
RUN mkdir -p /home/appuser && chown appuser:appuser /home/appuser

COPY --from=builder /app/.venv /app/.venv

# Put every CUDA lib that ships inside the copied venv onto the system loader path. torch loads
# its own libs via RUNPATH, but standalone extensions resolve NEEDED libs through the system
# search path — without this, torchcodec can't find libnppicc.so.13 and CTranslate2 can't find
# libcublas.so.12 (the builder-stage ld.so.conf does not survive the COPY above). One ldconfig
# covers cublas/cudnn (CTranslate2), npp (torchcodec), and torch's cu13 runtime libs.
RUN find /app/.venv -type d -path '*/nvidia/*/lib' > /etc/ld.so.conf.d/nvidia-venv.conf && ldconfig

RUN mkdir /app/models && chown appuser:appuser /app/models

ENV PATH="/app/.venv/bin:$PATH"
ENV HOME=/home/appuser

USER appuser

EXPOSE 8000

# Default to the model-owner OpenAI server. Run the WebDAV poll loop instead by overriding
# the command to `whisperwebdav` (see docker-compose.yaml for both services).
CMD ["whisperwebdav-server"]
