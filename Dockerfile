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

# CTranslate2 dlopens libcublas.so.12/libcudnn.so.9 on GPU. Wheels provide
# these on x86_64 only; arm64 is CPU-only. libcuda.so.1 is injected at
# runtime by NVIDIA Container Toolkit (--gpus all) or bind-mounted manually.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
    uv pip install --python /app/.venv/bin/python --no-deps \
        nvidia-cublas-cu12 nvidia-cudnn-cu12 && \
    echo "/app/.venv/lib/python3.14/site-packages/nvidia/cublas/lib"  > /etc/ld.so.conf.d/cuda12-extras.conf && \
    echo "/app/.venv/lib/python3.14/site-packages/nvidia/cudnn/lib" >> /etc/ld.so.conf.d/cuda12-extras.conf && \
    ldconfig; fi


# Stage 2: final runtime image
FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system appuser && useradd --system --gid appuser appuser
RUN mkdir -p /home/appuser && chown appuser:appuser /home/appuser

COPY --from=builder /app/.venv /app/.venv

RUN mkdir /app/models && chown appuser:appuser /app/models

ENV PATH="/app/.venv/bin:$PATH"
ENV HOME=/home/appuser

USER appuser

EXPOSE 8000

# Default to the model-owner OpenAI server. Run the WebDAV poll loop instead by overriding
# the command to `whisperwebdav` (see docker-compose.yaml for both services).
CMD ["whisperwebdav-server"]
