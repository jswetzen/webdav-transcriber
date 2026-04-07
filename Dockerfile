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

# Apply patches for upstream library bugs
COPY patches/ /tmp/patches/
RUN SITE=$(python -c "import site; print(site.getsitepackages()[0])") && \
    cp /tmp/patches/easyaligner_collators.py      $SITE/easyaligner/data/collators.py && \
    cp /tmp/patches/easyaligner_pipelines.py      $SITE/easyaligner/pipelines.py && \
    cp /tmp/patches/easytranscriber_pipelines.py  $SITE/easytranscriber/pipelines.py && \
    cp /tmp/patches/easytranscriber_ct2.py        $SITE/easytranscriber/asr/ct2.py && \
    find $SITE/easyaligner $SITE/easytranscriber -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true


# Stage 2: final runtime image
FROM python:3.14-slim

RUN groupadd --system appuser && useradd --system --gid appuser appuser
RUN mkdir -p /home/appuser && chown appuser:appuser /home/appuser

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv

RUN mkdir /app/models && chown appuser:appuser /app/models

ENV PATH="/app/.venv/bin:$PATH"
ENV HOME=/home/appuser

USER appuser

ENTRYPOINT ["whisperwebdav"]
