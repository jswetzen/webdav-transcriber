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


# Stage 2: final runtime image
FROM python:3.14-slim

RUN groupadd --system appuser && useradd --system --gid appuser appuser

COPY --from=builder /app/.venv /app/.venv

RUN mkdir /app/models && chown appuser:appuser /app/models

ENV PATH="/app/.venv/bin:$PATH"

USER appuser

ENTRYPOINT ["whisperwebdav"]
