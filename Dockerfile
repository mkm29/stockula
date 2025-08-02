# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for Stockula trading platform
#
# Build stages:
#   - base: System dependencies and build tools
#   - dependencies: Python packages installation
#   - source: Application source code
#   - production: Minimal runtime image
#   - cli: Production + interactive tools
#
# Usage:
#   docker buildx build --target production -t stockula:prod .
#   docker buildx build --target cli -t stockula:cli .

# Stage 1: Base image with uv and system dependencies
FROM ghcr.io/astral-sh/uv:0.5.21-python3.13-bookworm-slim AS base

# Build arguments for labels
ARG VERSION="0.0.0"
ARG BUILD_DATE
ARG GIT_COMMIT
ARG GIT_URL

# Use bash with pipefail option for better error handling
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

# Install system dependencies with cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies builder (cache Python packages)
FROM base AS dependencies

WORKDIR /app

# Copy dependency files and README (required by pyproject.toml)
COPY pyproject.toml uv.lock README.md ./

# Create virtual environment and install dependencies with cache mount
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv venv /opt/venv --python 3.13 && \
    . /opt/venv/bin/activate && \
    uv sync --frozen --no-dev --compile-bytecode

# Stage 3: Source builder
FROM dependencies AS source

# Copy source code
COPY README.md ./
COPY src/ src/
COPY alembic.ini ./
COPY alembic/ alembic/
COPY examples/ examples/

# Install the package
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    . /opt/venv/bin/activate && \
    uv pip install --no-deps -e .

# Stage 4: Production runtime - minimal image
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS production

# Re-declare build arguments for this stage
ARG VERSION=dev
ARG BUILD_DATE
ARG GIT_COMMIT
ARG GIT_URL

# OCI Standard Labels (https://github.com/opencontainers/image-spec/blob/main/annotations.md)
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.authors="Stockula Contributors <https://github.com/mkm29/stockula>" \
      org.opencontainers.image.url="https://github.com/mkm29/stockula" \
      org.opencontainers.image.documentation="https://github.com/mkm29/stockula/tree/main/docs" \
      org.opencontainers.image.source="${GIT_URL}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.vendor="Stockula Project" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.ref.name="${VERSION}" \
      org.opencontainers.image.title="Stockula" \
      org.opencontainers.image.description="A comprehensive Python trading platform for technical analysis, backtesting, and forecasting"

# Custom Labels for additional metadata
LABEL com.stockula.python.version="3.13" \
      com.stockula.base.image="ghcr.io/astral-sh/uv:python3.13-bookworm-slim" \
      com.stockula.build.stage="production" \
      com.stockula.maintainer="mkm29"

# Install minimal runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
# Using specific UID/GID for consistency across environments
RUN groupadd --gid 1000 stockula && \
    useradd --uid 1000 --gid stockula --shell /bin/bash --create-home stockula

WORKDIR /app

# Copy virtual environment and source from builder
COPY --from=source --chown=stockula:stockula /opt/venv /opt/venv
COPY --from=source --chown=stockula:stockula /app /app

# Set up environment
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    STOCKULA_VERSION="${VERSION}"

# Create data directories
RUN mkdir -p /app/data /app/results && \
    chown -R stockula:stockula /app

USER stockula

VOLUME ["/app/data", "/app/results"]

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import stockula; print('Stockula is healthy')" || exit 1

# Expose metadata about the image
EXPOSE 8888/tcp

# Default entrypoint and command
ENTRYPOINT ["python", "-m"]
CMD ["stockula.main", "--help"]

# Stage 5: CLI stage - optimized for command-line usage
FROM production AS cli

# Add stage-specific label
LABEL com.stockula.build.stage="cli" \
      org.opencontainers.image.description="Stockula CLI - Interactive command-line interface for trading analysis"

USER root
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    less \
    nano \
    && rm -rf /var/lib/apt/lists/*

USER stockula
CMD ["/bin/bash"]
