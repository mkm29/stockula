# Multi-stage Dockerfile for Stockula trading library
# Based on best practices from:
# - https://dev.to/kummerer94/multi-stage-docker-builds-for-pyton-projects-using-uv-223g
# - https://pythonspeed.com/articles/multi-stage-docker-python/
# - https://pythonspeed.com/articles/activate-virtualenv-dockerfile/

# Stage 1: Base image with uv and system dependencies
FROM python:3.13-slim AS base

# Build arguments for labels
ARG VERSION=dev
ARG BUILD_DATE
ARG GIT_COMMIT
ARG GIT_URL

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Stage 2: Build stage - install dependencies
FROM base AS builder

# Set working directory
WORKDIR /app

# Copy uv configuration files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN uv venv /opt/venv --python 3.13
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install production dependencies
RUN uv sync --frozen --no-dev

# Stage 3: Development stage (for testing and development)
FROM builder AS development

# Install development dependencies
RUN uv sync --frozen

# Copy source code
COPY . .

# Install the package in editable mode
RUN uv pip install -e .

# Expose port for Jupyter
EXPOSE 8888

# Default command for development
CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Stage 4: Test stage
FROM development AS test

# Run tests
RUN uv run pytest tests/ --verbose --cov=src/stockula --cov-report=term-missing

# Run linting
RUN uv run ruff check src/ tests/

# Stage 5: Production stage - minimal runtime image
FROM base AS production

# Re-declare build arguments for this stage
ARG VERSION=dev
ARG BUILD_DATE
ARG GIT_COMMIT
ARG GIT_URL

# Add labels following OCI Image Format Specification
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.authors="Stockula Contributors" \
      org.opencontainers.image.url="${GIT_URL}" \
      org.opencontainers.image.documentation="${GIT_URL}" \
      org.opencontainers.image.source="${GIT_URL}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.vendor="Stockula" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.title="Stockula" \
      org.opencontainers.image.description="Quantitative trading platform with backtesting and forecasting"

# Create non-root user for security
RUN groupadd --gid 1000 stockula && \
    useradd --uid 1000 --gid stockula --shell /bin/bash --create-home stockula

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy source code and necessary files
COPY --chown=stockula:stockula src/ src/
COPY --chown=stockula:stockula pyproject.toml ./
COPY --chown=stockula:stockula README.md ./
COPY --chown=stockula:stockula alembic.ini ./
COPY --chown=stockula:stockula alembic/ alembic/
COPY --chown=stockula:stockula examples/ examples/

# Set up environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV STOCKULA_VERSION="${VERSION}"

# Install the package
RUN /opt/venv/bin/python -m pip install -e .

# Create directories for data and results
RUN mkdir -p /app/data /app/results && \
    chown -R stockula:stockula /app

# Switch to non-root user
USER stockula

# Set up volumes for persistent data
VOLUME ["/app/data", "/app/results"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import stockula; print('Stockula is healthy')" || exit 1

# Default command
CMD ["python", "-m", "stockula.main", "--help"]

# Stage 6: CLI stage - optimized for command-line usage
FROM production AS cli

# Install additional CLI tools if needed
USER root
RUN apt-get update && apt-get install -y \
    less \
    nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER stockula

# Default to interactive shell
CMD ["/bin/bash"]

# Stage 7: API stage - for running as a service (if API endpoints are added)
FROM production AS api

# Expose port for API
EXPOSE 8000

# Install additional production dependencies for API
USER root
RUN /opt/venv/bin/python -m pip install fastapi uvicorn

USER stockula

# Command for API server (placeholder for future API implementation)
CMD ["python", "-c", "print('API service not yet implemented. Use CLI stage instead.')"]

# Stage 8: Jupyter stage - for interactive analysis
FROM production AS jupyter

# Install Jupyter in production venv
USER root
RUN /opt/venv/bin/python -m pip install jupyter jupyterlab

USER stockula

# Create Jupyter config directory
RUN mkdir -p /home/stockula/.jupyter

# Copy notebook examples
COPY --chown=stockula:stockula notebooks/ notebooks/

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]