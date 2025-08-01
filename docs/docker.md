# Docker Documentation

This document describes how to build, run, and optimize Stockula Docker images.

## Overview

Stockula provides optimized Docker images for running the trading platform in containerized environments. The images are:

- Multi-platform (amd64, arm64)
- Multi-stage for minimal size
- Optimized for fast builds with advanced caching
- Security-hardened with non-root user
- Available on GitHub Container Registry

## Quick Start

### Pull Pre-built Image

```bash
# Pull latest CLI version (default, includes interactive tools)
docker pull ghcr.io/mkm29/stockula:latest
docker pull ghcr.io/mkm29/stockula:cli

# Pull specific CLI version
docker pull ghcr.io/mkm29/stockula:0.4.1-cli
docker pull ghcr.io/mkm29/stockula:0.4-cli

# Pull production version (minimal, no extra tools)
docker pull ghcr.io/mkm29/stockula:production
docker pull ghcr.io/mkm29/stockula:0.4.1-production
docker pull ghcr.io/mkm29/stockula:0.4-production
```

### Run Container

```bash
# Run with default help (production image)
docker run --rm ghcr.io/mkm29/stockula:production

# Run analysis with production image
docker run --rm -v $(pwd)/data:/app/data \
  ghcr.io/mkm29/stockula:production \
  stockula.main --ticker AAPL --mode ta

# Interactive shell with CLI image
docker run -it --rm -v $(pwd)/data:/app/data \
  ghcr.io/mkm29/stockula:cli bash

# Or use latest (which is CLI)
docker run -it --rm -v $(pwd)/data:/app/data \
  ghcr.io/mkm29/stockula:latest bash
```

## Building Images

### Local Build

```bash
# Build for current platform
docker buildx build -t stockula:local --target cli .

# Build specific stage
docker buildx build -t stockula:prod --target production .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 \
  -t stockula:multi --target cli .
```

### Build Stages

The Dockerfile uses multiple stages for optimization:

1. **base**: Base image with system dependencies
1. **dependencies**: Python virtual environment with packages
1. **source**: Application source code
1. **production**: Minimal runtime image (default for automated tools)
1. **cli**: Production + CLI tools (default for interactive use)

## Optimization Techniques

### Build Performance

Our Docker builds are optimized for speed using several techniques:

#### 1. Layer Caching

```dockerfile
# Dependencies cached separately from source
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

# Source copied later
COPY src/ src/
```

#### 2. Cache Mounts

```dockerfile
# APT cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y ...

# UV cache mount
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev
```

#### 3. Registry Caching

The CI/CD pipeline uses registry caching for persistence:

```yaml
cache-from: |
  type=gha
  type=registry,ref=ghcr.io/mkm29/stockula:buildcache
cache-to: |
  type=gha,mode=max
  type=registry,ref=ghcr.io/mkm29/stockula:buildcache,mode=max
```

### Image Size Optimization

- Minimal base image (bookworm-slim)
- Multi-stage builds (only runtime dependencies in final image)
- No development tools in production
- Compiled Python bytecode

### Security

- Non-root user (`stockula`)
- Read-only root filesystem compatible
- No unnecessary packages
- Regular security scanning with Docker Scout
- SHELL directive with pipefail for better error handling

### Labels and Metadata

The Docker images include comprehensive labels following OCI standards:

```bash
# View all labels
docker inspect stockula:latest | jq '.[0].Config.Labels'

# View specific labels
docker inspect -f '{{.Config.Labels}}' stockula:latest
```

**OCI Standard Labels:**

- `org.opencontainers.image.created`: Build timestamp
- `org.opencontainers.image.version`: Version tag
- `org.opencontainers.image.revision`: Git commit SHA
- `org.opencontainers.image.source`: Source repository URL
- `org.opencontainers.image.licenses`: MIT
- `org.opencontainers.image.description`: Image description

**Custom Labels:**

- `com.stockula.python.version`: Python version (3.13)
- `com.stockula.base.image`: Base image used
- `com.stockula.build.stage`: Build stage (production/cli)
- `com.stockula.maintainer`: Project maintainer

## Docker Scout Analysis

### Running Security Scans

```bash
# Build image
docker buildx build -t stockula:latest .

# Get recommendations
docker scout recommendations stockula:latest

# Check CVEs
docker scout cves stockula:latest

# View SBOM
docker scout sbom stockula:latest
```

### Current Security Status

As of the latest scan:

- **Critical**: 0
- **High**: 1 (PAM library)
- **Medium**: 3
- **Low**: 27

Most vulnerabilities are in system packages with no available fixes in Debian stable.

### Vulnerability Mitigation

1. **Regular Updates**: Rebuild images when base images are updated
1. **Minimal Attack Surface**: Only install required packages
1. **Runtime Security**: Use read-only filesystems where possible
1. **Network Security**: Don't expose unnecessary ports

## Volume Mounts

### Data Directory

```bash
docker run -v $(pwd)/data:/app/data stockula:latest
```

- Historical price data cache
- SQLite database files
- Downloaded datasets

### Results Directory

```bash
docker run -v $(pwd)/results:/app/results stockula:latest
```

- Backtest results
- Generated reports
- Export files

### Configuration

```bash
docker run -v $(pwd)/.config.yaml:/app/.config.yaml:ro stockula:latest
```

- Read-only mount for configuration
- Uses example config if not provided

## Environment Variables

```bash
# Set Stockula environment
docker run -e STOCKULA_ENV=production stockula:latest

# Database URL
docker run -e DATABASE_URL=sqlite:///data/prod.db stockula:latest

# API Keys (if needed)
docker run -e ALPHA_VANTAGE_KEY=your_key stockula:latest
```

## Docker Compose

Example `docker-compose.yml`:

```yaml
version: '3.8'

services:
  stockula:
    image: ghcr.io/mkm29/stockula:latest
    volumes:
      - ./data:/app/data
      - ./results:/app/results
      - ./.config.yaml:/app/.config.yaml:ro
    environment:
      - STOCKULA_ENV=production
    command: python -m stockula.main --mode forecast

  stockula-cli:
    image: ghcr.io/mkm29/stockula:cli
    volumes:
      - ./data:/app/data
      - ./results:/app/results
    stdin_open: true
    tty: true
    command: /bin/bash
```

## Troubleshooting

### Build Issues

1. **Slow builds**: Ensure Docker BuildKit is enabled

   ```bash
   export DOCKER_BUILDKIT=1
   ```

1. **Out of space**: Clean Docker cache

   ```bash
   docker system prune -a
   ```

1. **Platform errors**: Use `--platform` flag explicitly

   ```bash
   docker buildx build --platform linux/amd64 ...
   ```

### Runtime Issues

1. **Permission denied**: Check volume mount permissions
1. **Module not found**: Ensure using correct stage (cli vs production)
1. **Memory issues**: Increase Docker memory limit

## Best Practices

1. **Use specific tags**: Don't rely on `:latest` in production
1. **Resource limits**: Set memory and CPU limits
1. **Health checks**: Monitor container health
1. **Logging**: Use centralized logging for production
1. **Secrets**: Never hardcode secrets in images

## CI/CD Integration

The Docker build process is automated in GitHub Actions:

- Triggers on version tags (`v*`)
- Builds for multiple platforms in parallel
- Pushes to GitHub Container Registry
- Uses advanced caching strategies

See [CI/CD Documentation](development/ci-cd.md) for details.
