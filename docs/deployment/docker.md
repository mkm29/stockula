# Docker Deployment Guide

This guide explains how to build and run Stockula using Docker for different use cases. The Stockula project uses a comprehensive multi-stage Docker setup following industry best practices.

## Overview

The Docker implementation provides optimized containers for different scenarios using a multi-stage Dockerfile with 8 specialized stages:

- **base**: Common foundation with Python 3.13 and uv package manager
- **builder**: Dependencies installation with virtual environment
- **development**: Full development environment with all dependencies
- **test**: Testing environment with coverage and linting tools
- **production**: Minimal production runtime (~500MB)
- **cli**: Command-line interface optimized image (~550MB)
- **api**: API server (placeholder for future implementation)
- **jupyter**: Interactive analysis with Jupyter Lab (~1.5GB)

## Quick Start

### Using Docker Compose (Recommended)

The easiest way to get started is using Docker Compose:

```bash
# Start development environment with Jupyter Lab
docker-compose up stockula-dev
# Access Jupyter Lab at http://localhost:8888

# Or start CLI environment
docker-compose up stockula-cli

# Run tests
docker-compose up stockula-test
```

### Using Makefile (Recommended)

For convenience, use the provided Makefile commands:

```bash
# Start development environment
make dev

# Run tests in Docker
make test

# Start CLI environment
make cli

# Build all images
make build-all

# Show all available commands
make help
```

### Building Individual Images

```bash
# Build production image
docker build --target production -t stockula:latest .

# Build development image
docker build --target development -t stockula:dev .

# Build CLI image
docker build --target cli -t stockula:cli .

# Build Jupyter image
docker build --target jupyter -t stockula:jupyter .
```

## Services and Use Cases

### 1. Development Environment

For active development with hot reloading and all dependencies:

```bash
# Using Docker Compose
docker-compose up stockula-dev

# Using Makefile
make dev

# Or using docker run directly
docker run -it --rm \
  -v $(pwd):/app \
  -v stockula-data:/app/data \
  -p 8888:8888 \
  stockula:dev
```

**Features:**

- Jupyter Lab accessible at `http://localhost:8888`
- Hot reloading for live code changes
- All development dependencies included
- Volume mounting for persistent data

### 2. Production CLI Usage

For running backtests and analyses in production:

```bash
# Interactive shell
make cli
# or
docker-compose run --rm stockula-cli

# Run specific example
make run-example-dynamic
# or
docker run --rm \
  -v stockula-data:/app/data \
  -v stockula-results:/app/results \
  stockula:cli \
  python examples/automatic_dynamic_rates_example.py

# Run with custom configuration
docker run --rm \
  -v $(pwd)/examples:/app/config:ro \
  -v stockula-data:/app/data \
  -v stockula-results:/app/results \
  stockula:cli \
  python -m stockula.main --config /app/config/config.example.yaml
```

### 3. Jupyter Analysis Environment

For interactive data analysis and backtesting:

```bash
# Start Jupyter service
make jupyter
# or
docker-compose up stockula-jupyter

# Access at http://localhost:8889
```

### 4. Running Tests

```bash
# Run all tests
make test
# or
docker-compose up stockula-test

# Run specific test suites
make test-unit       # Unit tests only
make test-integration # Integration tests only
make test-coverage   # With coverage report

# Run with custom options
docker run --rm \
  -v $(pwd):/app \
  stockula:test \
  uv run pytest tests/unit/ -v
```

## Data Persistence

The Docker setup uses named volumes for data persistence:

- **stockula-data**: Market data cache and database files
- **stockula-results**: Backtest results and reports

### Managing Data Volumes

```bash
# List volumes
make list-volumes

# Inspect volumes
make inspect-data-volume
make inspect-results-volume

# Backup data volume
make backup-data

# Restore data volume (requires BACKUP_FILE variable)
make restore-data BACKUP_FILE=stockula-data-backup-20240115-143022.tar.gz
```

### Manual Volume Operations

```bash
# Backup data volume manually
docker run --rm \
  -v stockula-data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/stockula-data-backup.tar.gz -C /data .

# Restore data volume manually
docker run --rm \
  -v stockula-data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/stockula-data-backup.tar.gz -C /data
```

## Environment Configuration

Configure the application using environment variables:

```bash
# Set environment in docker-compose.yml or pass to docker run
STOCKULA_ENV=production          # Environment: development, production, test
PYTHONPATH=/app/src             # Python path for imports
STOCKULA_LOG_LEVEL=INFO         # Logging level
STOCKULA_DATABASE_URL=...       # Database connection string
```

## Performance Optimization

### Image Size Optimization

The multi-stage build strategy minimizes final image sizes:

- **Development**: ~1.5GB (includes all dev tools and Jupyter)
- **Production**: ~500MB (minimal runtime)
- **CLI**: ~550MB (includes basic CLI tools)

### Build Cache Optimization

```bash
# Use BuildKit for better caching
export DOCKER_BUILDKIT=1

# Build with cache from registry
docker build --cache-from stockula:latest -t stockula:latest .

# Multi-platform builds
make buildx-setup
make buildx-build
```

## Available Examples

Run the provided examples to test functionality:

```bash
# Dynamic Treasury rates example
make run-example-dynamic

# Treasury rates fetching example
make run-example-treasury

# Dynamic Sharpe ratio calculation example
make run-example-sharpe
```

## Development Workflow

### Code Quality Tools

```bash
# Run linting
make lint

# Format code
make format

# Check formatting
make format-check

# Check dependencies for vulnerabilities
make check-deps
```

### Quick Development Cycle

```bash
# Quick build and test cycle
make quick-test

# Quick build and start development
make quick-dev
```

## Troubleshooting

### Common Issues

1. **Permission Errors**

   ```bash
   # Fix volume permissions
   docker run --rm -v stockula-data:/data alpine chown -R 1000:1000 /data
   ```

1. **Memory Issues with Large Datasets**

   ```bash
   # Increase memory limit
   docker run --memory=4g stockula:latest
   ```

1. **Port Conflicts**

   ```bash
   # Use different ports
   docker run -p 8889:8888 stockula:jupyter
   ```

### Debugging

```bash
# Access container shell
make cli-shell
make dev-shell

# Check logs
make logs          # Development logs
make logs-jupyter  # Jupyter logs

# Show running containers
make ps

# Monitor resource usage
docker stats

# Check container health
make health-check
```

### Clean Up

```bash
# Clean up Docker images and containers
make clean

# Clean up everything including volumes
make clean-all

# Show image sizes
make size
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Docker Build and Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build test image
        run: docker build --target test -t stockula:test .

      - name: Run tests
        run: docker run --rm stockula:test

      - name: Build production image
        run: docker build --target production -t stockula:latest .
```

### Production Deployment

```bash
# Tag for registry
make tag-version VERSION=v1.0.0

# Push to registry
make push-version VERSION=v1.0.0

# Deploy to production
docker run -d \
  --name stockula-prod \
  --restart unless-stopped \
  -v stockula-data:/app/data \
  -v stockula-results:/app/results \
  your-registry.com/stockula:v1.0.0
```

## Security Best Practices

1. **Non-root User**: Production images run as non-root user `stockula` (UID/GID 1000)
1. **Minimal Base**: Uses slim Python images to reduce attack surface
1. **No Secrets in Images**: Use environment variables or mounted secrets
1. **Volume Permissions**: Proper file permissions on mounted volumes
1. **Network Security**: Use custom networks for multi-container deployments
1. **Health Checks**: Built-in health checks for container monitoring

## Advanced Usage

### Custom Base Images

```dockerfile
# Use custom base image
FROM your-registry.com/python:3.13-custom as base
# ... rest of Dockerfile
```

### Multi-Architecture Builds

```bash
# Setup buildx
make buildx-setup

# Build for multiple architectures
make buildx-build
```

### Health Checks

The production image includes health checks:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' container-name
```

## Integration with Stockula Features

The Docker implementation seamlessly integrates with all Stockula features:

### ✅ Dynamic Treasury Rates

- Containers automatically use the enhanced `BacktestRunner`
- Dynamic Sharpe ratio calculations work out-of-the-box
- Treasury rate caching persists across container restarts

### ✅ Database Migrations

- Alembic migrations run automatically in containers
- Database files persist in mounted volumes
- Migration scripts included in all relevant images

### ✅ Examples and Documentation

- All existing examples work in Docker containers
- Examples directory mounted for easy testing
- Documentation includes Docker-specific usage

### ✅ Testing Framework

- Complete test suite runs in isolated containers
- Coverage reporting generates HTML reports
- Linting and formatting tools included

## Architecture Benefits

### 🚀 Performance Optimizations

- Multi-stage builds minimize final image sizes
- Layer caching optimized for development workflow
- uv package manager for fast dependency resolution

### 🔒 Security Best Practices

- Non-root user in production images
- Minimal base images reduce attack surface
- No secrets in images - uses environment variables
- Proper file permissions on mounted volumes

### 🛠 Developer Experience

- Hot reloading in development containers
- Jupyter Lab integration for interactive analysis
- Volume mounting for persistent data and live code changes
- Makefile shortcuts for common operations

### 📦 Production Ready

- Minimal runtime images for efficient deployment
- Health checks for container orchestration
- Multi-architecture support (AMD64/ARM64)
- CI/CD integration examples for automated testing

## Best Practices

1. **Layer Caching**: Order Dockerfile instructions from least to most frequently changing
1. **Multi-stage Builds**: Use appropriate target for your use case
1. **Volume Management**: Use named volumes for persistent data
1. **Security**: Always run production containers as non-root
1. **Resource Limits**: Set appropriate memory and CPU limits
1. **Logging**: Configure proper log drivers for production

## Support

For Docker-related issues:

1. Check the troubleshooting section above
1. Review Docker logs: `make logs`
1. Verify image builds: `make build-test`
1. Run validation script: `scripts/validate-docker.sh`
1. Submit issues to the [Stockula GitHub repository](https://github.com/mkm29/stockula/issues)

## Summary

This Docker implementation provides a production-ready, secure, and developer-friendly containerization solution for the Stockula trading library. It follows industry best practices while being tailored specifically for the project's needs, including comprehensive tooling for development, testing, and deployment scenarios.
