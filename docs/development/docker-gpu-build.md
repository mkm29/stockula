# Docker GPU Build Guide

## Overview

This guide documents the GPU-accelerated Docker build for Stockula, including configuration, best practices, and troubleshooting.

## Build Architecture

The GPU Docker build uses a **standalone approach** that is independent from the main development setup:

- **Standalone**: Uses `requirements-gpu.txt` directly, NOT `pyproject.toml`
- **Multi-stage**: Optimized build following Docker best practices
- **Python 3.12**: Uses Ubuntu 24.04's default Python (no external PPAs)
- **NVIDIA CUDA**: Base image with CUDA 13.0 runtime support

## Quick Start

### Build Command

```bash
docker buildx build \
  -f Dockerfile.nvidia \
  --platform linux/amd64 \
  --target gpu-cli \
  -t smigula/stockula:v0.13.0-gpu .
```

### Run Commands

```bash
# Run with GPU support
docker run --gpus all smigula/stockula:v0.13.0-gpu stockula --help

# Interactive shell
docker run --gpus all -it smigula/stockula:v0.13.0-gpu bash

# With volume mounts
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  smigula/stockula:v0.13.0-gpu stockula --ticker AAPL --mode ta

# Check GPU availability
docker run --gpus all smigula/stockula:v0.13.0-gpu gpu-check
```

## Dockerfile Structure

### Multi-Stage Build Design

```
┌─────────────────────────────────────────┐
│ Stage 1: Builder                        │
│ - NVIDIA CUDA base image                │
│ - Python 3.12 & build dependencies      │
│ - Virtual environment creation          │
│ - Core package installation             │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Stage 2: Runtime                        │
│ - Minimal CUDA runtime                  │
│ - Python runtime only                   │
│ - Copy venv from builder                │
│ - Non-root user (ubuntu)                │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Stage 3: GPU-CLI (Development)         │
│ - Additional debugging tools            │
│ - GPU diagnostic script                 │
│ - Interactive shell                     │
└─────────────────────────────────────────┘
```

### Stage Details

#### Stage 1: Builder

- **Base**: `nvidia/cuda:13.0.0-runtime-ubuntu24.04`
- **Purpose**: Compile dependencies and prepare virtual environment
- **Key Features**:
  - Python 3.12 with pip and venv
  - Build cache mounts for faster rebuilds
  - Core dependencies installed (skip heavy GPU packages for speed)

#### Stage 2: Runtime

- **Base**: Same CUDA runtime image
- **Purpose**: Minimal production image
- **Key Features**:
  - Only Python runtime and ca-certificates
  - Virtual environment copied from builder
  - Non-root user execution (ubuntu)
  - Health checks for orchestration

#### Stage 3: GPU-CLI

- **Base**: Extends runtime stage
- **Purpose**: Development and debugging
- **Key Features**:
  - Additional tools (less, vim-tiny, curl)
  - GPU diagnostic script at `/usr/local/bin/gpu-check`
  - Interactive bash shell

## Best Practices Implementation

### 1. Security

- ✅ Non-root user execution (existing `ubuntu` user)
- ✅ Minimal runtime dependencies
- ✅ No build tools in production image
- ✅ Proper file ownership with --chown

### 2. Performance

- ✅ Multi-stage build for smaller images
- ✅ Layer caching optimization
- ✅ BuildKit cache mounts for apt and pip
- ✅ Requirements copied before source code

### 3. Maintainability

- ✅ ARG variables for version management
- ✅ Environment variables consolidated
- ✅ OCI standard labels for metadata
- ✅ hadolint compliance with documented ignores

### 4. Size Optimization

- ✅ Build tools only in builder stage
- ✅ Package caches cleaned in same layer
- ✅ `--no-install-recommends` for all apt installs
- ✅ Heavy GPU packages optional (can be installed at runtime)

## Key Files

### `requirements-gpu.txt`

Contains all Python dependencies for GPU builds:

- Core data science packages (numpy, pandas, scikit-learn)
- Trading libraries (yfinance, backtesting, finta)
- Database and models (sqlalchemy, sqlmodel, pydantic)
- Forecasting (autots)
- GPU packages (torch, xgboost, lightgbm - optional)

### `Dockerfile.nvidia`

Multi-stage Dockerfile optimized for GPU builds with:

- BuildKit syntax for advanced features
- Cache mounts for efficient rebuilds
- Proper layer ordering
- Security best practices

### Scripts

- `scripts/verify_docker_build.sh` - Validates Docker configuration
- `/usr/local/bin/gpu-check` - Runtime GPU diagnostics (in container)

## GPU Package Support

### Core GPU Libraries

- **PyTorch**: CUDA 11.8+ support
- **XGBoost**: GPU acceleration
- **LightGBM**: GPU support
- **nvidia-ml-py**: GPU monitoring

### Optional GPU Packages

These can be installed at runtime if needed:

- TensorFlow with CUDA support
- MXNet with CUDA
- GluonTS for time series
- AutoGluon for AutoML

## Build Verification

### Build Time

- Typical build: ~50 seconds
- Rebuild (with cache): ~20 seconds

### Verification Steps

1. **Check Build Success**:

```bash
docker buildx build -f Dockerfile.nvidia --target gpu-cli -t test:gpu .
```

2. **Verify GPU Access**:

```bash
docker run --gpus all test:gpu nvidia-smi
```

3. **Test Python Import**:

```bash
docker run --gpus all test:gpu python -c "import stockula; print('Success')"
```

4. **Run GPU Diagnostic**:

```bash
docker run --gpus all test:gpu gpu-check
```

## Important Notes

### Python Version

- **Python 3.12**: Uses Ubuntu 24.04's default Python 3.12.3
- **No PPA needed**: No deadsnakes or external repositories
- **Full compatibility**: All packages support Python 3.12

### Build Independence

1. GPU Docker build is **self-contained**
1. Does **NOT** use `pyproject.toml`
1. All dependencies in `requirements-gpu.txt`
1. Source copied directly, not installed as package
1. Uses system Python 3.12 (no PPA)

### User Management

- Uses existing `ubuntu` user (UID 1000)
- No need to create additional users
- Proper permissions set with chown

## Troubleshooting

### Common Issues

#### 1. Build Fails with Package Version Error

**Solution**: Remove version pins and use latest compatible versions

```dockerfile
# Instead of specific versions
RUN apt-get install -y python3=3.12.3-0ubuntu2

# Use unversioned
RUN apt-get install -y python3
```

#### 2. GPU Not Available in Container

**Solution**: Ensure NVIDIA Container Toolkit is installed

```bash
# Check NVIDIA runtime
docker info | grep nvidia

# Run with GPU support
docker run --gpus all ...
```

#### 3. Import Errors at Runtime

**Solution**: Verify Python path configuration

```bash
# Check Python path in container
docker run test:gpu python -c "import sys; print(sys.path)"
```

#### 4. Build Timeout

**Solution**: Skip heavy GPU packages during build

- Install only core packages in Dockerfile
- Install GPU packages at runtime if needed

### Validation Script

Run the included validation script to check configuration:

```bash
./scripts/verify_docker_build.sh
```

This verifies:

- ✓ requirements-gpu.txt exists
- ✓ No pyproject.toml references in Dockerfile
- ✓ No Python 3.13 specific files
- ✓ Using Python 3.12
- ✓ No deadsnakes PPA

## Performance Metrics

### Image Sizes

- **Builder stage**: ~2.5GB (includes build tools)
- **Runtime stage**: ~1.2GB (minimal runtime)
- **GPU-CLI stage**: ~1.3GB (adds dev tools)

### Build Performance

- **Initial build**: ~50 seconds
- **Cached rebuild**: ~20 seconds
- **Source-only change**: ~5 seconds

## Future Improvements

1. **Layer Optimization**: Further reduce layers in runtime stage
1. **Multi-arch Support**: Add ARM64 support for Apple Silicon
1. **GPU Package Variants**: Create slim vs full GPU variants
1. **Compose Integration**: Add docker-compose.gpu.yml
1. **CI/CD Integration**: Automated GPU image builds

## References

- [Docker Build Best Practices](https://docs.docker.com/build/building/best-practices/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [hadolint - Dockerfile Linter](https://github.com/hadolint/hadolint)
- [OCI Image Specification](https://github.com/opencontainers/image-spec)
