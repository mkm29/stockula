# Docker GPU Build - Python 3.12 Migration

## Summary

Successfully migrated the Docker GPU build from Python 3.13 (which required problematic deadsnakes PPA) to Python 3.12
(which is included in Ubuntu 24.04 by default).

## Changes Made

### 1. Project Configuration (`pyproject.toml`)

- Changed `requires-python = ">=3.12"` (was `>=3.13`)
- Updated ruff `target-version = "py312"` (was `py313`)
- Updated mypy `python_version = "3.12"` (was `3.13`)

### 2. Dockerfile.nvidia Simplification

- **Removed**: All deadsnakes PPA references (source of exit code 100 error)
- **Using**: Ubuntu 24.04's default Python 3.12.3
- **Fixed**: Virtual environment creation with proper pip installation
- **Simplified**: No complex fallback logic needed

### 3. Key Docker Build Fixes

#### Original Error

```
ERROR: exit code: 100
add-apt-repository ppa:deadsnakes/ppa failed
```

#### Root Cause

- Trying to install Python 3.13 from deadsnakes PPA
- PPA may not be accessible or compatible with Docker build environment
- Ubuntu 24.04 base image issues with PPA

#### Solution

- Use Python 3.12 which is already in Ubuntu 24.04
- No external repositories needed
- All GPU packages support Python 3.12

### 4. Virtual Environment Setup

#### Issue

- `uv venv` creates venv without pip by default
- Need pip for GPU package installation with custom index URLs

#### Solution

```dockerfile
# Create venv with system Python 3.12
uv venv /opt/venv --python /usr/bin/python3
# Install project with uv
uv pip install --python /opt/venv/bin/python -e .
# Install pip for GPU packages
/opt/venv/bin/python -m ensurepip --default-pip
```

## Benefits of Python 3.12

1. **Native Support**: Included in Ubuntu 24.04 by default
1. **Full GPU Package Support**: All packages available
   - PyTorch ✅
   - TensorFlow ✅
   - XGBoost ✅
   - LightGBM ✅
   - MXNet ✅
   - GluonTS ✅
1. **Simpler Dockerfile**: No PPA management
1. **Faster Builds**: No external repository downloads
1. **More Reliable**: No network dependencies for Python installation

## Build Command

```bash
docker buildx build \
  -t smigula/stockula:v0.13.0-gpu \
  -f Dockerfile.nvidia \
  --platform linux/amd64 \
  --target gpu-cli \
  .
```

## Testing

Use the test script to verify build stages:

```bash
# Verify build configuration
uv run verify-build --gpu

# Or test the actual Docker build
docker buildx build -f Dockerfile.nvidia --target gpu-cli -t stockula:gpu-test .
```

## Files Modified

1. `pyproject.toml` - Python version requirements
1. `Dockerfile.nvidia` - Complete simplification for Python 3.12
1. `CLAUDE.md` - Updated Python version documentation
1. Various scripts for testing and verification

## Migration Complete

The Docker GPU build now uses Python 3.12 and should build successfully without PPA-related errors.
