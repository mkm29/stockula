# Docker GPU Build Fixes for Python 3.13

## Issues Addressed

### 1. Python 3.13 Path Issues

- **Problem**: `uv venv` couldn't find Python 3.13 after installation from deadsnakes PPA
- **Fix**: Use explicit path `/usr/bin/python3.13` when creating virtual environment

### 2. Wrong Requirements File

- **Problem**: Dockerfile was using `requirements-gpu.txt` which contains Python 3.12-only packages
- **Fix**: Changed to use `requirements-gpu-py313.txt` which only includes Python 3.13 compatible packages

### 3. Script Creation Issues

- **Problem**: Complex echo commands for creating gpu_info.sh were error-prone
- **Fix**: Created separate `scripts/gpu_info.sh` file and copy it into the image

### 4. Added Verification Steps

- **Problem**: Silent failures when Python 3.13 wasn't properly installed
- **Fix**: Added `/usr/bin/python3.13 --version` check before creating venv

## Remaining Potential Issues

1. **Deadsnakes PPA Access**: The PPA might not be accessible in some Docker build environments

   - **Workaround**: Could build Python 3.13 from source if PPA fails

1. **Package Compatibility**: Some packages might still have hidden Python 3.13 issues

   - **Current Mitigation**: Fallback logic to install core packages if full install fails

1. **CUDA Version Mismatch**: The base image uses CUDA 13.0, but PyTorch might expect different versions

   - **Current Mitigation**: Using PyTorch with CUDA 11.8 support via extra-index-url

## Build Command

```bash
docker buildx build \
  -t smigula/stockula:v0.13.0-gpu \
  -f Dockerfile.nvidia \
  --platform linux/amd64 \
  --target gpu-cli \
  .
```

## Verification

After build, run:

```bash
docker run --gpus all smigula/stockula:v0.13.0-gpu /home/stockula/gpu_info.sh
```

This will show:

- PyTorch CUDA availability
- GPU device information
- XGBoost GPU support
- NVIDIA driver info
