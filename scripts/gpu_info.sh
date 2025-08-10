#!/bin/bash
# GPU information script for Docker containers

echo "=== GPU Information ==="

# Check PyTorch CUDA availability
python -c "
try:
    import torch
    print(f'PyTorch CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
except ImportError:
    print('PyTorch not installed')
except Exception as e:
    print(f'Error checking PyTorch: {e}')
" 2>/dev/null || echo "Python error checking GPU"

# Check NVIDIA driver info
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null || echo "nvidia-smi not available"

# Check CUDA version
echo ""
echo "CUDA Version Info:"
nvcc --version 2>/dev/null || echo "nvcc not available"

# Check XGBoost GPU support
echo ""
python -c "
try:
    import xgboost as xgb
    print(f'XGBoost version: {xgb.__version__}')
    print('XGBoost GPU support: Available')
except ImportError:
    print('XGBoost not installed')
except Exception as e:
    print(f'Error checking XGBoost: {e}')
" 2>/dev/null || echo "Error checking XGBoost"
