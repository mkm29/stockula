# GPU Acceleration for Stockula

This guide explains how to set up and use GPU acceleration for time series forecasting in Stockula using NVIDIA CUDA.

## Overview

GPU acceleration significantly improves performance for computationally expensive time series forecasting tasks. Stockula's GPU support includes:

- **NVIDIA CUDA 12.1** with cuDNN for optimal GPU compute performance
- **PyTorch with CUDA support** for neural network models (PytorchForecasting, TiDE, NeuralForecast)
- **TensorFlow with GPU support** for TensorFlow-based regression models
- **GluonTS** for advanced probabilistic forecasting models
- **Rapids (cuDF/cuML)** for GPU-accelerated data processing
- **GPU-optimized XGBoost and LightGBM** for gradient boosting models

## Prerequisites

### Hardware Requirements

- **NVIDIA GPU** with CUDA Compute Capability 6.0+ (Pascal architecture or newer)
- **8GB+ VRAM recommended** for large datasets and complex models
- **16GB+ system RAM** for data preprocessing and model coordination

### Software Requirements

1. **NVIDIA GPU Drivers** >= 450.80.02

   ```bash
   # Check driver version
   nvidia-smi
   ```

1. **Docker with GPU support**:

   ```bash
   # Install NVIDIA Container Runtime
   curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
     sudo apt-key add -
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
   sudo apt-get update
   sudo apt-get install nvidia-container-runtime
   ```

1. **Docker Compose** >= 1.28.0 (for GPU compose support)

## Quick Start

### 1. Build GPU-Enabled Image

```bash
# Build production GPU image
docker buildx build -f Dockerfile.nvidia --target gpu-production -t stockula:gpu .

# Or build interactive CLI version
docker buildx build -f Dockerfile.nvidia --target gpu-cli -t stockula:gpu-cli .
```

### 2. Run with Docker Compose

```bash
# Start GPU-enabled CLI
docker compose -f docker-compose.gpu.yml run stockula-gpu-cli

# Run Jupyter Lab with GPU support
docker compose -f docker-compose.gpu.yml up stockula-jupyter-gpu
```

### 3. Verify GPU Access

```bash
# Inside the container, check GPU availability
./gpu_info.sh

# Expected output:
# === GPU Information ===
# PyTorch CUDA: True, Devices: 1
# TensorFlow GPUs: 1
# name, memory.total [MiB], memory.used [MiB]
# NVIDIA GeForce RTX 4090, 24564, 150
```

## GPU-Accelerated Models

AutoTS includes several models that benefit from GPU acceleration:

### Neural Network Models

- **PytorchForecasting**: LSTM, GRU, TFT (Temporal Fusion Transformer)
- **NeuralForecast**: NBEATS, NHITS, TiDE
- **TiDE**: Time-series Dense Encoder for long-term forecasting
- **GluonTS**: DeepAR, Transformer, MQ-CNN

### Gradient Boosting Models

- **XGBoost**: `tree_method='gpu_hist'`
- **LightGBM**: `device_type='gpu'`

### TensorFlow Models

- **TensorFlow Regressions**: Dense neural networks for feature-based forecasting

## Configuration for GPU

### Environment Variables

The GPU Docker images include optimized environment variables:

```bash
# Enable GPU support
AUTOTS_GPU_ENABLED=true

# Reduce parallel jobs for stability (AutoTS recommendation)
AUTOTS_N_JOBS=1

# TensorFlow GPU settings
TF_FORCE_GPU_ALLOW_GROWTH=true

# PyTorch CUDA architecture support
TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 9.0+PTX"
```

### Stockula Configuration

Update your `.stockula.yaml` to use GPU-optimized models:

```yaml
forecast:
  # Use models with GPU support
  model_list: "gpu_optimized"  # Custom preset for GPU models
  
  # Enable GPU-accelerated models
  use_gpu: true
  
  # Reduce parallel workers for GPU stability
  max_workers: 1
  
  # Longer timeout for GPU model initialization
  model_timeout: 300
  
  # Enable advanced models
  enable_deep_learning: true

# GPU-specific model configuration
gpu_models:
  # PytorchForecasting models
  pytorch_forecasting:
    - TemporalFusionTransformer
    - DeepAR
    - LSTM
  
  # NeuralForecast models  
  neural_forecast:
    - NBEATS
    - NHITS
    - TiDE
  
  # GluonTS models
  gluonts:
    - DeepAR
    - Transformer
    - MQ_CNN
```

## Usage Examples

### Command Line

```bash
# Run GPU-accelerated forecasting
docker compose -f docker-compose.gpu.yml run stockula-gpu-cli \
  python -m stockula --config .stockula.yaml --mode forecast

# Backtest with GPU acceleration
docker compose -f docker-compose.gpu.yml run stockula-gpu-cli \
  python -m stockula --config .stockula.yaml --mode backtest
```

### Python API

```python
import torch
from stockula.forecasting.gpu_manager import GPUForecastManager
from stockula.config.settings import load_config

# Verify GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Devices: {torch.cuda.device_count()}")

# Load configuration
config = load_config('.stockula.yaml')

# Create GPU-enabled forecast manager
gpu_manager = GPUForecastManager(config)

# Run GPU-accelerated forecasting
results = gpu_manager.forecast_portfolio(
    model_list=['TiDE', 'NBEATS', 'DeepAR'],
    use_gpu=True
)
```

### Jupyter Notebook

```bash
# Start Jupyter with GPU support
docker compose -f docker-compose.gpu.yml up stockula-jupyter-gpu

# Access at http://localhost:8888 (token: stockula)
```

Example notebook cell:

```python
# Check GPU in Jupyter
import torch
import tensorflow as tf

print(f"PyTorch CUDA: {torch.cuda.is_available()}")
print(f"TensorFlow GPUs: {len(tf.config.list_physical_devices('GPU'))}")

# Run GPU-accelerated AutoTS
from stockula.forecasting import StockForecaster

forecaster = StockForecaster(
    model_list=['TiDE', 'NBEATS'],
    use_gpu=True,
    max_generations=5
)

forecast = forecaster.forecast_symbol('AAPL', periods=30)
```

## Performance Optimization

### Memory Management

```python
# Monitor GPU memory usage
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# Clear GPU cache when needed
torch.cuda.empty_cache()
```

### Batch Size Optimization

```python
# Optimize batch sizes based on GPU memory
def get_optimal_batch_size():
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory_gb >= 24:
        return 512  # High-end GPU
    elif gpu_memory_gb >= 16:
        return 256  # Mid-range GPU
    elif gpu_memory_gb >= 8:
        return 128  # Entry-level GPU
    else:
        return 64   # Low memory
```

## Monitoring and Debugging

### GPU Utilization

```bash
# Monitor GPU usage during training
nvidia-smi -l 1

# Detailed GPU monitoring
nvidia-smi dmon -s puc -d 5
```

### Memory Profiling

```python
# PyTorch memory profiling
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    # Run your forecasting code here
    results = forecaster.forecast(data)

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   ```python
   # Solution: Reduce batch size or use gradient accumulation
   torch.cuda.empty_cache()  # Clear cache
   # Restart with smaller batch_size or fewer parallel jobs
   ```

1. **Driver Version Mismatch**

   ```bash
   # Check compatibility
   nvidia-smi
   nvidia-container-cli --version
   ```

1. **Container Can't See GPU**

   ```bash
   # Test GPU access
   docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi
   ```

1. **Model Crashes (AutoTS GPU Instability)**

   ```yaml
   # In .stockula.yaml, reduce parallel processing
   forecast:
     max_workers: 1
     n_jobs: 1
   ```

### Performance Tips

1. **Use Mixed Precision Training**

   ```python
   # Enable automatic mixed precision
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()
   with autocast():
       # Your forecasting code here
   ```

1. **Optimize Data Pipeline**

   ```python
   # Use GPU-accelerated data loading
   import cudf

   # Load data directly to GPU
   data = cudf.read_csv('data.csv')
   ```

1. **Model Caching**

   ```python
   # Cache trained models on GPU
   torch.save(model.state_dict(), 'model_gpu.pth')
   model.load_state_dict(torch.load('model_gpu.pth'))
   ```

## Benchmarking

Expected performance improvements with GPU acceleration:

| Model Type            | CPU Time | GPU Time | Speedup |
| --------------------- | -------- | -------- | ------- |
| TiDE                  | 45 min   | 8 min    | 5.6x    |
| NBEATS                | 30 min   | 5 min    | 6x      |
| DeepAR                | 60 min   | 12 min   | 5x      |
| TensorFlow Regression | 20 min   | 4 min    | 5x      |
| XGBoost               | 15 min   | 3 min    | 5x      |

*Benchmarks based on 10,000 time series with 1000 observations each*

## Advanced Configuration

### Custom GPU Model Selection

```python
# Create custom GPU-optimized model list
gpu_model_list = {
    "name": "gpu_optimized",
    "models": [
        "TiDE",
        "NBEATS", 
        "NHITS",
        "TemporalFusionTransformer",
        "DeepAR",
        "Transformer"
    ]
}
```

### Multi-GPU Support

```python
# Enable multi-GPU training (experimental)
import torch.nn as nn

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")
```

For additional help and GPU-specific issues, see the [troubleshooting guide](../troubleshooting.md#gpu-acceleration) or check the AutoTS documentation on GPU support.
