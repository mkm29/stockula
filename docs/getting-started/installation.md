# Installation

Stockula offers multiple installation methods to suit different needs, from Docker containers with GPU support to local
development setups.

## Prerequisites

### System Requirements

- **Python**: 3.11 or higher (3.11 recommended for GPU support)
- **Operating System**: macOS, Linux, or Windows
- **Memory**: Minimum 8GB RAM recommended (16GB+ for GPU operations)
- **Storage**: 100MB free space (more if caching extensive historical data)
- **GPU** (optional): NVIDIA GPU with CUDA 11.8+ for acceleration

### Install uv

First, install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows users:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation Steps

1. **Clone the repository**:

   ```bash
   git clone https://github.com/mkm29/stockula.git
   cd stockula
   ```

1. **Install dependencies**:

   ```bash
   uv sync
   ```

1. **Verify installation**:

   ```bash
   uv run python -m stockula --help
   ```

## Development Installation

If you plan to contribute to Stockula:

```bash
# Clone with development dependencies
git clone https://github.com/mkm29/stockula.git
cd stockula

# Install all dependencies including dev tools
uv sync --all-extras

# Install pre-commit hooks (optional)
uv run pre-commit install
```

## Docker Installation (Recommended for GPU)

Docker provides the easiest way to get started, especially for GPU acceleration:

### CPU Version

```bash
# Pull the latest image
docker pull ghcr.io/mkm29/stockula:latest

# Run a simple analysis
docker run ghcr.io/mkm29/stockula:latest -m stockula --ticker AAPL

# Run with persistent data
docker run -v $(pwd)/data:/app/data ghcr.io/mkm29/stockula:latest \
    -m stockula --ticker NVDA --mode forecast
```

### GPU-Accelerated Version

Based on PyTorch 2.8.0 with Python 3.11 and advanced time series models:

```bash
# Pull the GPU image
docker pull ghcr.io/mkm29/stockula-gpu:latest

# Run with GPU support
docker run --gpus all ghcr.io/mkm29/stockula-gpu:latest \
    -m stockula --ticker AAPL --mode forecast --days 30

# Check GPU availability
docker run --gpus all ghcr.io/mkm29/stockula-gpu:latest \
    bash -c "/home/stockula/gpu_info.sh"
```

**GPU Image Features:**

- PyTorch 2.8.0 with CUDA 12.9 and cuDNN 9
- Chronos for zero-shot forecasting
- GluonTS for probabilistic models
- AutoGluon TimeSeries with full GPU support
- Runs as non-root user `stockula` (UID 1000)

## Troubleshooting

### Common Issues

**ImportError: No module named 'stockula'**

- Ensure you're running commands with `uv run` prefix
- Verify installation with `uv sync`

**Permission denied errors**

- On Linux/macOS, you may need to make the install script executable
- Try `chmod +x` on the install script

**Python version conflicts**

- Use `uv python install 3.13` to install Python 3.13
- Verify version with `python --version`

### Platform-Specific Notes

=== "macOS" - Install via Homebrew: `brew install uv` - Xcode Command Line Tools may be required for some dependencies

=== "Linux" - Some distributions may require additional packages for compilation - Ubuntu/Debian:
`sudo apt-get install build-essential python3-dev`

=== "Windows" - Windows Subsystem for Linux (WSL) is recommended - Ensure Visual Studio Build Tools are installed for
compiled dependencies

## Next Steps

After installation, check out:

- [Quick Start Guide](../getting-started/quick-start.md)
- [Configuration Options](../getting-started/configuration.md)
- [Architecture Overview](../user-guide/architecture.md)

## Runtime Setup Note

> Note: No import-time side effects
>
> Stockula avoids configuring logging, warnings, or environment variables during import. Runtime configuration is
> applied by the CLI (`python -m stockula`) via `stockula.main.setup_logging`. If you embed Stockula in your own
> application, call `setup_logging(config)` yourself (or provide your own logging setup) before invoking managers.

Example:

```python
from stockula.main import setup_logging
from stockula.config import load_config

cfg = load_config(".stockula.yaml")
setup_logging(cfg)  # configure logging/warnings/env at runtime

# proceed to use Stockula components
```
