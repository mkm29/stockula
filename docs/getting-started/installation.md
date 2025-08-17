# Installation

Stockula offers multiple installation methods to suit different needs, from Docker containers with GPU support to local
development setups.

## Prerequisites

### System Requirements

- **Python**: 3.11 or higher (3.11 recommended for GPU support)
- **Operating System**: macOS, Linux, or Windows
- **Memory**: Minimum 8GB RAM recommended (16GB+ for GPU operations)
- **Storage**: 100MB free space (more if caching extensive historical data)
- **Database**: TimescaleDB (PostgreSQL with TimescaleDB extension)
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

1. **Setup TimescaleDB**:

   ```bash
   # Using Docker (recommended)
   docker run -d --name stockula-timescaledb \
     -e POSTGRES_USER=stockula_user \
     -e POSTGRES_PASSWORD=stockula_password \
     -e POSTGRES_DB=stockula \
     -p 5432:5432 \
     timescale/timescaledb:latest-pg16

   # Or follow the detailed setup guide
   # See TIMESCALEDB_QUICKSTART.md for complete instructions
   ```

1. **Configure environment**:

   ```bash
   # Set TimescaleDB connection details
   export STOCKULA_DB_HOST=localhost
   export STOCKULA_DB_PORT=5432
   export STOCKULA_DB_NAME=stockula
   export STOCKULA_DB_USER=stockula_user
   export STOCKULA_DB_PASSWORD=stockula_password
   ```

1. **Install dependencies**:

   ```bash
   uv sync
   ```

1. **Initialize database schema with simplified manager**:

   ```bash
   # Single consolidated manager handles all setup
   uv run python -m stockula.database.manager setup
   ```

1. **Verify installation**:

   ```bash
   uv run python -m stockula --help
   # Check the consolidated manager status
   uv run python -m stockula.database.manager status
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

## Docker Installation (Recommended)

Docker provides the easiest way to get started with both the application and TimescaleDB:

### Quick Start with Docker Compose

```bash
# Start both Stockula and TimescaleDB
docker compose -f docker-compose.timescale.yml up -d

# Run analysis with integrated setup
docker compose exec stockula python -m stockula --ticker AAPL

# Check TimescaleDB status using consolidated manager
docker compose exec stockula python -m stockula.database.manager status
```

### Standalone CPU Version

```bash
# Pull the latest image
docker pull ghcr.io/mkm29/stockula:latest

# Run with external TimescaleDB
docker run --network host \
  -e STOCKULA_DB_HOST=localhost \
  -e STOCKULA_DB_PASSWORD=your_password \
  ghcr.io/mkm29/stockula:latest \
  -m stockula --ticker AAPL
```

### GPU-Accelerated Version

Based on PyTorch 2.8.0 with Python 3.11 and advanced time series models:

```bash
# Pull the GPU image
docker pull ghcr.io/mkm29/stockula-gpu:latest

# Run with GPU support and TimescaleDB
docker run --gpus all --network host \
    -e STOCKULA_DB_HOST=localhost \
    -e STOCKULA_DB_PASSWORD=your_password \
    ghcr.io/mkm29/stockula-gpu:latest \
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
- TimescaleDB connectivity for high-performance data storage
- Runs as non-root user `stockula` (UID 1000)

## Troubleshooting

### Common Issues

**ImportError: No module named 'stockula'**

- Ensure you're running commands with `uv run` prefix
- Verify installation with `uv sync`

**TimescaleDB connection errors**

- Verify TimescaleDB is running: `docker ps | grep timescaledb`
- Check connection settings: `echo $STOCKULA_DB_HOST`
- Test connection: `psql -h localhost -p 5432 -U stockula_user -d stockula`

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

- [TimescaleDB Quick Start](../../TIMESCALEDB_QUICKSTART.md) - Database setup guide
- [Quick Start Guide](../getting-started/quick-start.md) - Application usage
- [Configuration Options](../getting-started/configuration.md) - Full configuration reference
- [Architecture Overview](../user-guide/architecture.md) - System architecture

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
