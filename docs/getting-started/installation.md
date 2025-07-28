# Installation

Stockula uses `uv` for fast Python package management and requires Python 3.13 or higher.

## Prerequisites

### System Requirements

- **Python**: 3.13 or higher
- **Operating System**: macOS, Linux, or Windows
- **Memory**: Minimum 8GB RAM recommended
- **Storage**: 100MB free space (more if caching extensive historical data)

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
   uv run python -m stockula.main --help
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

## Docker Installation (Alternative)

For containerized deployment:

```bash
# Build the container
docker build -t stockula .

# Run with volume mount for data persistence
docker run -v $(pwd)/data:/app/data stockula --ticker AAPL
```

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

=== "macOS"
\- Install via Homebrew: `brew install uv`
\- Xcode Command Line Tools may be required for some dependencies

=== "Linux"
\- Some distributions may require additional packages for compilation
\- Ubuntu/Debian: `sudo apt-get install build-essential python3-dev`

=== "Windows"
\- Windows Subsystem for Linux (WSL) is recommended
\- Ensure Visual Studio Build Tools are installed for compiled dependencies

## Next Steps

After installation, check out:

- [Quick Start Guide](../getting-started/quick-start.md)
- [Configuration Options](../getting-started/configuration.md)
- [Architecture Overview](../user-guide/architecture.md)
