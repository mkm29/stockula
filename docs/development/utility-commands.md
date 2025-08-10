# Utility Commands

This document describes the utility commands available via `uv run` for development and maintenance tasks.

## Available Commands

All commands can be run using `uv run <command-name>` or directly as Python scripts from the `utils/` directory.

### Code Quality

#### `lint`

Check and optionally fix code quality issues using Ruff.

```bash
# Check for issues only
uv run lint

# Automatically fix issues
uv run lint --fix
```

#### `format-yaml`

Format YAML files consistently across the project.

```bash
uv run format-yaml
```

### Docker & Deployment

#### `validate-docker`

Comprehensive Docker setup validation including daemon, compose, builds, and functionality.

```bash
# Full validation with builds
uv run validate-docker

# Quick validation (skip builds)
uv run validate-docker --quick
```

#### `docker-verify`

Verify Docker build and GPU support with container testing.

```bash
# Verify standard Docker build
uv run docker-verify

# Verify GPU Docker build
uv run docker-verify --gpu

# Quick validation only
uv run docker-verify --quick
```

#### `verify-build`

Check Docker build configuration and requirements files.

```bash
# Verify all build configurations
uv run verify-build

# Verify GPU build specifically
uv run verify-build --gpu
```

### Environment & Dependencies

#### `check-python`

Check Python version compatibility and package support.

```bash
uv run check-python
```

Shows:

- Current Python version
- Package compatibility matrix
- Installed packages check
- Dockerfile recommendations

#### `verify-gpu`

Verify GPU package installation and CUDA availability.

```bash
uv run verify-gpu
```

Shows:

- GPU package installation status
- CUDA availability and version
- GPU device information (if available)
- Summary of available packages

## Implementation Details

All utility commands are implemented as Python scripts in the `utils/` directory:

- `utils/lint.py` - Linting and code formatting
- `utils/format_yaml.py` - YAML formatting
- `utils/validate_docker.py` - Docker validation
- `utils/verify_docker.py` - Docker verification and testing
- `utils/verify_build.py` - Build configuration verification
- `utils/check_python.py` - Python compatibility checking
- `utils/verify_gpu.py` - GPU package verification

### Adding New Commands

To add a new utility command:

1. Create a Python script in `utils/` with a `main()` function

1. Add an entry to `[project.scripts]` in `pyproject.toml`:

   ```toml
   [project.scripts]
   your-command = "utils.your_module:main"
   ```

1. Ensure the script includes proper docstrings and help text

1. Test with `uv run your-command`

### Common Patterns

All utility scripts follow these patterns:

```python
#!/usr/bin/env python
"""Brief description of the utility.

Detailed description of what the utility does and its purpose.

Usage:
    uv run command-name [options]
"""

import argparse
import sys


def main() -> None:
    """Main entry point for the command."""
    parser = argparse.ArgumentParser(description="Command description")
    parser.add_argument("--option", help="Option description")

    args = parser.parse_args()

    # Implementation here

    # Exit with appropriate code
    sys.exit(0)  # Success
    # or
    sys.exit(1)  # Failure


if __name__ == "__main__":
    main()
```

## Remaining Shell Scripts

While most utilities have been migrated to Python, some shell scripts remain in `scripts/` for specific purposes:

### Testing Scripts

- **`scripts/test.sh`** - Runs pytest with pkg_resources warnings suppressed (workaround for transitive dependency
  warnings)
- **`scripts/test_docker_build.sh`** - Tests Docker build stages
- **`scripts/test_docker_python.sh`** - Verifies Python version in Docker images

### Information Scripts

- **`scripts/gpu_info.sh`** - GPU information script (copied into Docker images)
- **`scripts/check_ubuntu_python.sh`** - Checks Python availability in Ubuntu

### Data Processing Scripts

- **`scripts/chronos_batch_infer.py`** - Batch inference with Chronos models
- **`scripts/export_to_gluonts_file_dataset.py`** - Export data for GluonTS

## Migration from Shell Scripts

These utility commands replace the following shell scripts that were previously in `scripts/`:

- `scripts/verify_gpu_packages.py` → `uv run verify-gpu`
- `scripts/check_python_compatibility.py` → `uv run check-python`
- `scripts/verify_docker_build.sh` → `uv run verify-build`

The Python implementations provide:

- Better cross-platform compatibility
- Consistent error handling
- Integrated with the project's dependency management
- Type hints and documentation
- Easier testing and maintenance
