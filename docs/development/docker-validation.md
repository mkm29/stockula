# Docker Validation

This project includes a comprehensive Docker validation script to ensure proper Docker setup and functionality.

## Usage

### Command Line Interface

The validation script is available as a Python script with a convenient `uv run` interface:

```bash
# Full validation (default) - includes image builds and functionality tests
uv run validate-docker

# Quick validation - only checks basic requirements
uv run validate-docker --quick

# Same as --quick
uv run validate-docker --check-only
```

### Direct Script Execution

You can also run the Python script directly:

```bash
python utils/validate_docker.py
python utils/validate_docker.py --quick
python utils/validate_docker.py --check-only
```

## Validation Checks

### Basic Checks (Always Run)

1. **Docker Installation**: Verifies Docker is installed and daemon is running
2. **Docker Compose**: Checks for Docker Compose availability (optional)
3. **Required Files**: Ensures `Dockerfile`, `pyproject.toml`, `uv.lock`, and `.dockerignore` exist

### Full Validation (Default Mode)

In addition to basic checks, full validation includes:

4. **Production Image Build**: Tests building the production target
5. **Test Image Build**: Tests building the test target
6. **Basic Functionality**: Verifies Stockula package imports correctly
7. **UV Functionality**: Ensures UV package manager works
8. **Python Version**: Confirms Python 3.13 is being used
9. **Security Check**: Verifies containers run as non-root user
10. **Volume Operations**: Tests Docker volume creation and removal
11. **Docker Compose**: Validates `docker-compose.yml` syntax (if present)
12. **Examples Directory**: Checks for examples directory

### Quick Mode

Use `--quick` or `--check-only` to skip image builds and functionality tests. This is useful for:

- Quick CI/CD pipeline checks
- Initial setup validation
- Troubleshooting basic Docker issues

## Output

The script provides clear status indicators:

- ✅ Success indicators for passing checks
- ❌ Error indicators for failing checks  
- ⚠️  Warning indicators for optional issues

Example output:

```
🐳 Validating Stockula Docker Setup
===================================
Checking Docker installation...
✅ Docker is installed and running

Checking Docker Compose...
✅ Docker Compose (v2) is available

Checking required files...
✅ Dockerfile exists
✅ pyproject.toml exists
✅ uv.lock exists
✅ .dockerignore exists

Testing Docker build (production target)...
✅ Production image builds successfully

Testing Docker build (test target)...
✅ Test image builds successfully

Testing basic functionality...
✅ Stockula package imports successfully

Testing uv functionality...
✅ UV is working: uv 0.4.29

Testing Python version...
✅ Python version: Python 3.13.3

Testing security (non-root user)...
✅ Running as non-root user (UID: 1000)

Testing volume functionality...
✅ Volume creation works

Testing Docker Compose configuration...
✅ docker-compose.yml syntax is valid

Testing examples directory...
✅ Examples directory exists with 3 Python files

Cleaning up test images...

🎉 Docker setup validation completed successfully!

Next steps:
1. Run 'make build' to build all images
2. Run 'make dev' to start development environment
3. Run 'make test' to run tests in Docker
4. See 'make help' for all available commands

For detailed usage, see docs/deployment/docker.md
```

## Error Handling

The script handles common issues gracefully:

- **Missing Docker**: Provides installation instructions
- **Docker Daemon Not Running**: Clear message to start Docker
- **Missing Files**: Identifies which required files are missing
- **Build Failures**: Displays build logs for troubleshooting
- **Test Failures**: Shows error messages for failed functionality tests

## Integration

### CI/CD Pipelines

Use the quick mode for fast CI/CD checks:

```yaml
- name: Validate Docker Setup
  run: uv run validate-docker --quick
```

### Development Workflow

Use full validation when setting up development environment:

```bash
# Initial setup
uv run validate-docker

# After Docker configuration changes
uv run validate-docker
```

### Troubleshooting

When Docker issues occur:

```bash
# Quick check of basics
uv run validate-docker --quick

# Full validation to identify specific issues
uv run validate-docker
```

## Dependencies

The script requires:

- Python 3.13+
- Docker installed and running
- Docker Compose (optional, warns if missing)

## Security Considerations

The validation script:

- ✅ Verifies containers run as non-root user
- ✅ Tests volume permissions and functionality
- ✅ Validates Docker Compose configuration syntax
- ✅ Cleans up test images and temporary files
- ✅ Uses minimal privileges for validation operations

## Performance

- **Quick Mode**: Completes in seconds
- **Full Mode**: Takes 1-3 minutes depending on image build time
- **Resource Usage**: Minimal - only creates temporary test images
- **Cleanup**: Automatically removes test images and temporary files

This validation script ensures your Docker setup is properly configured for the Stockula development and deployment workflow.