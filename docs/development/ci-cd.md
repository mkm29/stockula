# CI/CD Documentation

This document describes the continuous integration and deployment setup for Stockula.

## Overview

Stockula uses GitHub Actions for automated testing, building, and deployment. The CI/CD pipeline consists of three main workflows:

1. **Testing** - Automated code quality and test execution
1. **Release Management** - Automated versioning and PyPI publishing
1. **Docker Builds** - Multi-platform container image creation

## GitHub Actions Workflows

### Test Workflow (`test.yml`)

**Triggers:**

- Push to `main` branch
- All pull requests

**Jobs:**

#### 1. Linting

- **Purpose**: Ensure code quality and consistent formatting
- **Tools**: `ruff` for linting and formatting
- **Checks**:
  - Code style violations
  - Import sorting
  - Format consistency
- **Configuration**: Uses `pyproject.toml` for ruff settings
- **Local Script**: `scripts/lint.py` runs the same commands as CI for consistency

#### 2. Unit Tests

- **Purpose**: Test individual components in isolation
- **Framework**: pytest
- **Coverage**: Reports to Codecov with `unit` flag
- **Location**: `tests/unit/`

#### 3. Integration Tests (Currently Disabled)

- **Purpose**: Test component interactions with database
- **Database**: SQLite (will migrate to PostgreSQL)
- **Coverage**: Reports to Codecov with `integration` flag
- **Location**: `tests/integration/`
- **Note**: Temporarily disabled pending test infrastructure updates

### Release Please Workflow (`release-please.yml`)

**Triggers:**

- Push to `main` branch

**Features:**

- Monitors commits for Conventional Commits format
- Creates/updates release PRs automatically
- On PR merge:
  - Creates GitHub release
  - Tags version (e.g., `v0.1.0`)
  - Updates CHANGELOG.md
  - Publishes to PyPI

**Configuration:**

- `release-please-config.json`: Workflow configuration
- `.release-please-manifest.json`: Current version tracking
- `include-component-in-tag: false`: Simple version tags without project prefix

### Docker Build Workflow (`docker-build.yml`)

**Triggers:**

- Push of version tags (`v*`)
- Manual workflow dispatch (for testing)

**Features:**

- Multi-platform builds using buildx
- Platforms: `linux/amd64`, `linux/arm64/v8`
- Publishes to GitHub Container Registry (ghcr.io)
- Builds single image with CLI target (includes production + interactive tools)
- Automatic tagging:
  - Version tags: `v0.4.1`, `0.4.1`, `0.4`
  - Latest tag: `latest`
  - Branch/PR specific tags

**Optimizations:**

- **Layer Caching**: Uses both GitHub Actions cache and registry cache
- **BuildKit**: Latest BuildKit with performance optimizations
- **Registry Cache**: Stores build cache in ghcr.io for persistence
- **Optimized Dockerfile**:
  - Cache mounts for package managers
  - Separated dependency and source layers
  - Compiled Python bytecode during build
- **Reduced Context**: Comprehensive `.dockerignore` file

**Performance:**

- Initial build: ~30 minutes (multi-platform)
- Subsequent builds: ~5-10 minutes (with warm cache)
- Cache strategies:
  - GitHub Actions cache (ephemeral, fast)
  - Registry cache (persistent, shared across workflows)
  - Base image caching from `:latest` tag

## Linting Script

The `scripts` module provides a `lint` command for consistent linting checks between local development and CI:

### Usage

```bash
# Run the same checks as CI
uv run lint
```

### What it does

1. **Ruff Check**: Runs `uv run ruff check src tests`

   - Validates code style, imports, and common issues
   - Same scope as CI (only `src` and `tests` directories)

1. **Format Check**: Runs `uv run ruff format --check src tests`

   - Validates code formatting without making changes
   - Ensures consistent style across the codebase

### Output

- ✅ **Success**: All checks pass (same as CI)
- ❌ **Failure**: Shows specific issues and provides fix commands:
  ```bash
  🔧 To fix these issues, run:
    uv run ruff check src tests --fix
    uv run ruff format src tests
  ```

### Benefits

- **CI Consistency**: Runs identical commands to the CI pipeline
- **Easy Access**: Available as `uv run lint` from any directory in the project
- **Fast Feedback**: Catch issues locally before pushing
- **Clear Instructions**: Provides exact commands to fix issues
- **Module Integration**: Part of the project's Python package structure

## Development Workflow

### 1. Local Development

```bash
# Install dependencies
uv sync --all-extras --dev

# Run tests locally
uv run pytest

# Check code style (consistent with CI)
uv run lint

# Or run individual commands
uv run ruff check src tests
uv run ruff format --check src tests

# Fix linting issues
uv run ruff check src tests --fix
uv run ruff format src tests
```

### 2. Creating a Pull Request

1. Create feature branch: `git checkout -b feat/my-feature`
1. Make changes following Conventional Commits:
   ```bash
   git commit -m "feat: add new trading strategy"
   git commit -m "fix: correct calculation in backtest"
   git commit -m "chore: update dependencies"
   ```
1. Push branch and create PR
1. CI will automatically run tests and linting

### 3. Release Process

1. **Automatic PR Creation**: Release Please monitors `main` and creates release PRs
1. **Review Release PR**: Check CHANGELOG.md and version bump
1. **Merge Release PR**: Triggers:
   - GitHub release creation
   - PyPI package publishing
   - Docker image building

## Secrets and Environment Variables

### Required Secrets

1. **`RELEASE_PLEASE_TOKEN`**

   - Personal Access Token with `repo` and `workflow` scopes
   - Used by release-please to create PRs and releases

1. **`PYPI_API_TOKEN`**

   - PyPI API token for package publishing
   - Used in release workflow

1. **`CODECOV_TOKEN`**

   - Required for uploading coverage reports to Codecov
   - Get token from: https://app.codecov.io/gh/mkm29/stockula/settings
   - Used in test workflow for coverage reporting

### Optional Secrets

1. **`POSTGRES_TEST_PASSWORD`**
   - For future PostgreSQL integration tests
   - Falls back to `test_password_only` if not set

## Conventional Commits

All commits should follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or modifications
- `build`: Build system changes
- `ci`: CI configuration changes
- `chore`: Other changes (dependencies, etc.)

### Examples

```bash
feat: add momentum trading strategy
fix: correct profit calculation in backtester
docs: update API documentation for forecasting
chore: bump pandas to 2.2.0
```

## Troubleshooting

### Common Issues

1. **Release Please not creating releases**

   - Ensure commits follow Conventional Commits format
   - Check that `RELEASE_PLEASE_TOKEN` has correct permissions
   - Verify no open release PRs exist

1. **Docker build failures**

   - Check multi-platform compatibility
   - Ensure base images support all target architectures
   - Review build logs for architecture-specific issues
   - Clear caches if encountering stale cache issues:
     ```bash
     # Clear GitHub Actions cache (automatic on workflow update)
     # Clear registry cache by pushing new buildcache tag
     ```

1. **Test failures in CI but not locally**

   - Check for environment-specific dependencies
   - Verify database connections and migrations
   - Review CI environment variables

### Debugging Workflows

1. **Enable debug logging**:

   - Add `ACTIONS_STEP_DEBUG: true` to repository secrets
   - Provides detailed logs for troubleshooting

1. **Run workflows manually**:

   - Use `workflow_dispatch` trigger for testing
   - Helpful for debugging without code changes

## Best Practices

1. **Keep workflows fast**:

   - Use caching for dependencies (uv cache)
   - Run tests in parallel where possible
   - Minimize unnecessary steps

1. **Fail fast**:

   - Order jobs from fastest to slowest
   - Run linting before tests
   - Use job dependencies wisely

1. **Security**:

   - Use least-privilege principle for tokens
   - Rotate secrets regularly
   - Avoid hardcoding sensitive data

1. **Monitoring**:

   - Set up notifications for workflow failures
   - Monitor workflow run times
   - Track test coverage trends

## Future Improvements

1. **PostgreSQL Migration**

   - Enable integration tests with PostgreSQL
   - Add database migration testing
   - Performance benchmarking

1. **Advanced Testing**

   - Add performance regression tests
   - Implement visual regression for charts
   - Add security scanning (SAST/DAST)

1. **Deployment Automation**

   - Add staging environment deployment
   - Implement blue-green deployments
   - Add automated rollback capabilities
