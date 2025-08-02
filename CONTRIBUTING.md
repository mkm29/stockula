# Contributing to Stockula

## Commit Message Format

This project uses [Conventional Commits](https://www.conventionalcommits.org/) for automated releases with Release Please.

### Commit Message Structure

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- **feat**: A new feature (triggers minor version bump)
- **fix**: A bug fix (triggers patch version bump)
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to our CI configuration files and scripts
- **chore**: Other changes that don't modify src or test files
- **revert**: Reverts a previous commit

### Breaking Changes

To trigger a major version bump, include `BREAKING CHANGE:` in the commit footer or append `!` after the type:

```
feat!: remove deprecated API endpoints

BREAKING CHANGE: The v1 API endpoints have been removed.
```

### Examples

```bash
# Feature
feat: add train/test split for backtesting

# Bug fix
fix: handle None values in backtest date_range

# Documentation
docs: update backtesting guide with train/test split examples

# Performance improvement
perf: optimize portfolio calculation with vectorized operations

# Breaking change
feat!: change default broker configuration format

BREAKING CHANGE: Broker configuration now requires explicit fee structure definition
```

## Release Process

1. Make commits using conventional commit format
1. Push to main branch
1. Release Please will automatically create/update a PR
1. Review and merge the Release PR to:
   - Update version numbers
   - Update CHANGELOG.md
   - Create GitHub release and tag
   - Publish to PyPI (if configured)

## Development Setup

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
1. Install dependencies: `uv sync`
1. Run tests: `uv run pytest`
1. Run linting: `uv run lint` (consistent with CI)
1. Fix linting issues:
   - `uv run ruff check src tests --fix`
   - `uv run ruff format src tests`
