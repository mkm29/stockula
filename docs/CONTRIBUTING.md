# Contributing to Stockula

Thank you for your interest in contributing to Stockula! This document outlines our development workflow and contribution guidelines.

## 🌳 Branching Strategy (Git Flow)

We follow a modified Git Flow branching strategy:

### Protected Branches

- **`main`** - Production-ready code. All releases are tagged from here.
- **`develop`** - Integration branch for features. This is where active development happens.

### Branch Types

#### Feature Branches (`feature/*`)

- Created from: `develop`
- Merge back to: `develop`
- Naming: `feature/description-of-feature`
- Example: `feature/add-portfolio-optimization`

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

#### Bugfix Branches (`bugfix/*` or `fix/*`)

- Created from: `develop`
- Merge back to: `develop`
- Naming: `bugfix/description-of-bug`
- Example: `bugfix/fix-data-fetcher-timeout`

```bash
git checkout develop
git pull origin develop
git checkout -b bugfix/your-bugfix-name
```

#### Hotfix Branches (`hotfix/*`)

- Created from: `main`
- Merge back to: `main` (then automatically backported to `develop`)
- Naming: `hotfix/critical-issue`
- Example: `hotfix/security-patch`
- Use for critical production issues only

```bash
git checkout main
git pull origin main
git checkout -b hotfix/your-hotfix-name
```

#### Release Branches (`release/*`) - Optional

- Created from: `develop`
- Merge back to: `main` and `develop`
- Naming: `release/version-number`
- Example: `release/1.2.0`

### Other Branch Prefixes

- `chore/*` - Maintenance tasks, dependency updates
- `docs/*` - Documentation updates
- `test/*` - Test additions or modifications
- `refactor/*` - Code refactoring without changing functionality

## 📋 Development Workflow

1. **Start from the correct branch**

   ```bash
   # For features and regular bugs
   git checkout develop
   git pull origin develop

   # For hotfixes only
   git checkout main
   git pull origin main
   ```

1. **Create your branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

1. **Make your changes**

   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

1. **Run tests locally**

   ```bash
   # Run linting
   uv run ruff check src tests
   uv run ruff format src tests

   # Run tests
   uv run pytest tests/unit
   ```

1. **Commit your changes**

   - Use conventional commit messages (see below)
   - Keep commits focused and atomic

1. **Push and create PR**

   ```bash
   git push origin feature/your-feature-name
   ```

   Then create a PR on GitHub targeting the appropriate branch.

## 📝 Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or modifications
- `chore`: Maintenance tasks
- `perf`: Performance improvements

### Examples

```
feat(portfolio): add portfolio rebalancing algorithm

fix(data): handle missing dividend data gracefully

docs: update installation instructions

chore: upgrade pandas to 2.3.2
```

### Breaking Changes

For breaking changes, add `!` after the type or include `BREAKING CHANGE:` in the footer:

```
feat!: change API response format

feat(api): add new endpoint

BREAKING CHANGE: The /portfolio endpoint now returns a different structure
```

## 🏷️ Versioning

### Version Format

We use Semantic Versioning (SemVer): `MAJOR.MINOR.PATCH`

### Automatic Versioning with Release Please

[Release Please](https://github.com/googleapis/release-please) automates our release process:

- **Commits to `develop`**: Release Please creates/updates a PR for the next pre-release
- **Commits to `main`**: Release Please creates/updates a PR for the next stable release
- **When Release PR is merged**: Automatically creates GitHub release and tags

### Version Bumps

Version bumps are determined by conventional commit messages:

- **Major** (`1.0.0` → `2.0.0`): Commits with `BREAKING CHANGE:` in footer or `!` after type
- **Minor** (`0.12.0` → `0.13.0`): Commits with `feat:` type
- **Patch** (`0.12.1` → `0.12.2`): Commits with `fix:` type

Other commit types (`chore:`, `docs:`, `style:`, `refactor:`, `test:`) don't trigger version bumps but are included in the changelog.

## 🏷️ PR Labels

Labels are automatically added based on branch names:

- `feature` - Feature branches
- `bugfix` - Bugfix branches
- `hotfix` - Hotfix branches (also adds `urgent`)
- `documentation` - Documentation updates
- `chore` - Maintenance tasks
- `test` - Test updates
- `refactor` - Code refactoring

You can also manually add:

- `breaking-change` - For breaking changes (triggers major version bump)
- `enhancement` - For new features (triggers minor version bump)

## ✅ Pull Request Checklist

Before submitting a PR:

- [ ] Branch follows naming convention (`feature/*`, `bugfix/*`, etc.)
- [ ] Targeting correct branch (`develop` for features, `main` for hotfixes)
- [ ] Tests pass locally
- [ ] Code follows project style (ruff)
- [ ] Documentation updated if needed
- [ ] Commit messages follow convention
- [ ] PR description clearly explains changes

## 🚀 Release Process

### How Release Please Works

1. **Development Phase** (`develop` branch)

   - Developers merge feature/bugfix PRs to `develop`
   - Release Please automatically creates/updates a PR titled "chore(develop): release X.Y.Z-rc.N"
   - This PR accumulates all changes since last release
   - Merging this PR:
     - Creates a release candidate tag (e.g., `0.12.1-rc.1`)
     - Triggers Docker image build with RC tag
     - Images pushed to `ghcr.io/mkm29/stockula:0.12.1-rc.1`

1. **Production Release** (`main` branch)

   - Create PR from `develop` to `main` when ready for stable release
   - After merge, Release Please creates a PR titled "chore(main): release X.Y.Z"
   - Review the auto-generated changelog in the PR
   - Merging this PR:
     - Creates a GitHub Release with changelog
     - Tags the release as `vX.Y.Z`
     - Publishes to PyPI
     - Triggers Docker image build with stable tag
     - Images pushed to `ghcr.io/mkm29/stockula:vX.Y.Z` and `:latest`

1. **Hotfixes** (directly to `main`)

   - Create `hotfix/*` branch from `main`
   - Fix the issue with proper commit message (e.g., `fix: critical security issue`)
   - Create PR to `main`
   - After merge:
     - Release Please creates a patch release PR
     - Automatic backport PR to `develop` is created

## 🔍 Code Review

All PRs require review before merging:

- At least one approval required
- CI checks must pass
- No merge conflicts
- Branch protection rules enforced

## 💡 Tips

- Keep PRs focused and reasonable in size
- Update your branch with the latest changes before creating PR:
  ```bash
  git checkout develop
  git pull origin develop
  git checkout feature/your-feature
  git rebase develop
  ```
- Use draft PRs for work in progress
- Link related issues in PR description
- Be responsive to review feedback

## 🐳 Docker Images

### Available Images

Docker images are automatically built and pushed to GitHub Container Registry (GHCR):

| Image                        | Description                | Tags                                                     |
| ---------------------------- | -------------------------- | -------------------------------------------------------- |
| `ghcr.io/mkm29/stockula`     | CLI with development tools | `latest`, `vX.Y.Z`, `0.Y.Z-rc.N`, `X.Y.Z-feat.*`, `feat` |
| `ghcr.io/mkm29/stockula-gpu` | GPU-accelerated CLI        | `latest`, `vX.Y.Z`, `0.Y.Z-rc.N`, `X.Y.Z-feat.*`, `feat` |

### Image Tagging Strategy

Docker tags match Git tags for releases, with special formatting for feature branches:

- **Feature Branches**: `X.Y.Z-feat.<branch-name>.<short-sha>`

  - Built automatically when pushing to `feature/*` or `feat/*` branches
  - No Git tags created (Docker images only)
  - Also tagged as `:feat` for latest feature build
  - Example: `0.12.1-feat.new-api.a1b2c3d`

- **Release Candidates**: `0.12.1-rc.1`, `0.12.1-rc.2`, etc.

  - Built when Release Please PR is merged on `develop`
  - Git tag and Docker tag match exactly
  - Also tagged as `:rc` for latest RC

- **Stable Releases**: `v0.12.1`, `v0.13.0`, etc.

  - Built when Release Please PR is merged on `main`
  - Git tag and Docker tag match exactly
  - Also tagged as `:latest` for latest stable

- **Development**: `:dev`

  - Can be built manually via workflow dispatch

### Using the Images

```bash
# Pull latest stable
docker pull ghcr.io/mkm29/stockula:latest

# Pull specific versions (Docker tags match Git tags for releases)
docker pull ghcr.io/mkm29/stockula:v0.12.1      # Stable release
docker pull ghcr.io/mkm29/stockula:0.12.1-rc.1  # Release candidate

# Pull latest RC
docker pull ghcr.io/mkm29/stockula:rc

# Pull latest feature branch build
docker pull ghcr.io/mkm29/stockula:feat

# Pull specific feature branch build
docker pull ghcr.io/mkm29/stockula:0.12.1-feat.new-api.a1b2c3d

# Run with GPU support
docker run --gpus all ghcr.io/mkm29/stockula-gpu:latest
```

## 📚 Resources

- [Git Flow Explanation](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [Release Please](https://github.com/googleapis/release-please)

## ❓ Questions?

If you have questions about contributing, feel free to:

- Open a discussion on GitHub
- Check existing issues and PRs
- Review the documentation

Thank you for contributing to Stockula! 🚀
