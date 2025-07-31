#!/bin/bash

# Migration script from bumpversion to release-please

echo "🚀 Migrating from bumpversion to release-please..."

# Remove bumpversion configuration
if [ -f ".bumpversion.cfg" ]; then
    echo "📦 Removing .bumpversion.cfg..."
    rm .bumpversion.cfg
fi

# Remove bumpversion from dev dependencies if present
echo "📝 Checking for bumpversion in dependencies..."
if grep -q "bumpversion" pyproject.toml; then
    echo "⚠️  Please manually remove bumpversion from pyproject.toml dev dependencies"
fi

echo "✅ Migration complete!"
echo ""
echo "📋 Next steps:"
echo "1. Remove bumpversion from your dev dependencies if present"
echo "2. Commit these changes with a conventional commit message:"
echo "   git add ."
echo "   git commit -m 'build: migrate from bumpversion to release-please'"
echo ""
echo "3. Push to main branch to trigger release-please"
echo ""
echo "📝 How to use release-please:"
echo "- Use conventional commits (feat:, fix:, docs:, etc.)"
echo "- Release Please will automatically create PRs for releases"
echo "- Merging the release PR will:"
echo "  - Update version in pyproject.toml and __init__.py"
echo "  - Update CHANGELOG.md"
echo "  - Create a GitHub release with tag"
echo "  - Trigger PyPI publication (if PYPI_API_TOKEN is set)"
echo ""
echo "🔧 To manually trigger a release:"
echo "- Make sure you have conventional commits since last release"
echo "- Push to main branch"
echo "- Release Please will create a PR within a few minutes"
