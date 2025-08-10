#!/usr/bin/env python
"""Markdown formatting and linting utility for the project.

This script provides automated Markdown formatting using markdownlint,
which uses the existing .markdownlint.json configuration.

Usage:
    uv run format-md                    # Format all Markdown files
    uv run format-md --check            # Check formatting only, don't modify
    uv run format-md README.md          # Format specific files
    uv run format-md docs/              # Format all files in directory

    # Or directly:
    python utils/format_markdown.py
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and optionally exit on failure."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if check and result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        sys.exit(result.returncode)

    return result


def check_markdown_tool() -> bool:
    """Check if markdownlint is installed."""
    if not shutil.which("markdownlint"):
        print("❌ markdownlint is not installed!")
        print("\nTo install:")
        print("  npm install -g markdownlint-cli")
        print("  # or")
        print("  brew install markdownlint-cli")
        return False
    return True


def get_markdown_files(paths: list[str] | None = None) -> list[Path]:
    """Get all Markdown files to format."""
    if not paths:
        # Default: format all .md files in the project
        paths = ["."]

    md_files = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_file():
            if path.suffix.lower() in {".md", ".markdown"}:
                md_files.append(path)
        elif path.is_dir():
            # Recursively find all .md files
            md_files.extend(path.glob("**/*.md"))
            md_files.extend(path.glob("**/*.markdown"))

    # Filter out common directories to ignore
    ignore_dirs = {".venv", "venv", "node_modules", ".git", "dist", "build", ".tox"}
    md_files = [f for f in md_files if not any(ignored in f.parts for ignored in ignore_dirs)]

    return sorted(set(md_files))


def format_markdown(files: list[Path], check_only: bool = False) -> bool:
    """Format Markdown files using markdownlint."""
    if not files:
        print("No Markdown files found to format.")
        return True

    print(f"\n{'🔍 Checking' if check_only else '🔧 Formatting'} {len(files)} Markdown file(s)...")

    # Build markdownlint command
    cmd = ["markdownlint"]

    # Use config file if it exists
    config_file = Path(".markdownlint.json")
    if config_file.exists():
        cmd.extend(["--config", str(config_file)])

    if not check_only:
        cmd.append("--fix")

    # Add file paths
    cmd.extend(str(f) for f in files)

    # Run markdownlint
    result = run_command(cmd, check=False)

    if result.returncode != 0:
        if check_only:
            print("❌ Some files need formatting")
            if result.stdout:
                print(result.stdout)
            print("\n🔧 To fix formatting issues, run:")
            print("  uv run format-md")
        else:
            print("❌ Formatting failed")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        return False

    if check_only:
        print("✅ All Markdown files are properly formatted!")
    else:
        print("✅ Markdown formatting complete!")

    return True


def main() -> None:
    """Main entry point for the formatter."""
    parser = argparse.ArgumentParser(description="Format and lint Markdown files with markdownlint")
    parser.add_argument("paths", nargs="*", help="Files or directories to format (default: all .md files)")
    parser.add_argument("--check", action="store_true", help="Check formatting only, don't modify files")

    args = parser.parse_args()

    # Check if markdownlint is installed
    if not check_markdown_tool():
        sys.exit(1)

    # Get files to format
    md_files = get_markdown_files(args.paths)

    if not md_files:
        print("No Markdown files found.")
        sys.exit(0)

    # Format the files
    success = format_markdown(md_files, check_only=args.check)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
