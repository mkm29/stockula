#!/usr/bin/env python
"""Verify Docker build configuration and requirements.

This utility checks that the Docker build files are properly configured,
requirements files exist, and no deprecated configurations are present.

Usage:
    uv run verify-build          # Verify build configuration
    uv run verify-build --gpu    # Verify GPU build specifically
"""

import argparse
import re
import sys
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists.

    Args:
        filepath: Path to the file
        description: Description for output

    Returns:
        True if file exists, False otherwise
    """
    if Path(filepath).exists():
        print(f"✓ {description} exists")
        return True
    else:
        print(f"✗ {description} not found")
        return False


def check_file_not_exists(filepath: str, description: str) -> bool:
    """Check that a file does not exist.

    Args:
        filepath: Path to the file
        description: Description for output

    Returns:
        True if file doesn't exist, False otherwise
    """
    if not Path(filepath).exists():
        print(f"✓ {description} has been removed")
        return True
    else:
        print(f"✗ {description} still exists (should be deleted)")
        return False


def check_pattern_in_file(filepath: str, pattern: str, should_exist: bool, description: str) -> bool:
    """Check if a pattern exists or doesn't exist in a file.

    Args:
        filepath: Path to the file
        pattern: Regex pattern to search for
        should_exist: True if pattern should exist, False if it shouldn't
        description: Description for output

    Returns:
        True if check passes, False otherwise
    """
    if not Path(filepath).exists():
        print(f"⚠️  {filepath} not found for pattern check")
        return False

    content = Path(filepath).read_text()
    pattern_found = bool(re.search(pattern, content, re.MULTILINE))

    if should_exist and pattern_found:
        print(f"✓ {description}")
        return True
    elif not should_exist and not pattern_found:
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description}")
        return False


def verify_gpu_build() -> tuple[bool, list[str]]:
    """Verify GPU Docker build configuration.

    Returns:
        Tuple of (success, list of error messages)
    """
    print("\nVerifying Docker GPU Build Configuration")
    print("=" * 50)

    errors = []

    # Check requirements file exists
    if not check_file_exists("requirements-gpu.txt", "requirements-gpu.txt"):
        errors.append("Missing requirements-gpu.txt")

    # Check Dockerfile.nvidia exists
    if not Path("Dockerfile.nvidia").exists():
        print("⚠️  Dockerfile.nvidia not found, skipping GPU-specific checks")
        return len(errors) == 0, errors

    # Check that we're not copying pyproject.toml in Dockerfile
    if not check_pattern_in_file(
        "Dockerfile.nvidia", r"COPY.*pyproject\.toml", False, "Dockerfile.nvidia does not copy pyproject.toml"
    ):
        errors.append("Dockerfile.nvidia still copies pyproject.toml")

    # Check that we're not using requirements-gpu-py313.txt
    if not check_file_not_exists("requirements-gpu-py313.txt", "requirements-gpu-py313.txt"):
        errors.append("requirements-gpu-py313.txt should be removed")

    # Check Python version in Dockerfile (PyTorch images include Python)
    # PyTorch 2.x images use Python 3.11+ by default
    if not check_pattern_in_file(
        "Dockerfile.nvidia",
        r"(Python 3\.1[12]|pytorch/pytorch:2\.|FROM.*python.*3\.1[12])",
        True,
        "Dockerfile uses compatible Python version (PyTorch 2.x includes Python 3.11+)",
    ):
        errors.append("Dockerfile not using compatible Python version")

    # Check that we're not using deadsnakes PPA
    if not check_pattern_in_file("Dockerfile.nvidia", r"deadsnakes", False, "Dockerfile does not use deadsnakes PPA"):
        errors.append("Dockerfile still uses deadsnakes PPA")

    return len(errors) == 0, errors


def verify_standard_build() -> tuple[bool, list[str]]:
    """Verify standard Docker build configuration.

    Returns:
        Tuple of (success, list of error messages)
    """
    print("\nVerifying Standard Docker Build Configuration")
    print("=" * 50)

    errors = []

    # Check Dockerfile exists
    if not check_file_exists("Dockerfile", "Dockerfile"):
        errors.append("Missing Dockerfile")
        return False, errors

    # Check pyproject.toml exists
    if not check_file_exists("pyproject.toml", "pyproject.toml"):
        errors.append("Missing pyproject.toml")

    # Check uv.lock exists
    if not check_file_exists("uv.lock", "uv.lock"):
        errors.append("Missing uv.lock")

    # Check .dockerignore exists
    if not check_file_exists(".dockerignore", ".dockerignore"):
        print("⚠️  .dockerignore not found (recommended but not required)")

    # Check Python version in Dockerfile (handles both python: and uv:python images)
    if not check_pattern_in_file(
        "Dockerfile",
        r"(FROM python:3\.1[123]|FROM.*uv.*python3\.1[123])",
        True,
        "Dockerfile uses Python 3.11, 3.12, or 3.13",
    ):
        errors.append("Dockerfile not using Python 3.11+")

    return len(errors) == 0, errors


def main() -> None:
    """Run build verification."""
    parser = argparse.ArgumentParser(description="Verify Docker build configuration")
    parser.add_argument("--gpu", action="store_true", help="Verify GPU build specifically")

    args = parser.parse_args()

    print("=" * 50)
    print("Docker Build Configuration Verification")
    print("=" * 50)

    if args.gpu:
        success, errors = verify_gpu_build()
    else:
        # Verify both standard and GPU builds
        std_success, std_errors = verify_standard_build()
        gpu_success, gpu_errors = verify_gpu_build()

        success = std_success and gpu_success
        errors = std_errors + gpu_errors

    print("\n" + "=" * 50)

    if success:
        print("✅ Configuration looks good! Ready to build:")

        if args.gpu or Path("Dockerfile.nvidia").exists():
            print("\nGPU build command:")
            print("  docker buildx build -t stockula:gpu \\")
            print("    -f Dockerfile.nvidia --platform linux/amd64 --target gpu-cli .")

        print("\nStandard build command:")
        print("  docker buildx build -t stockula:latest .")

        print("\nFor multi-platform builds:")
        print("  docker buildx build --platform linux/amd64,linux/arm64 \\")
        print("    -t stockula:latest .")
    else:
        print("❌ Build configuration has issues:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
