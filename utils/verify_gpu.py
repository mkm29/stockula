#!/usr/bin/env python
"""Verify GPU package installation.

This utility checks if GPU-related packages are properly installed and configured,
including verifying CUDA availability and package versions.

Usage:
    uv run verify-gpu        # Check all GPU packages
"""

import sys


def check_package(package_name: str, version_attr: str = "__version__") -> bool:
    """Check if a package is installed and print its version.

    Args:
        package_name: Name of the package to check
        version_attr: Attribute name for version information

    Returns:
        True if package is available, False otherwise
    """
    try:
        module = __import__(package_name)
        version = getattr(module, version_attr, "unknown")
        print(f"✓ {package_name} installed: {version}")

        # Special checks for specific packages
        if package_name == "torch":
            import torch

            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if torch.version.cuda else "None"
            device_count = torch.cuda.device_count() if cuda_available else 0
            print(f"  - CUDA available: {cuda_available}")
            print(f"  - CUDA version: {cuda_version}")
            print(f"  - GPU devices: {device_count}")

            if cuda_available and device_count > 0:
                for i in range(device_count):
                    print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

        return True
    except ImportError:
        print(f"✗ {package_name} not available")
        return False


def main() -> None:
    """Check all GPU packages."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify GPU package installation and CUDA availability")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.parse_args()

    print("=" * 50)
    print("GPU Package Verification")
    print("=" * 50)

    packages = [
        "torch",
        "torchvision",
        "torchaudio",
        "xgboost",
        "lightgbm",
        "tensorflow",  # Expected to fail on Python 3.13
        "mxnet",  # Expected to fail on Python 3.13
        "gluonts",  # Expected to fail on Python 3.13
    ]

    print(f"\nChecking Python version: {sys.version}")
    print("-" * 50)

    available = 0
    for package in packages:
        if check_package(package):
            available += 1

    print("=" * 50)
    print(f"Summary: {available}/{len(packages)} packages available")

    if available == len(packages):
        print("✅ All GPU packages are installed!")
    elif available > 0:
        print("⚠️  Some GPU packages are missing (this may be expected)")
    else:
        print("❌ No GPU packages found")

    print("=" * 50)

    # Always exit successfully
    sys.exit(0)


if __name__ == "__main__":
    main()
