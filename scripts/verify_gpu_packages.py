#!/usr/bin/env python
"""Verify GPU package installation."""

import sys


def check_package(package_name, version_attr="__version__"):
    """Check if a package is installed and print its version."""
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

        return True
    except ImportError:
        print(f"✗ {package_name} not available")
        return False


def main():
    """Check all GPU packages."""
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

    available = 0
    for package in packages:
        if check_package(package):
            available += 1

    print("=" * 50)
    print(f"Summary: {available}/{len(packages)} packages available")
    print("=" * 50)

    # Always exit successfully
    sys.exit(0)


if __name__ == "__main__":
    main()
