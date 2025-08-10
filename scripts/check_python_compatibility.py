#!/usr/bin/env python3
"""Check if we can relax Python version requirement for Docker builds."""

import sys


def check_python_version():
    """Check current Python version."""
    version = sys.version_info
    print(f"Current Python: {version.major}.{version.minor}.{version.micro}")
    return version


def check_package_compatibility(python_version):
    """Check if key packages support the Python version."""
    packages = {
        "3.12": {
            "torch": "✓ Full support",
            "tensorflow": "✓ Full support",
            "autots": "✓ Full support",
            "xgboost": "✓ Full support",
            "lightgbm": "✓ Full support",
            "mxnet": "✓ Full support",
            "gluonts": "✓ Full support",
        },
        "3.13": {
            "torch": "✓ Full support",
            "tensorflow": "✗ Not yet available",
            "autots": "✓ Should work",
            "xgboost": "✓ Full support",
            "lightgbm": "✓ Full support",
            "mxnet": "✗ Not yet available",
            "gluonts": "✗ Depends on mxnet",
        },
    }

    version_str = f"{python_version.major}.{python_version.minor}"
    if version_str in packages:
        print(f"\nPackage compatibility for Python {version_str}:")
        for pkg, status in packages[version_str].items():
            print(f"  {pkg:15} {status}")
    else:
        print(f"Unknown Python version: {version_str}")


def suggest_dockerfile():
    """Suggest which Dockerfile to use."""
    print("\n" + "=" * 60)
    print("Dockerfile Recommendations:")
    print("=" * 60)

    print("\n1. For Python 3.13 (current requirement):")
    print("   - Try: Dockerfile.nvidia (updated with fixes)")
    print("   - Fallback: Dockerfile.nvidia.simple (minimal complexity)")
    print("   - Last resort: Dockerfile.nvidia.robust (builds from source)")

    print("\n2. If Python 3.13 fails completely:")
    print("   - Modify pyproject.toml: requires-python = '>=3.12'")
    print("   - Use Dockerfile.nvidia.alt (Python 3.12 based)")
    print("   - Benefits: All GPU packages available")

    print("\n3. Quick test command:")
    print("   docker run --rm nvidia/cuda:13.0.0-devel-ubuntu24.04 python3 --version")


if __name__ == "__main__":
    version = check_python_version()
    check_package_compatibility(version)
    suggest_dockerfile()
