#!/usr/bin/env python3
"""Check Python version compatibility for the project.

This utility verifies the current Python version and checks package compatibility,
providing recommendations for Docker builds and configuration.

Usage:
    uv run check-python      # Check Python version and package compatibility
"""

import sys


def check_python_version() -> tuple[int, int, int]:
    """Check current Python version.

    Returns:
        Tuple of (major, minor, micro) version numbers
    """
    version = sys.version_info
    print(f"Current Python: {version.major}.{version.minor}.{version.micro}")
    print(f"Full version: {sys.version}")
    return version.major, version.minor, version.micro


def check_package_compatibility(python_version: tuple[int, int, int]) -> None:
    """Check if key packages support the Python version.

    Args:
        python_version: Tuple of (major, minor, micro) version numbers
    """
    packages: dict[str, dict[str, str]] = {
        "3.11": {
            "torch": "✓ Full support",
            "tensorflow": "✓ Full support (2.15+)",
            "autogluon": "✓ Full support",
            "xgboost": "✓ Full support",
            "lightgbm": "✓ Full support",
            "mxnet": "✓ Full support",
            "gluonts": "✓ Full support",
        },
        "3.12": {
            "torch": "✓ Full support",
            "tensorflow": "✓ Full support (2.16+)",
            "autogluon": "✓ Full support",
            "xgboost": "✓ Full support",
            "lightgbm": "✓ Full support",
            "mxnet": "✓ Full support",
            "gluonts": "✓ Full support",
        },
        "3.13": {
            "torch": "✓ Full support",
            "tensorflow": "✗ Not yet available",
            "autogluon": "✗ Ray dependency issue",
            "xgboost": "✓ Full support",
            "lightgbm": "✓ Full support",
            "mxnet": "✗ Not yet available",
            "gluonts": "✗ Depends on mxnet",
        },
    }

    version_str = f"{python_version[0]}.{python_version[1]}"
    if version_str in packages:
        print(f"\nPackage compatibility for Python {version_str}:")
        print("-" * 50)
        for pkg, status in packages[version_str].items():
            print(f"  {pkg:15} {status}")
    else:
        print(f"⚠️  Unknown Python version: {version_str}")
        print("This project requires Python 3.11+")


def check_installed_packages() -> None:
    """Check which packages are actually installed."""
    print("\nInstalled package check:")
    print("-" * 50)

    packages_to_check = ["torch", "tensorflow", "autogluon.timeseries", "xgboost", "lightgbm", "mxnet", "gluonts"]

    for package in packages_to_check:
        package_import = package.replace(".", "_").replace("-", "_")
        try:
            if package == "autogluon.timeseries":
                import autogluon.timeseries

                version = autogluon.timeseries.__version__
            else:
                module = __import__(package_import)
                version = getattr(module, "__version__", "unknown")
            print(f"  ✓ {package:20} {version}")
        except ImportError:
            print(f"  ✗ {package:20} not installed")


def suggest_dockerfile() -> None:
    """Suggest which Dockerfile to use based on Python version."""
    print("\n" + "=" * 60)
    print("Dockerfile Recommendations:")
    print("=" * 60)

    version = sys.version_info

    if version.major == 3 and version.minor == 13:
        print("\n⚠️  Python 3.13 detected - Limited package support")
        print("\nOptions:")
        print("1. Use Dockerfile with Python 3.13 (no AutoGluon):")
        print("   - Basic functionality only")
        print("   - GPU support for torch, xgboost, lightgbm")
        print("\n2. Switch to Python 3.12 for full support:")
        print("   - Modify pyproject.toml if needed")
        print("   - All packages available including AutoGluon")

    elif version.major == 3 and version.minor == 12:
        print("\n✅ Python 3.12 - Full package support available")
        print("\nRecommended Dockerfiles:")
        print("   - Dockerfile.nvidia (GPU support)")
        print("   - Dockerfile (CPU only)")

    elif version.major == 3 and version.minor == 11:
        print("\n✅ Python 3.11 - Full package support available")
        print("\nRecommended Dockerfiles:")
        print("   - Dockerfile.nvidia (GPU support)")
        print("   - Dockerfile (CPU only)")

    else:
        print(f"\n❌ Python {version.major}.{version.minor} is not supported")
        print("This project requires Python 3.11 or higher")

    print("\nQuick test commands:")
    print("  docker build -t stockula:test .")
    print("  docker run --rm stockula:test python -c 'import torch; print(torch.__version__)'")


def main() -> None:
    """Run all compatibility checks."""
    import argparse

    parser = argparse.ArgumentParser(description="Check Python version compatibility for the project")
    parser.add_argument("--packages-only", action="store_true", help="Only check installed packages")
    parser.add_argument("--docker-only", action="store_true", help="Only show Docker recommendations")
    parser.parse_args()

    print("=" * 60)
    print("Python Compatibility Check")
    print("=" * 60)

    version = check_python_version()
    check_package_compatibility(version)
    check_installed_packages()
    suggest_dockerfile()

    print("\n" + "=" * 60)
    print("Check complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
