#!/usr/bin/env python
"""Docker validation utilities for the project.

This script provides comprehensive Docker setup validation including Docker daemon,
Docker Compose, required files, image builds, and basic functionality testing.

Usage:
    python scripts/validate_docker.py           # Full validation (default)
    python scripts/validate_docker.py --quick   # Quick validation (skip builds)
    python scripts/validate_docker.py --check-only  # Same as --quick

    # Or using uv:
    uv run validate-docker                       # Full validation
    uv run validate-docker --quick              # Quick validation
    uv run validate-docker --check-only         # Quick validation

The script validates Docker setup, builds test images, and verifies basic functionality,
providing comprehensive validation for the Docker-based development workflow.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_command(cmd: str, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and optionally exit on failure."""
    if not capture_output:
        print(f"Running: {cmd}")

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=capture_output,
        text=True,
        stdout=subprocess.DEVNULL if not capture_output else subprocess.PIPE,
        stderr=subprocess.DEVNULL if not capture_output else subprocess.PIPE,
    )

    if check and result.returncode != 0:
        if capture_output:
            print(f"Command failed: {cmd}")
            if result.stdout:
                print("stdout:", result.stdout)
            if result.stderr:
                print("stderr:", result.stderr)
        sys.exit(result.returncode)

    return result


def print_status(success: bool, message: str) -> bool:
    """Print status message with appropriate emoji."""
    if success:
        print(f"✅ {message}")
        return True
    else:
        print(f"❌ {message}")
        return False


def print_warning(message: str):
    """Print warning message."""
    print(f"⚠️  {message}")


def check_docker_installation() -> bool:
    """Check if Docker is installed and running."""
    print("Checking Docker installation...")

    # Check if docker command exists
    if not shutil.which("docker"):
        print_status(False, "Docker is not installed")
        print("Please install Docker from https://docs.docker.com/get-docker/")
        return False

    # Check if Docker daemon is running
    result = run_command("docker info", check=False)
    if result.returncode != 0:
        print_status(False, "Docker daemon is not running")
        print("Please start Docker daemon")
        return False

    print_status(True, "Docker is installed and running")
    return True


def check_docker_compose() -> bool:
    """Check Docker Compose availability."""
    print("\nChecking Docker Compose...")

    # Check for docker-compose command
    if shutil.which("docker-compose"):
        print_status(True, "Docker Compose is available")
        return True

    # Check for docker compose (v2)
    result = run_command("docker compose version", check=False)
    if result.returncode == 0:
        print_status(True, "Docker Compose (v2) is available")
        return True

    print_warning("Docker Compose not found, but continuing as it's optional")
    return False


def check_required_files() -> bool:
    """Check if required files exist."""
    print("\nChecking required files...")
    required_files = ["Dockerfile", "pyproject.toml", "uv.lock", ".dockerignore"]

    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print_status(True, f"{file} exists")
        else:
            print_status(False, f"{file} missing")
            all_exist = False

    return all_exist


def test_docker_build(target: str, tag: str) -> bool:
    """Test Docker build for a specific target."""
    print(f"\nTesting Docker build ({target} target)...")

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False) as log_file:
        log_path = log_file.name

    try:
        cmd = f"docker build --target {target} -t {tag} ."
        result = run_command(cmd, check=False)

        if result.returncode == 0:
            print_status(True, f"{target.title()} image builds successfully")
            return True
        else:
            print_status(False, f"{target.title()} image build failed")
            if result.stdout:
                print("Build stdout:")
                print(result.stdout)
            if result.stderr:
                print("Build stderr:")
                print(result.stderr)
            return False
    finally:
        # Clean up log file
        try:
            os.unlink(log_path)
        except OSError:
            pass


def test_basic_functionality(image_tag: str) -> bool:
    """Test basic functionality of the built image."""
    print("\nTesting basic functionality...")

    # Test Stockula import
    cmd = f"docker run --rm {image_tag} python -c \"import stockula; print('Stockula imports successfully')\""
    result = run_command(cmd, check=False)

    if result.returncode == 0:
        print_status(True, "Stockula package imports successfully")
    else:
        print_status(False, "Stockula package import failed")
        if result.stderr:
            print("Error:", result.stderr)
        return False

    return True


def test_uv_functionality(image_tag: str) -> bool:
    """Test UV functionality in the image."""
    print("\nTesting uv functionality...")

    result = run_command(f"docker run --rm {image_tag} uv --version", check=False)

    if result.returncode == 0:
        uv_version = result.stdout.strip()
        print_status(True, f"UV is working: {uv_version}")
        return True
    else:
        print_status(False, "UV command failed")
        if result.stderr:
            print("Error:", result.stderr)
        return False


def test_python_version(image_tag: str) -> bool:
    """Test Python version in the image."""
    print("\nTesting Python version...")

    result = run_command(f"docker run --rm {image_tag} python --version", check=False)

    if result.returncode == 0:
        python_version = result.stdout.strip()
        if "3.13" in python_version:
            print_status(True, f"Python version: {python_version}")
            return True
        else:
            print_warning(f"Expected Python 3.13, got: {python_version}")
            return True  # Warning, not failure
    else:
        print_status(False, "Failed to get Python version")
        return False


def test_security(image_tag: str) -> bool:
    """Test security (non-root user)."""
    print("\nTesting security (non-root user)...")

    result = run_command(f"docker run --rm {image_tag} id -u", check=False)

    if result.returncode == 0:
        user_id = result.stdout.strip()
        if user_id != "0":
            print_status(True, f"Running as non-root user (UID: {user_id})")
            return True
        else:
            print_warning("Running as root user (not recommended for production)")
            return True  # Warning, not failure
    else:
        print_status(False, "Failed to get user ID")
        return False


def test_volume_functionality() -> bool:
    """Test volume creation functionality."""
    print("\nTesting volume functionality...")

    volume_name = "stockula-test-volume"

    # Create volume
    result = run_command(f"docker volume create {volume_name}", check=False)

    if result.returncode == 0:
        print_status(True, "Volume creation works")

        # Clean up
        run_command(f"docker volume rm {volume_name}", check=False)
        return True
    else:
        print_status(False, "Volume creation failed")
        return False


def test_docker_compose() -> bool:
    """Test Docker Compose configuration if available."""
    if not Path("docker-compose.yml").exists():
        print_warning("docker-compose.yml not found")
        return True

    print("\nTesting Docker Compose configuration...")

    # Try docker-compose first
    result = run_command("docker-compose config", check=False)
    if result.returncode == 0:
        print_status(True, "docker-compose.yml syntax is valid")
        return True

    # Try docker compose (v2)
    result = run_command("docker compose config", check=False)
    if result.returncode == 0:
        print_status(True, "docker-compose.yml syntax is valid")
        return True

    print_status(False, "docker-compose.yml syntax is invalid")
    return False


def test_examples_directory() -> bool:
    """Test that examples directory exists."""
    print("\nTesting examples directory...")

    examples_dir = Path("examples")
    if examples_dir.exists() and examples_dir.is_dir():
        python_files = list(examples_dir.glob("*.py"))
        example_count = len(python_files)
        print_status(True, f"Examples directory exists with {example_count} Python files")
        return True
    else:
        print_warning("Examples directory not found")
        return True  # Warning, not failure


def cleanup_test_images():
    """Clean up test images."""
    print("\nCleaning up test images...")

    test_images = ["stockula:validate-test", "stockula:test-validate"]

    for image in test_images:
        run_command(f"docker rmi {image}", check=False, capture_output=True)


def main() -> None:
    """Run Docker validation."""
    parser = argparse.ArgumentParser(description="Validate Docker setup for Stockula")
    parser.add_argument(
        "--quick", action="store_true", help="Quick validation (skip image builds and functionality tests)"
    )
    parser.add_argument("--check-only", action="store_true", help="Same as --quick - only check basic requirements")

    args = parser.parse_args()

    # Quick mode skips builds and functionality tests
    quick_mode = args.quick or args.check_only

    print("🐳 Validating Stockula Docker Setup")
    print("===================================")

    success = True

    # Basic checks (always run)
    if not check_docker_installation():
        sys.exit(1)

    check_docker_compose()  # Optional, warns but doesn't fail

    if not check_required_files():
        sys.exit(1)

    if quick_mode:
        print("\n✅ Quick Docker validation completed!")
        print("\nTo run full validation with image builds, use:")
        print("  uv run validate-docker")
        print("  # or")
        print("  python scripts/validate_docker.py")
        return

    # Full validation (build and test images)
    production_success = test_docker_build("production", "stockula:validate-test")
    if not production_success:
        success = False

    test_success = test_docker_build("test", "stockula:test-validate")
    if not test_success:
        success = False

    if not success:
        print("\n❌ Docker build validation failed!")
        cleanup_test_images()
        sys.exit(1)

    # Test functionality if builds succeeded
    if not test_basic_functionality("stockula:validate-test"):
        success = False

    if not test_uv_functionality("stockula:test-validate"):
        success = False

    if not test_python_version("stockula:validate-test"):
        success = False

    if not test_security("stockula:validate-test"):
        success = False

    if not test_volume_functionality():
        success = False

    # Optional tests (don't fail on these)
    test_docker_compose()
    test_examples_directory()

    # Clean up
    cleanup_test_images()

    if success:
        print("\n🎉 Docker setup validation completed successfully!")
        print("\nNext steps:")
        print("1. Run 'make build' to build all images")
        print("2. Run 'make dev' to start development environment")
        print("3. Run 'make test' to run tests in Docker")
        print("4. See 'make help' for all available commands")
        print("\nFor detailed usage, see docs/deployment/docker.md")
    else:
        print("\n❌ Docker validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
