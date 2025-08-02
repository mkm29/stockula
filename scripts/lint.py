#!/usr/bin/env python
"""Linting utilities for the project."""

import subprocess
import sys


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and optionally exit on failure."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if check and result.returncode != 0:
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        sys.exit(result.returncode)

    return result


def main() -> None:
    """Run linting checks consistent with CI pipeline."""
    print("🔍 Checking code with ruff (consistent with CI)...")
    
    # Run the same commands as CI
    print("Running: uv run ruff check src tests")
    check_result = run_command("uv run ruff check src tests", check=False)
    
    print("\nRunning: uv run ruff format --check src tests")  
    format_result = run_command("uv run ruff format --check src tests", check=False)
    
    # Report results
    if check_result.returncode == 0 and format_result.returncode == 0:
        print("\n✅ All linting checks passed!")
    else:
        print("\n❌ Linting issues found:")
        
        if check_result.returncode != 0:
            print("\n📋 Ruff check issues:")
            print(check_result.stdout)
            if check_result.stderr:
                print(check_result.stderr)
        
        if format_result.returncode != 0:
            print("\n📋 Format check issues:")
            print(format_result.stdout)
            if format_result.stderr:
                print(format_result.stderr)
        
        print("\n🔧 To fix these issues, run:")
        print("  uv run ruff check src tests --fix")
        print("  uv run ruff format src tests")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
