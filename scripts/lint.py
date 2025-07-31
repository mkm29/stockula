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
    """Run formatting and linting."""
    print("🔧 Formatting code...")
    run_command("ruff format .")

    print("\n🔍 Attempting to fix linting issues...")
    # First try to fix what we can
    fix_result = run_command("ruff check . --fix --unsafe-fixes", check=False)

    if fix_result.returncode != 0:
        print("\n⚠️  Some issues couldn't be automatically fixed:")
        print(fix_result.stdout)

        # Show remaining issues
        print("\n📋 Remaining issues that need manual fixes:")
        check_result = run_command("ruff check .", check=False)

        if check_result.returncode != 0:
            # Count issues by type
            lines = check_result.stdout.strip().split("\n")
            error_types: dict[str, int] = {}
            for line in lines:
                if " E" in line or " B" in line or " F" in line:
                    parts = line.split(":")
                    if len(parts) >= 4:
                        error_code = parts[3].strip().split()[0]
                        error_types[error_code] = error_types.get(error_code, 0) + 1

            print("\nSummary by error type:")
            for code, count in sorted(error_types.items()):
                print(f"  {code}: {count} issue(s)")

            print(f"\nTotal: {check_result.stdout.strip().split('\n')[-1]}")
            sys.exit(1)
    else:
        print("\n✅ All linting issues fixed!")


if __name__ == "__main__":
    main()
