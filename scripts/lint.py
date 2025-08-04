#!/usr/bin/env python
"""Linting utilities for the project.

This script provides automated linting and formatting using ruff, with the ability
to automatically apply fixes when issues are found.

Usage:
    python scripts/lint.py           # Check for issues only (default)
    python scripts/lint.py --fix     # Check and automatically apply fixes
    python scripts/lint.py --check-only  # Explicitly check only, no fixes
    
    # Or using uv:
    uv run lint                      # Check for issues only
    uv run lint --fix                # Check and automatically apply fixes

The script runs the same checks as the CI pipeline and provides consistent
formatting and linting across the codebase.
"""

import argparse
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


def apply_fixes() -> bool:
    """Apply automatic linting fixes and return True if successful."""
    print("\n🔧 Applying automatic fixes...")
    
    # Apply ruff fixes
    print("Applying ruff check fixes...")
    fix_result = run_command("uv run ruff check src tests --fix", check=False)
    
    # Apply formatting
    print("Applying code formatting...")
    format_result = run_command("uv run ruff format src tests", check=False)
    
    if fix_result.returncode != 0 or format_result.returncode != 0:
        print("❌ Some fixes could not be applied automatically")
        if fix_result.returncode != 0:
            print("Ruff fix output:", fix_result.stdout)
            if fix_result.stderr:
                print("Ruff fix errors:", fix_result.stderr)
        if format_result.returncode != 0:
            print("Format output:", format_result.stdout)
            if format_result.stderr:
                print("Format errors:", format_result.stderr)
        return False
    
    print("✅ Automatic fixes applied successfully!")
    return True


def main() -> None:
    """Run linting checks consistent with CI pipeline."""
    parser = argparse.ArgumentParser(description="Run linting checks and optionally apply fixes")
    parser.add_argument(
        "--fix", 
        action="store_true", 
        help="Automatically apply fixes when linting issues are found"
    )
    parser.add_argument(
        "--check-only", 
        action="store_true", 
        help="Only check for issues, don't apply fixes (same as no --fix)"
    )
    
    args = parser.parse_args()
    
    # Default behavior is to apply fixes unless --check-only is specified
    should_fix = args.fix and not args.check_only
    
    print("🔍 Checking code with ruff (consistent with CI)...")
    
    # Run the same commands as CI
    print("Running: uv run ruff check src tests")
    check_result = run_command("uv run ruff check src tests", check=False)
    
    print("\nRunning: uv run ruff format --check src tests")  
    format_result = run_command("uv run ruff format --check src tests", check=False)
    
    # Check if there are any issues
    has_issues = check_result.returncode != 0 or format_result.returncode != 0
    
    if not has_issues:
        print("\n✅ All linting checks passed!")
        return
    
    # Report issues found
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
    
    # Apply fixes if requested
    if should_fix:
        if apply_fixes():
            # Re-run checks to verify fixes worked
            print("\n🔍 Re-checking after applying fixes...")
            final_check = run_command("uv run ruff check src tests", check=False)
            final_format = run_command("uv run ruff format --check src tests", check=False)
            
            if final_check.returncode == 0 and final_format.returncode == 0:
                print("\n✅ All issues fixed successfully!")
                return
            else:
                print("\n⚠️  Some issues remain after applying fixes:")
                if final_check.returncode != 0:
                    print("Remaining check issues:", final_check.stdout)
                if final_format.returncode != 0:
                    print("Remaining format issues:", final_format.stdout)
                sys.exit(1)
        else:
            sys.exit(1)
    else:
        # Show manual fix commands
        print("\n🔧 To fix these issues, run:")
        print("  uv run ruff check src tests --fix")
        print("  uv run ruff format src tests")
        print("\nOr run this script with --fix to apply fixes automatically:")
        print("  python scripts/lint.py --fix")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
