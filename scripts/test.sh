#!/bin/bash
# Test runner script that suppresses pkg_resources deprecation warnings
# These warnings come from the fs package (transitive dependency via fugue)

# Set environment variable to suppress the specific warning
export PYTHONWARNINGS="ignore:pkg_resources is deprecated:UserWarning"

# Run pytest with all provided arguments
uv run pytest "$@"
