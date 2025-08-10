#!/bin/bash
# Verify Docker GPU build configuration

echo "Verifying Docker GPU Build Configuration"
echo "========================================="

# Check requirements file exists
if [ -f "requirements-gpu.txt" ]; then
    echo "✓ requirements-gpu.txt exists"
else
    echo "✗ requirements-gpu.txt not found"
    exit 1
fi

# Check that we're not copying pyproject.toml in Dockerfile
if grep -q "COPY.*pyproject.toml" Dockerfile.nvidia; then
    echo "✗ Dockerfile.nvidia still copies pyproject.toml"
    exit 1
else
    echo "✓ Dockerfile.nvidia does not copy pyproject.toml"
fi

# Check that we're not using requirements-gpu-py313.txt
if [ -f "requirements-gpu-py313.txt" ]; then
    echo "✗ requirements-gpu-py313.txt still exists (should be deleted)"
    exit 1
else
    echo "✓ requirements-gpu-py313.txt has been removed"
fi

# Check Python version in Dockerfile
if grep -q "Python 3.12" Dockerfile.nvidia; then
    echo "✓ Dockerfile uses Python 3.12"
else
    echo "✗ Dockerfile not using Python 3.12"
fi

# Check that we're not using deadsnakes PPA
if grep -q "deadsnakes" Dockerfile.nvidia; then
    echo "✗ Dockerfile still uses deadsnakes PPA"
    exit 1
else
    echo "✓ Dockerfile does not use deadsnakes PPA"
fi

echo ""
echo "Configuration looks good! Ready to build:"
echo "  docker buildx build -t smigula/stockula:v0.13.0-gpu \\"
echo "    -f Dockerfile.nvidia --platform linux/amd64 --target gpu-cli ."