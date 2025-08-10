#!/bin/bash
# Test Python 3.13 availability in NVIDIA CUDA Docker image

echo "Testing Python 3.13 installation methods in nvidia/cuda:13.0.0-devel-ubuntu24.04"
echo "=" * 60

# Test 1: Check base image Python version
echo -e "\n1. Testing base image Python version..."
docker run --rm nvidia/cuda:13.0.0-devel-ubuntu24.04 bash -c "
    apt-get update -qq 2>/dev/null
    echo 'Default Python 3 version in base image:'
    python3 --version 2>/dev/null || echo 'No Python 3 installed'
    apt-cache policy python3 2>/dev/null | grep -E 'Candidate|Installed' || echo 'python3 package not found'
"

# Test 2: Try deadsnakes PPA
echo -e "\n2. Testing deadsnakes PPA installation..."
docker run --rm nvidia/cuda:13.0.0-devel-ubuntu24.04 bash -c "
    apt-get update -qq && \
    apt-get install -y -qq software-properties-common curl 2>/dev/null && \
    add-apt-repository ppa:deadsnakes/ppa -y 2>/dev/null && \
    apt-get update -qq && \
    echo 'Checking available Python 3.13 packages:' && \
    apt-cache search '^python3.13' | head -10 && \
    echo '' && \
    echo 'Attempting to install python3.13:' && \
    apt-get install -y python3.13 2>&1 | tail -5 && \
    python3.13 --version 2>/dev/null || echo 'Python 3.13 installation failed'
"

# Test 3: Test uv with Python 3.13
echo -e "\n3. Testing uv compatibility with Python 3.13..."
docker run --rm nvidia/cuda:13.0.0-devel-ubuntu24.04 bash -c "
    apt-get update -qq && \
    apt-get install -y -qq software-properties-common curl 2>/dev/null && \
    add-apt-repository ppa:deadsnakes/ppa -y 2>/dev/null && \
    apt-get update -qq && \
    apt-get install -y -qq python3.13 python3.13-venv 2>/dev/null && \
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null && \
    export PATH='/root/.local/bin:\$PATH' && \
    echo 'uv version:' && \
    uv --version && \
    echo '' && \
    echo 'Creating venv with Python 3.13:' && \
    uv venv /tmp/test --python python3.13 2>&1 | head -5 && \
    /tmp/test/bin/python --version 2>/dev/null || echo 'venv creation failed'
"

echo -e "\nTest complete. Review output above to identify issues."