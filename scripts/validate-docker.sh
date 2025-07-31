#!/bin/bash
# Script to validate Docker setup for Stockula project

set -e

echo "🐳 Validating Stockula Docker Setup"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        return 1
    fi
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if Docker is installed and running
echo "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗${NC} Docker is not installed"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}✗${NC} Docker daemon is not running"
    echo "Please start Docker daemon"
    exit 1
fi

print_status 0 "Docker is installed and running"

# Check if Docker Compose is available
echo ""
echo "Checking Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    if ! docker compose version &> /dev/null; then
        print_warning "Docker Compose not found, but continuing as it's optional"
    else
        print_status 0 "Docker Compose (v2) is available"
    fi
else
    print_status 0 "Docker Compose is available"
fi

# Check if required files exist
echo ""
echo "Checking required files..."
required_files=("Dockerfile" "pyproject.toml" "uv.lock" ".dockerignore")

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_status 0 "$file exists"
    else
        print_status 1 "$file missing"
        exit 1
    fi
done

# Test build for production target
echo ""
echo "Testing Docker build (production target)..."
if docker build --target production -t stockula:validate-test . > /tmp/docker-build.log 2>&1; then
    print_status 0 "Production image builds successfully"
else
    print_status 1 "Production image build failed"
    echo "Build log:"
    cat /tmp/docker-build.log
    exit 1
fi

# Test build for test target
echo ""
echo "Testing Docker build (test target)..."
if docker build --target test -t stockula:test-validate . > /tmp/docker-test-build.log 2>&1; then
    print_status 0 "Test image builds successfully"
else
    print_status 1 "Test image build failed"
    echo "Build log:"
    cat /tmp/docker-test-build.log
    exit 1
fi

# Test basic functionality
echo ""
echo "Testing basic functionality..."
if docker run --rm stockula:validate-test python -c "import stockula; print('Stockula imports successfully')" > /tmp/docker-import.log 2>&1; then
    print_status 0 "Stockula package imports successfully"
else
    print_status 1 "Stockula package import failed"
    cat /tmp/docker-import.log
    exit 1
fi

# Test uv command
echo ""
echo "Testing uv functionality..."
if docker run --rm stockula:test-validate uv --version > /tmp/docker-uv.log 2>&1; then
    UV_VERSION=$(cat /tmp/docker-uv.log)
    print_status 0 "UV is working: $UV_VERSION"
else
    print_status 1 "UV command failed"
    cat /tmp/docker-uv.log
    exit 1
fi

# Test Python version
echo ""
echo "Testing Python version..."
PYTHON_VERSION=$(docker run --rm stockula:validate-test python --version)
if [[ "$PYTHON_VERSION" == *"3.13"* ]]; then
    print_status 0 "Python version: $PYTHON_VERSION"
else
    print_warning "Expected Python 3.13, got: $PYTHON_VERSION"
fi

# Test that non-root user is used
echo ""
echo "Testing security (non-root user)..."
USER_ID=$(docker run --rm stockula:validate-test id -u)
if [ "$USER_ID" != "0" ]; then
    print_status 0 "Running as non-root user (UID: $USER_ID)"
else
    print_warning "Running as root user (not recommended for production)"
fi

# Test volume creation
echo ""
echo "Testing volume functionality..."
if docker volume create stockula-test-volume > /dev/null 2>&1; then
    print_status 0 "Volume creation works"
    docker volume rm stockula-test-volume > /dev/null 2>&1
else
    print_status 1 "Volume creation failed"
fi

# Test Docker Compose if available
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    echo ""
    echo "Testing Docker Compose configuration..."

    # Validate docker-compose.yml syntax
    if [ -f "docker-compose.yml" ]; then
        if docker-compose config > /dev/null 2>&1 || docker compose config > /dev/null 2>&1; then
            print_status 0 "docker-compose.yml syntax is valid"
        else
            print_status 1 "docker-compose.yml syntax is invalid"
        fi
    else
        print_warning "docker-compose.yml not found"
    fi
fi

# Test that examples directory exists
echo ""
echo "Testing examples directory..."
if [ -d "examples" ]; then
    EXAMPLE_COUNT=$(find examples -name "*.py" | wc -l)
    print_status 0 "Examples directory exists with $EXAMPLE_COUNT Python files"
else
    print_warning "Examples directory not found"
fi

# Clean up test images
echo ""
echo "Cleaning up test images..."
docker rmi stockula:validate-test > /dev/null 2>&1 || true
docker rmi stockula:test-validate > /dev/null 2>&1 || true
rm -f /tmp/docker-*.log

echo ""
echo -e "${GREEN}🎉 Docker setup validation completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "1. Run 'make build' to build all images"
echo "2. Run 'make dev' to start development environment"
echo "3. Run 'make test' to run tests in Docker"
echo "4. See 'make help' for all available commands"
echo ""
echo "For detailed usage, see docs/DOCKER.md"
