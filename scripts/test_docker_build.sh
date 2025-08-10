#!/bin/bash
# Test Docker GPU build in stages to identify issues

echo "Testing Docker GPU Build for Stockula"
echo "======================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Base image and Python installation
echo -e "\n${YELLOW}Test 1: Building nvidia-base stage...${NC}"
docker buildx build \
  -f Dockerfile.nvidia \
  --target nvidia-base \
  --platform linux/amd64 \
  -t stockula:test-base \
  --progress plain \
  . 2>&1 | tail -20

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✓ Base stage built successfully${NC}"
else
  echo -e "${RED}✗ Base stage failed${NC}"
  exit 1
fi

# Test 2: Dependencies stage
echo -e "\n${YELLOW}Test 2: Building gpu-dependencies stage...${NC}"
docker buildx build \
  -f Dockerfile.nvidia \
  --target gpu-dependencies \
  --platform linux/amd64 \
  -t stockula:test-deps \
  --progress plain \
  . 2>&1 | tail -20

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✓ Dependencies stage built successfully${NC}"
else
  echo -e "${RED}✗ Dependencies stage failed${NC}"
  exit 1
fi

# Test 3: Full build
echo -e "\n${YELLOW}Test 3: Building full gpu-cli stage...${NC}"
docker buildx build \
  -f Dockerfile.nvidia \
  --target gpu-cli \
  --platform linux/amd64 \
  -t smigula/stockula:v0.13.0-gpu \
  . 2>&1 | tail -20

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✓ Full build completed successfully!${NC}"
  echo -e "\n${GREEN}Docker image ready: smigula/stockula:v0.13.0-gpu${NC}"
else
  echo -e "${RED}✗ Full build failed${NC}"
  exit 1
fi

# Test 4: Run the image
echo -e "\n${YELLOW}Test 4: Testing the built image...${NC}"
docker run --rm smigula/stockula:v0.13.0-gpu stockula --version

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✓ Image runs successfully${NC}"
else
  echo -e "${RED}✗ Image failed to run${NC}"
fi
