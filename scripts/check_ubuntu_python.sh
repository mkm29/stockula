#!/bin/bash
# Script to check Python versions available in Ubuntu 24.04

echo "Checking Python versions in Ubuntu 24.04 base image..."
docker run --rm ubuntu:24.04 bash -c "
apt-get update >/dev/null 2>&1
echo 'Default Python 3 version:'
apt-cache policy python3 | head -5
echo ''
echo 'Available Python 3 packages:'
apt-cache search '^python3\.[0-9]+$' | sort
echo ''
echo 'Checking if Python 3.13 is available by default:'
apt-cache policy python3.13 2>/dev/null || echo 'Python 3.13 not available in default repos'
"