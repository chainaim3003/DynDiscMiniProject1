#!/bin/bash
################################################################################
# fix-docker-credentials.sh - Fix Docker credential issues on WSL2
################################################################################

set -e

echo "====================================="
echo " Fixing Docker Credentials"
echo "====================================="
echo ""

echo "Step 1: Checking Docker daemon..."
docker info > /dev/null 2>&1 || {
    echo "ERROR: Docker daemon not running"
    echo "Please start Docker Desktop and try again"
    exit 1
}
echo "✓ Docker daemon is running"
echo ""

echo "Step 2: Checking Docker login..."
docker login -u oauth2accesstoken --password-stdin < /dev/null > /dev/null 2>&1 || true
echo ""

echo "Step 3: Clearing Docker credential helper cache..."
# Remove credential helper from config temporarily
if [ -f ~/.docker/config.json ]; then
    echo "Current Docker config:"
    cat ~/.docker/config.json
    echo ""
    
    # Backup config
    cp ~/.docker/config.json ~/.docker/config.json.bak
    
    # Remove credential helper
    jq 'del(.credsStore)' ~/.docker/config.json > ~/.docker/config.json.tmp
    mv ~/.docker/config.json.tmp ~/.docker/config.json
    
    echo "Updated Docker config:"
    cat ~/.docker/config.json
    echo ""
fi

echo "Step 4: Testing Docker pull..."
docker pull alpine:latest || {
    echo "ERROR: Cannot pull images from Docker Hub"
    echo "Please check your internet connection and Docker Desktop settings"
    exit 1
}
echo "✓ Docker pull works"
echo ""

echo "====================================="
echo " Docker Credentials Fixed"
echo "====================================="
echo ""
echo "You can now try rebuilding:"
echo "  docker compose build tsx-shell"
