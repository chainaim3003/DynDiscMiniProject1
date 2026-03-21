#!/bin/bash
################################################################################
# sync-windows-to-wsl.sh
# Sync all files from Windows to WSL
################################################################################

set -e

WINDOWS_PATH="/mnt/c/SATHYA/CHAINAIM3003/mcp-servers/stellarboston/vLEI1/vLEIWorkLinux1"
WSL_PATH="$HOME/projects/vLEIWorkLinux1"

echo "=========================================="
echo "Syncing Windows to WSL"
echo "=========================================="
echo ""
echo "From: $WINDOWS_PATH"
echo "To:   $WSL_PATH"
echo ""

# Ensure WSL directory exists
mkdir -p "$WSL_PATH"

# Sync all files (excluding node_modules, .git, and task-data)
echo "Copying files..."
rsync -av --progress \
  --exclude='node_modules/' \
  --exclude='.git/' \
  --exclude='task-data/' \
  --exclude='*.log' \
  "$WINDOWS_PATH/" "$WSL_PATH/"

echo ""
echo "=========================================="
echo "✅ Sync Complete!"
echo "=========================================="
echo ""

# Make shell scripts executable
echo "Making shell scripts executable..."
find "$WSL_PATH" -type f -name "*.sh" -exec chmod +x {} \;

echo ""
echo "Checking new invoice files..."
echo ""

# Check invoice files
if [ -d "$WSL_PATH/schemas" ]; then
    echo "✅ schemas/ directory synced"
else
    echo "❌ schemas/ directory missing"
fi

if [ -d "$WSL_PATH/sig-wallet/src/tasks/invoice" ]; then
    echo "✅ invoice TypeScript tasks synced"
    ls -la "$WSL_PATH/sig-wallet/src/tasks/invoice/"
else
    echo "❌ invoice tasks missing"
fi

if [ -d "$WSL_PATH/task-scripts/invoice" ]; then
    echo "✅ invoice shell scripts synced"
    ls -la "$WSL_PATH/task-scripts/invoice/"
else
    echo "❌ invoice shell scripts missing"
fi

echo ""
echo "Ready to run!"
echo ""
echo "Next steps:"
echo "  cd ~/projects/vLEIWorkLinux1"
echo "  ./verify-invoice-implementation.sh"
echo "  ./run-all-buyerseller-3-with-agents.sh"
