#!/bin/bash
# test-geda-verbose.sh - Test geda-aid-create with timeout and verbose output

set -e

echo "================================================"
echo "Testing GEDA AID Creation (with 30s timeout)"
echo "================================================"
echo ""

# Source environment
source ./task-scripts/workshop-env-vars.sh
echo "✓ Environment loaded (GEDA_SALT = ${GEDA_SALT})"
echo ""

# Check containers
echo "Checking containers..."
if ! docker ps --filter "name=tsx_shell" --format "{{.Names}}" | grep -q "tsx_shell"; then
    echo "✗ tsx_shell not running!"
    exit 1
fi
if ! docker ps --filter "name=keria" --format "{{.Names}}" | grep -q "keria"; then
    echo "✗ keria not running!"
    exit 1
fi
echo "✓ Containers are running"
echo ""

# Test KERIA connectivity
echo "Testing KERIA connectivity..."
if ! curl -s -m 5 http://127.0.0.1:3902/spec.yaml >/dev/null 2>&1; then
    echo "✗ KERIA not accessible from host!"
    echo "  KERIA may still be starting up..."
    echo "  Try: ./safe-geda-create.sh (which waits for KERIA)"
    exit 1
fi
echo "✓ KERIA is accessible"
echo ""

# Run with timeout
echo "Running geda-aid-create (will timeout after 30s if hung)..."
echo "Press Ctrl+C to cancel"
echo ""

timeout 30s docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh geda/geda-aid-create.ts \
    'docker' \
    "${GEDA_SALT}" \
    "/task-data" || {
    EXIT_CODE=$?
    echo ""
    echo "================================================"
    if [ $EXIT_CODE -eq 124 ]; then
        echo "✗ TIMEOUT after 30 seconds!"
        echo "  The script is hanging, likely at KERIA connection"
    else
        echo "✗ ERROR occurred (exit code: $EXIT_CODE)"
    fi
    echo "================================================"
    echo ""
    echo "Try:"
    echo "  1. ./diagnose-hang.sh  (to see what's wrong)"
    echo "  2. ./safe-geda-create.sh  (waits for KERIA automatically)"
    exit 1
}

echo ""
echo "✓ Success! Checking output files..."
if [ -f "./task-data/geda-aid.txt" ]; then
    GEDA_PREFIX=$(cat ./task-data/geda-aid.txt)
    echo "  GEDA AID: ${GEDA_PREFIX}"
else
    echo "  ✗ geda-aid.txt not created"
fi
