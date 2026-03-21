#!/bin/bash
#
# Diagnose witness container failure
#

echo "=========================================="
echo "  Witness Container Diagnostic"
echo "=========================================="
echo ""

echo "→ Checking witness container status..."
docker ps -a | grep witness
echo ""

echo "→ Checking witness logs..."
docker logs vleiworklinux1-witness-1 2>&1 | tail -100
echo ""

echo "→ Checking if witness config directory exists..."
ls -la ./config/witnesses/
echo ""

echo "→ Checking witness ports..."
netstat -tlnp 2>/dev/null | grep -E "564[2-7]" || echo "No witness ports found listening"
echo ""

echo "→ Checking docker-compose witness configuration..."
grep -A 20 "witness:" docker-compose.yml
echo ""

echo "=========================================="
echo "  Diagnostic Complete"
echo "=========================================="
