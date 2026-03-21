#!/bin/bash
#
# Deploy vLEI with Agent Delegation Verification Service
#

echo "======================================"
echo "  Deploying with Agent Verification"
echo "======================================"
echo ""

# Stop existing services
echo "→ Stopping existing services..."
./stop.sh 2>/dev/null || true
echo ""

# Build all services including vlei-verification
echo "→ Building all services (including vlei-verification)..."
docker compose build --no-cache
echo ""

# Deploy
echo "→ Starting all services..."
./deploy.sh
echo ""

# Wait for vlei-verification to be healthy
echo "→ Waiting for vlei-verification service..."
for i in {1..30}; do
    if curl -s http://localhost:9724/health >/dev/null 2>&1; then
        echo "✓ vlei-verification service is healthy"
        break
    fi
    sleep 2
    echo "  Waiting... ($i/30)"
done
echo ""

# Test the verification endpoint
echo "→ Testing verification service..."
HEALTH=$(curl -s http://localhost:9724/health 2>/dev/null || echo "ERROR")
if [[ "$HEALTH" == *"healthy"* ]]; then
    echo "✓ Verification service responding correctly"
    echo ""
    echo "Service details:"
    curl -s http://localhost:9724/health | python3 -m json.tool 2>/dev/null || echo "$HEALTH"
else
    echo "⚠ Verification service may not be ready yet"
    echo "   Check with: docker logs vlei_verification"
fi
echo ""

echo "======================================"
echo "  Deployment Complete"
echo "======================================"
echo ""
echo "Services running:"
echo "  • Standard verifier:     http://localhost:9723"
echo "  • Agent verifier:        http://localhost:9724"
echo "  • Agent verify endpoint: http://vlei-verification:9723/verify/agent-delegation"
echo ""
echo "Next step:"
echo "  ./run-all-buyerseller-2-with-agents.sh"
echo ""
