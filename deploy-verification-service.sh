#!/bin/bash
##########################################
# Deploy Standalone Verification Service
##########################################

echo "=========================================="
echo "  DEPLOYING VERIFICATION SERVICE"
echo "=========================================="
echo ""

cd ~/projects/vLEIWorkLinux1 || cd ~/vLEIWorkLinux1

# Step 1: Check if verification service is already in docker-compose.yml
echo "[1/5] Checking docker-compose.yml..."
if grep -q "vlei-verification:" docker-compose.yml; then
    echo "✓ Verification service already exists in docker-compose.yml"
else
    echo "⚠️  Adding verification service to docker-compose.yml..."
    
    # Backup docker-compose.yml
    cp docker-compose.yml docker-compose.yml.backup.$(date +%Y%m%d_%H%M%S)
    echo "✓ Backup created"
    
    # Add verification service to docker-compose.yml
    cat >> docker-compose.yml << 'EOF'

  # Standalone Agent Delegation Verification Service
  vlei-verification:
    container_name: vlei-verification
    build:
      context: ./config/verifier-sally
      dockerfile: Dockerfile.verification-keri
    stop_grace_period: 1s
    environment:
      KERIA_URL: http://keria:3902
    healthcheck:
      test: ["CMD", "curl", "-f", "http://127.0.0.1:9723/health"]
      interval: 3s
      timeout: 3s
      retries: 4
      start_period: 2s
    ports:
      - "9724:9723"  # External port 9724 to avoid conflict with Sally on 9723
    depends_on:
      keria:
        condition: service_healthy
EOF
    
    echo "✓ Verification service added to docker-compose.yml"
fi
echo ""

# Step 2: Build the service
echo "[2/5] Building verification service..."
docker compose build vlei-verification
echo ""

# Step 3: Start the service
echo "[3/5] Starting verification service..."
docker compose up -d vlei-verification
echo ""

# Step 4: Wait for health check
echo "[4/5] Waiting for service to be healthy..."
sleep 5

if docker compose ps vlei-verification | grep -q "Up"; then
    echo "✓ Verification service is running"
else
    echo "✗ Service failed to start"
    docker compose logs vlei-verification
    exit 1
fi
echo ""

# Step 5: Test the endpoint
echo "[5/5] Testing verification endpoint..."

# Test health endpoint
echo "Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:9724/health)
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo "✓ Health check passed"
    echo "$HEALTH_RESPONSE" | jq '.' 2>/dev/null || echo "$HEALTH_RESPONSE"
else
    echo "✗ Health check failed"
    echo "$HEALTH_RESPONSE"
fi
echo ""

# Test verification endpoint
echo "Testing verification endpoint..."
TEST_RESPONSE=$(curl -s -X POST http://localhost:9724/verify/agent-delegation \
  -H "Content-Type: application/json" \
  -d '{
    "aid": "EJjctqaH4B42r7ERQ0O7FND9q5Fme8j2P9rHe2S6VN80",
    "agent_aid": "EMBa71KdHNH1fhijsptdBXOAg4GhlWWh2JVWPitd4G1v",
    "verify_kel": false
  }')

if echo "$TEST_RESPONSE" | grep -q "valid"; then
    echo "✓ Verification endpoint working"
    echo "$TEST_RESPONSE" | jq '.' 2>/dev/null || echo "$TEST_RESPONSE"
else
    echo "⚠️  Verification endpoint response:"
    echo "$TEST_RESPONSE"
fi
echo ""

echo "=========================================="
echo "  DEPLOYMENT COMPLETE!"
echo "=========================================="
echo ""
echo "Service Info:"
echo "  Container: vlei-verification"
echo "  Internal URL: http://verification:9723"
echo "  External URL: http://localhost:9724"
echo "  Health: http://localhost:9724/health"
echo ""
echo "Next Steps:"
echo "  1. Copy updated TypeScript file to WSL"
echo "  2. Rebuild tsx-shell container"
echo "  3. Test with: ./test-agent-verification.sh"
echo ""
