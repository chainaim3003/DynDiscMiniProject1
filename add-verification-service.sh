#!/bin/bash

# Add KERI Agent Delegation Verification Service to docker-compose.yml

echo "Adding vlei-verification service to docker-compose.yml..."

# Create backup
cp docker-compose.yml docker-compose.yml.backup.$(date +%Y%m%d_%H%M%S)

# Add the new service after the verifier service
cat >> docker-compose.yml << 'EOF'

  # KERI-enabled Agent Delegation Verification Service
  vlei-verification:
    build:
      context: ./config/verifier-sally
      dockerfile: Dockerfile.verification-keri
      no_cache: true
    container_name: vlei_verification
    hostname: vlei-verification
    stop_grace_period: 1s
    environment:
      PYTHONUNBUFFERED: 1
      PYTHONIOENCODING: UTF-8
      PYTHONWARNINGS: 'ignore::SyntaxWarning'
      KERIA_URL: http://keria:3902
    healthcheck:
      test: [ "CMD", "wget", "--spider", "--tries=1", "--no-verbose", "http://127.0.0.1:9723/health" ]
      interval: 3s
      timeout: 3s
      retries: 4
      start_period: 2s
    ports:
      - "9724:9723"
    depends_on:
      keria:
        condition: service_healthy
      schema:
        condition: service_healthy
EOF

echo "✓ vlei-verification service added"
echo ""
echo "Service details:"
echo "  - Container name: vlei_verification"
echo "  - Hostname: vlei-verification"
echo "  - Internal port: 9723"
echo "  - External port: 9724"
echo "  - Endpoint: http://vlei-verification:9723/verify/agent-delegation"
echo ""
echo "Next steps:"
echo "  1. docker compose build --no-cache"
echo "  2. ./deploy.sh"
