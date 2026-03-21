#!/bin/bash
# deploy.sh - vLEI workshop module infrastructure deploy script
# Sets up schema server, six witnesses, KERIA server, verifier (sally), verification service, and a webhook

set -e

echo "vLEI module infrastructure deploying..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Please install Docker first."
    exit 1
fi

# Check if docker compose is available
if ! command -v docker compose &> /dev/null; then
    echo "docker compose not found. Please install docker compose first."
    exit 1
fi

# Create Docker network if it doesn't exist
echo "Creating vlei-workshop docker network network..."
docker network create vlei_workshop --driver bridge 2>/dev/null || echo "vlei_workshop network already exists"

# Stop and remove existing containers
echo "Cleaning up existing containers..."
docker compose down --remove-orphans --volumes 2>/dev/null || true

# Check if verification service needs to be set up
if [ -f "config/verifier-sally/verification_service_keri.py" ] && [ -f "config/verifier-sally/Dockerfile.verification-keri" ]; then
    echo "Setting up KERI-enabled verification service..."
    cd config/verifier-sally
    cp verification_service_keri.py verification_service.py 2>/dev/null || true
    cp Dockerfile.verification-keri Dockerfile.verification 2>/dev/null || true
    cd ../..
fi

# Build and start services
echo "Starting services and waiting to come up..."
docker compose up --wait -d

# Check service health
echo "Checking service health..."

# Check vLEI server
if curl -f http://127.0.0.1:7723/oobi/EBfdlu8R27Fbx-ehrqwImnK-8Cm79sqbAQ4MmvEAYqao >/dev/null 2>&1; then
    echo "vLEI Server is healthy"
else
    echo "vLEI Server is not responding"
fi

# Check witnesses
witness_urls=(
    "http://127.0.0.1:5642/oobi"  # wan
    "http://127.0.0.1:5643/oobi"  # wil
    "http://127.0.0.1:5644/oobi"  # wes
    "http://127.0.0.1:5645/oobi"  # wit
    "http://127.0.0.1:5646/oobi"  # wub
    "http://127.0.0.1:5647/oobi"  # wyz
    )

for wit_url in "${witness_urls[@]}"; do
    if curl -f "$wit_url" >/dev/null 2>&1; then
        echo "Witness at $wit_url is healthy"
    else
        echo "Witness at $wit_url is not responding"
    fi
done

# Check KERIA
if curl -f http://127.0.0.1:3902/spec.yaml >/dev/null 2>&1; then
    echo "KERIA is healthy"
else
    echo "KERIA is not responding"
fi

# Check hook service
if curl -f http://127.0.0.1:9923/health >/dev/null 2>&1; then
    echo "Hook service is healthy"
else
    echo "Hook service is not responding"
fi

# Check verifier service (Sally)
if curl -f http://127.0.0.1:9723/health >/dev/null 2>&1; then
    echo "Verifier service (Sally) is healthy"
else
    echo "Verifier service (Sally) is not responding"
fi

# Check KERI verification service (if configured)
if curl -f http://127.0.0.1:9724/health >/dev/null 2>&1; then
    echo "✨ KERI Verification service is healthy"
    # Get KERIA connection status
    KERIA_STATUS=$(curl -s http://127.0.0.1:9724/health | grep -o '"keria_status":"[^"]*"' | cut -d'"' -f4)
    if [ "$KERIA_STATUS" = "connected" ]; then
        echo "  └─ KERIA connection: ✓ Connected"
    else
        echo "  └─ KERIA connection: ⚠ Not connected ($KERIA_STATUS)"
    fi
else
    echo "⚠ KERI Verification service not found (optional - for agent delegation)"
    echo "  └─ To enable: Run ./deploy-verification.sh"
fi

echo ""
echo "Deployment complete"
echo ""
echo "Service URLs:"
echo "  schema (vLEI svr): http://127.0.0.1:7723"
echo "          Witnesses: http://127.0.0.1:5642 (wan)" 
echo "                     http://127.0.0.1:5643 (wil)" 
echo "                     http://127.0.0.1:5644 (wes)"
echo "                     http://127.0.0.1:5645 (wit)"
echo "                     http://127.0.0.1:5646 (wub)"
echo "                     http://127.0.0.1:5647 (wyz)"
echo "              KERIA: http://127.0.0.1:3901 (admin)"
echo "                     http://127.0.0.1:3902 (HTTP)"
echo "                     http://127.0.0.1:3903 (boot)"
echo "   Verifier (sally): http://127.0.0.1:9723"
echo "            Webhook: http://127.0.0.1:9923"
if curl -sf http://127.0.0.1:9724/health >/dev/null 2>&1; then
echo "✨ KERI Verification: http://127.0.0.1:9724 (agent delegation)"
fi
echo ""
echo "Next steps:"
echo "   1. Run ./task-scripts/create-geda-aid.sh to create the GEDA AID"
echo "   2. Run ./task-scripts/create-qvi-aid.sh to create the QVI AID"
echo "   3. Run ./task-scripts/create-le-aid.sh to create the LE AID"
echo "   4. Run ./task-scripts/create-person-aid.sh to create the Person AID"
echo "   5. Run ./task-scripts/create-qvi-acdc-credential.sh to issue QVI credential"
echo "   6. Run ./task-scripts/create-le-acdc-credential.sh to issue LE credential"
echo "   7. Run ./task-scripts/create-oor-acdc-credential.sh to issue OOR credentials"
echo ""
echo "To stop the environment, run ./stop.sh"
echo "To run the whole process above run ./run-all.sh"
echo "✨ To run with agent delegation: ./run-all-buyerseller-2-with-agents.sh"
echo ""
