#!/bin/bash
set -e

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     DEPLOYING KERI-ENABLED VERIFICATION SERVICE          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check we're in the right place
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found"
    echo "   Please run from vLEIWorkLinux1 directory"
    exit 1
fi

# Backup
BACKUP="docker-compose.yml.backup.$(date +%s)"
cp docker-compose.yml "$BACKUP"
echo "✓ Backup created: $BACKUP"

# Stop any existing verification service
echo ""
echo "Stopping existing service (if any)..."
docker compose stop verification 2>/dev/null || true

# Use KERI-enabled version
echo ""
echo "Setting up KERI-enabled verification..."
cd config/verifier-sally
cp verification_service_keri.py verification_service.py
cp Dockerfile.verification-keri Dockerfile.verification
cd ../..
echo "✓ Using KERI-enabled files"

# Add/update service in docker-compose.yml
echo ""
echo "Updating docker-compose.yml..."
python3 << 'PY_EOF'
import yaml
import sys

try:
    with open('docker-compose.yml', 'r') as f:
        compose = yaml.safe_load(f)
    
    if 'verification' not in compose.get('services', {}):
        compose.setdefault('services', {})['verification'] = {
            'build': {
                'context': './config/verifier-sally',
                'dockerfile': 'Dockerfile.verification'
            },
            'image': 'vlei-verification:latest',
            'container_name': 'vlei-verification',
            'ports': ['9724:9723'],
            'networks': ['default'],
            'restart': 'unless-stopped',
            'environment': {
                'KERIA_URL': 'http://keria:3902',
                'KERIA_BOOT_URL': 'http://keria:3903'
            },
            'depends_on': {
                'keria': {'condition': 'service_healthy'}
            },
            'healthcheck': {
                'test': ['CMD', 'curl', '-f', 'http://localhost:9723/health'],
                'interval': '30s',
                'timeout': '10s',
                'retries': 3
            }
        }
        print("✓ Added verification service to docker-compose.yml")
    else:
        # Update existing service with KERIA connection
        compose['services']['verification']['environment'] = {
            'KERIA_URL': 'http://keria:3902',
            'KERIA_BOOT_URL': 'http://keria:3903'
        }
        compose['services']['verification']['depends_on'] = {
            'keria': {'condition': 'service_healthy'}
        }
        print("✓ Updated verification service with KERIA connection")
    
    with open('docker-compose.yml', 'w') as f:
        yaml.dump(compose, f, default_flow_style=False, sort_keys=False)
    
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PY_EOF

if [ $? -ne 0 ]; then
    echo "❌ Failed to update docker-compose.yml"
    exit 1
fi

# Update TypeScript
echo ""
echo "Updating TypeScript to use verification service..."
TS_FILE="sig-wallet/src/tasks/agent/agent-verify-delegation.ts"

if [ -f "$TS_FILE" ]; then
    cp "$TS_FILE" "$TS_FILE.backup.$(date +%s)"
    sed -i "s|http://verifier:9723/verify/agent-delegation|http://verification:9723/verify/agent-delegation|g" "$TS_FILE"
    
    if grep -q "http://verification:9723/verify/agent-delegation" "$TS_FILE"; then
        echo "✓ Updated TypeScript to use verification service"
    else
        echo "⚠️  Could not update TypeScript automatically"
    fi
else
    echo "⚠️  TypeScript file not found: $TS_FILE"
fi

# Build the service
echo ""
echo "Building verification service with KERI support..."
docker compose build verification

# Start the service
echo ""
echo "Starting verification service..."
docker compose up -d verification

# Rebuild tsx-shell if TypeScript was updated
if [ -f "$TS_FILE" ]; then
    echo ""
    echo "Rebuilding tsx-shell with updated TypeScript..."
    docker compose build tsx-shell
    docker compose up -d tsx-shell
fi

# Wait for service to be ready
echo ""
echo "Waiting for service to be healthy..."
for i in {1..30}; do
    if curl -sf http://localhost:9724/health &>/dev/null; then
        echo "✓ Service is healthy!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Service may still be starting..."
    fi
    sleep 1
done

# Test the service
echo ""
echo "Testing verification service..."
curl -s http://localhost:9724/health | python3 -m json.tool

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              DEPLOYMENT COMPLETE! ✅                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Verification service is running:"
echo "  • Container: vlei-verification"
echo "  • Internal: http://verification:9723/verify/agent-delegation"
echo "  • External: http://localhost:9724/verify/agent-delegation"
echo "  • KERIA: http://keria:3902"
echo ""
echo "Features enabled:"
echo "  ✅ Format validation"
echo "  ✅ KEL queries via KERIA"
echo "  ✅ AID existence verification"
echo "  ✅ Delegation checking"
echo ""
echo "Quick commands:"
echo "  • Health: curl http://localhost:9724/health"
echo "  • Status: docker compose ps verification"
echo "  • Logs:   docker compose logs -f verification"
echo ""
echo "Test full workflow:"
echo "  ./test-agent-verification.sh"
echo ""
