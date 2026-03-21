#!/bin/bash
# test-sally-endpoint-direct.sh
# Direct test of Sally custom endpoint using curl
# Tests ONLY the Sally /verify/agent-delegation endpoint

set -e

AGENT_NAME="${1:-jupiterSellerAgent}"
OOR_HOLDER_NAME="${2:-Jupiter_Chief_Sales_Officer}"

echo "================================================"
echo "DIRECT SALLY ENDPOINT TEST"
echo "================================================"
echo ""

# Extract AIDs
AGENT_INFO="./task-data/${AGENT_NAME}-info.json"
OOR_INFO="./task-data/${OOR_HOLDER_NAME}-info.json"

if [ ! -f "$AGENT_INFO" ] || [ ! -f "$OOR_INFO" ]; then
    echo "❌ Error: Info files not found"
    echo "   Agent: $AGENT_INFO"
    echo "   OOR: $OOR_INFO"
    exit 1
fi

AGENT_AID=$(cat "$AGENT_INFO" | jq -r '.aid')
OOR_AID=$(cat "$OOR_INFO" | jq -r '.aid')

echo "Testing Sally Custom Extension:"
echo "  Endpoint: POST http://verifier:9723/verify/agent-delegation"
echo "  Agent AID: ${AGENT_AID}"
echo "  OOR Holder AID: ${OOR_AID}"
echo ""

# Test from Docker network
echo "Sending request..."
echo ""

docker compose exec -T tsx-shell sh -c "
curl -X POST http://verifier:9723/verify/agent-delegation \
  -H 'Content-Type: application/json' \
  -d '{
    \"agent_aid\": \"${AGENT_AID}\",
    \"oor_holder_aid\": \"${OOR_AID}\"
  }' | jq .
"

echo ""
echo "================================================"
echo "Test complete!"
echo "================================================"
