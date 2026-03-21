#!/bin/bash
################################################################################
# subagent-verify-delegation.sh
#
# Purpose: Verify sub-agent delegation via Sally verifier
#          Verifies that sub-agent is properly delegated from parent agent
#
# Usage: ./task-scripts/subagent/subagent-verify-delegation.sh <sub_agent_alias> <parent_agent_alias>
################################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SUB_AGENT_ALIAS=$1
PARENT_AGENT_ALIAS=$2

if [ -z "$SUB_AGENT_ALIAS" ] || [ -z "$PARENT_AGENT_ALIAS" ]; then
    echo -e "${RED}ERROR: Missing arguments${NC}"
    echo "Usage: $0 <sub_agent_alias> <parent_agent_alias>"
    exit 1
fi

echo -e "${BLUE}Verifying sub-agent delegation via Sally${NC}"
echo "  Sub-Agent: $SUB_AGENT_ALIAS"
echo "  Parent Agent: $PARENT_AGENT_ALIAS"
echo ""

# Load sub-agent info
SUB_INFO_FILE="./task-data/${SUB_AGENT_ALIAS}-info.json"
if [ ! -f "$SUB_INFO_FILE" ]; then
    echo -e "${RED}ERROR: Sub-agent info file not found: $SUB_INFO_FILE${NC}"
    exit 1
fi

SUB_AID=$(jq -r '.aid' "$SUB_INFO_FILE")

# Load parent agent info
PARENT_INFO_FILE="./task-data/${PARENT_AGENT_ALIAS}-info.json"
if [ ! -f "$PARENT_INFO_FILE" ]; then
    echo -e "${RED}ERROR: Parent agent info file not found: $PARENT_INFO_FILE${NC}"
    exit 1
fi

PARENT_AID=$(jq -r '.aid' "$PARENT_INFO_FILE")

echo "Verifying delegation for sub-agent $SUB_AGENT_ALIAS"
echo "Sub-Agent AID: $SUB_AID"
echo "Parent Agent AID: $PARENT_AID"
echo ""

# Call Sally verifier
VERIFY_URL="http://vlei-verification:9723/verify/agent-delegation"

echo "Calling Sally verifier at $VERIFY_URL"
echo ""

RESULT=$(docker compose exec -T tsx-shell curl -s -X POST "$VERIFY_URL" \
    -H "Content-Type: application/json" \
    -d "{
        \"agent_aid\": \"$SUB_AID\",
        \"controller_aid\": \"$PARENT_AID\",
        \"verification_type\": \"sub_delegation\"
    }")

echo "============================================================"
echo "SALLY VERIFICATION RESULT (SUB-DELEGATION)"
echo "============================================================"
echo "$RESULT" | jq '.'
echo "============================================================"
echo ""

# Check if verification passed
VERIFIED=$(echo "$RESULT" | jq -r '.verified // false')

if [ "$VERIFIED" = "true" ]; then
    echo -e "${GREEN}✓ Sub-agent delegation verified successfully${NC}"
    echo "  Sub-Agent: $SUB_AGENT_ALIAS ($SUB_AID)"
    echo "  Delegated from: $PARENT_AGENT_ALIAS ($PARENT_AID)"
    echo "  Delegation Type: Agent → Sub-Agent"
else
    echo -e "${RED}✗ Sub-agent delegation verification failed${NC}"
    exit 1
fi

exit 0