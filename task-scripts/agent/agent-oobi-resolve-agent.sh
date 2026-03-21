#!/bin/bash
################################################################################
# agent-oobi-resolve-agent.sh - Resolve OOBI between two agents
#
# Purpose: Allow one agent to resolve the OOBI of another agent
#          for peer-to-peer communication and credential exchange
#
# Usage: ./agent-oobi-resolve-agent.sh <SOURCE_AGENT> <TARGET_AGENT>
#
# Example: ./agent-oobi-resolve-agent.sh jupiterSalesAgent tommyBuyerAgent
#
# Date: November 14, 2025
################################################################################

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SOURCE_AGENT="${1}"
TARGET_AGENT="${2}"
ENV="${3:-docker}"

if [ -z "$SOURCE_AGENT" ] || [ -z "$TARGET_AGENT" ]; then
    echo -e "${RED}ERROR: Missing required arguments${NC}"
    echo "Usage: $0 <SOURCE_AGENT> <TARGET_AGENT>"
    exit 1
fi

# Passcode
AGENT_PASSCODE="AgentPass123"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Agent OOBI Resolution${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Source Agent: $SOURCE_AGENT"
echo "Target Agent: $TARGET_AGENT"
echo ""

# Execute OOBI resolution
echo -e "${BLUE}→ Resolving $TARGET_AGENT OOBI from $SOURCE_AGENT...${NC}"

docker compose exec -T tsx-shell tsx sig-wallet/src/tasks/agent/agent-oobi-resolve-agent.ts \
  "$ENV" \
  "$AGENT_PASSCODE" \
  "$SOURCE_AGENT" \
  "$TARGET_AGENT"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ OOBI resolution successful${NC}"
    echo -e "${GREEN}✓ $SOURCE_AGENT can now communicate with $TARGET_AGENT${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}✗ OOBI resolution failed${NC}"
    echo ""
    exit 1
fi

exit 0
