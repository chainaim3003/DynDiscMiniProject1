#!/bin/bash
################################################################################
# invoice-registry-create-agent.sh
# Create credential registry for agent's self-attested invoice credentials
################################################################################

set -e

AGENT_ALIAS="${1:-jupiterSellerAgent}"
REGISTRY_NAME="${2:-${AGENT_ALIAS}_INVOICE_REGISTRY}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Source environment variables
source ./task-scripts/workshop-env-vars.sh

echo "Creating invoice credential registry for agent..."
echo "  Agent: $AGENT_ALIAS"
echo "  Registry: $REGISTRY_NAME"
echo ""

# Check if agent info exists
if [ ! -f "./task-data/${AGENT_ALIAS}-info.json" ]; then
    echo -e "${RED}ERROR: Agent info file not found: ./task-data/${AGENT_ALIAS}-info.json${NC}"
    echo "The agent delegation must be completed before creating a registry."
    exit 1
fi

# Check if BRAN file exists
BRAN_FILE="./task-data/${AGENT_ALIAS}-bran.txt"
if [ ! -f "$BRAN_FILE" ]; then
    echo -e "${RED}ERROR: Agent BRAN file not found: $BRAN_FILE${NC}"
    echo "The agent must have been created with a unique BRAN."
    exit 1
fi

AGENT_BRAN=$(cat "$BRAN_FILE")
echo "  Using agent's unique BRAN: ${AGENT_BRAN:0:20}..."
echo ""

# Create registry using the agent's unique BRAN
docker compose exec -T tsx-shell tsx \
  sig-wallet/src/tasks/invoice/invoice-registry-create.ts \
  docker \
  "$AGENT_ALIAS" \
  "$AGENT_BRAN" \
  "$REGISTRY_NAME" \
  "/task-data"

# Verify registry info was created
REGISTRY_INFO_FILE="./task-data/${AGENT_ALIAS}-invoice-registry-info.json"
if [ -f "$REGISTRY_INFO_FILE" ]; then
    echo ""
    echo -e "${GREEN}✓ Invoice registry created successfully${NC}"
    echo "  Registry info: $REGISTRY_INFO_FILE"
    cat "$REGISTRY_INFO_FILE"
else
    echo -e "${YELLOW}Note: Registry info file may not have been created${NC}"
fi
