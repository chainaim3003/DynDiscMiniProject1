#!/bin/bash
################################################################################
# invoice-acdc-admit-self.sh - Admit self-attested invoice credential
#
# Purpose: Agent admits its own self-attested invoice credential
#          This completes the self-attestation process
#
# Usage: ./invoice-acdc-admit-self.sh <AGENT_NAME>
#
# Example: ./invoice-acdc-admit-self.sh jupiterSalesAgent
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

AGENT_NAME="${1:-jupiterSalesAgent}"
ENV="${2:-docker}"

# Passcode
AGENT_PASSCODE="AgentPass123"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  ADMIT SELF-ATTESTED INVOICE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Agent: $AGENT_NAME"
echo ""

# Execute admit
echo -e "${BLUE}→ Admitting self-attested invoice credential...${NC}"

docker compose exec -T tsx-shell tsx sig-wallet/src/tasks/invoice/invoice-acdc-admit-self.ts \
  "$ENV" \
  "$AGENT_PASSCODE" \
  "$AGENT_NAME"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Self-attested invoice admitted successfully${NC}"
    echo -e "${GREEN}✓ Credential now fully stored in $AGENT_NAME's KERIA${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}✗ Admit failed${NC}"
    echo ""
    exit 1
fi

exit 0
