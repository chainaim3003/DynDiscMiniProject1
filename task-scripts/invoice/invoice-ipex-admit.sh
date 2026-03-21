#!/bin/bash
################################################################################
# invoice-ipex-admit.sh - Admit IPEX grant for invoice credential
#
# Purpose: tommyBuyerAgent admits the IPEX grant from jupiterSellerAgent
#          for the invoice credential
#
# Usage: ./invoice-ipex-admit.sh <RECEIVER_AGENT> <SENDER_AGENT>
#
# Example: ./invoice-ipex-admit.sh tommyBuyerAgent jupiterSellerAgent
#
# Date: November 27, 2025
################################################################################

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

RECEIVER_AGENT="${1:-tommyBuyerAgent}"
SENDER_AGENT="${2:-jupiterSellerAgent}"
ENV="${3:-docker}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  IPEX ADMIT: Invoice Credential${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Receiver: $RECEIVER_AGENT"
echo "Sender: $SENDER_AGENT"
echo ""

# Check if receiver's BRAN file exists
RECEIVER_BRAN_FILE="./task-data/${RECEIVER_AGENT}-bran.txt"
if [ ! -f "$RECEIVER_BRAN_FILE" ]; then
    echo -e "${RED}ERROR: Receiver BRAN file not found: $RECEIVER_BRAN_FILE${NC}"
    echo -e "${YELLOW}The agent must have been created with a unique BRAN.${NC}"
    exit 1
fi

RECEIVER_BRAN=$(cat "$RECEIVER_BRAN_FILE")
echo "Using receiver's unique BRAN: ${RECEIVER_BRAN:0:20}..."
echo ""

# Execute IPEX admit
echo -e "${BLUE}→ Admitting IPEX grant...${NC}"

docker compose exec -T tsx-shell tsx sig-wallet/src/tasks/invoice/invoice-ipex-admit.ts \
  "$ENV" \
  "$RECEIVER_BRAN" \
  "$RECEIVER_AGENT" \
  "$SENDER_AGENT" \
  "/task-data"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ IPEX admit completed successfully${NC}"
    echo -e "${GREEN}✓ Invoice credential now available in $RECEIVER_AGENT's KERIA${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}✗ IPEX admit failed${NC}"
    echo ""
    exit 1
fi

exit 0
