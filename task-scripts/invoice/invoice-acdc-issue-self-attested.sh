#!/bin/bash
################################################################################
# invoice-acdc-issue-self-attested.sh
# Issue self-attested invoice credential from agent to itself
# The credential will be granted to another agent in a separate step
################################################################################

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ISSUER_AGENT="${1:-jupiterSellerAgent}"
INVOICE_CONFIG_FILE="${2:-./appconfig/invoiceConfig.json}"

echo "Issuing self-attested invoice credential..."
echo "  Issuer Agent: $ISSUER_AGENT"
echo ""

# Check if issuer's BRAN file exists
BRAN_FILE="./task-data/${ISSUER_AGENT}-bran.txt"
if [ ! -f "$BRAN_FILE" ]; then
    echo -e "${RED}ERROR: Agent BRAN file not found: $BRAN_FILE${NC}"
    echo -e "${YELLOW}The agent must have been created with a unique BRAN.${NC}"
    exit 1
fi

ISSUER_BRAN=$(cat "$BRAN_FILE")
echo "Using agent's unique BRAN: ${ISSUER_BRAN:0:20}..."

# Load configuration
REGISTRY_NAME="${ISSUER_AGENT}_INVOICE_REGISTRY"

# Get issuer agent AID
ISSUER_INFO_FILE="./task-data/${ISSUER_AGENT}-info.json"
if [ ! -f "$ISSUER_INFO_FILE" ]; then
    echo -e "${RED}ERROR: Issuer agent info file not found: $ISSUER_INFO_FILE${NC}"
    echo "The issuer agent must exist before issuing invoices"
    exit 1
fi
ISSUER_AID=$(cat "$ISSUER_INFO_FILE" | jq -r '.aid')

# Get invoice data from config
SELLER_LEI=$(jq -r '.invoice.issuer.lei' "$INVOICE_CONFIG_FILE")
BUYER_LEI=$(jq -r '.invoice.holder.lei' "$INVOICE_CONFIG_FILE")
INVOICE_DATA=$(jq -c '.invoice.sampleInvoice' "$INVOICE_CONFIG_FILE")

# Add issuer AID (self-attested: issuer = holder) and LEIs to invoice data
INVOICE_DATA=$(echo "$INVOICE_DATA" | jq -c \
  --arg issuerAid "$ISSUER_AID" \
  --arg sellerLEI "$SELLER_LEI" \
  --arg buyerLEI "$BUYER_LEI" \
  --arg dt "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  '. + {i: $issuerAid, sellerLEI: $sellerLEI, buyerLEI: $buyerLEI, dt: $dt}')

################################################################################
# Invoice Schema SAID - READ FROM CONFIG FILE (single source of truth)
################################################################################
if [ -f "./appconfig/schemaSaids.json" ]; then
    INVOICE_SCHEMA_SAID=$(jq -r '.invoiceSchema.said' "./appconfig/schemaSaids.json")
    if [ -z "$INVOICE_SCHEMA_SAID" ] || [ "$INVOICE_SCHEMA_SAID" = "null" ]; then
        echo -e "${RED}ERROR: Invoice schema SAID not found in schemaSaids.json${NC}"
        exit 1
    fi
    echo "  ✓ Loaded schema SAID from schemaSaids.json: $INVOICE_SCHEMA_SAID"
else
    echo -e "${RED}ERROR: ./appconfig/schemaSaids.json not found${NC}"
    echo "  Please run ./saidify-with-docker.sh first"
    exit 1
fi

OUTPUT_PATH="/task-data/${ISSUER_AGENT}-self-invoice-credential-info.json"

echo "  Issuer Agent AID: $ISSUER_AID"
echo "  Self-Attested: YES (issuer = issuee = ${ISSUER_AGENT})"
echo "  Registry: $REGISTRY_NAME"
echo "  Edge: NONE (no OOR chain)"
echo ""

docker compose exec -T tsx-shell tsx \
  sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only.ts \
  docker \
  "$ISSUER_AGENT" \
  "$ISSUER_BRAN" \
  "$REGISTRY_NAME" \
  "$INVOICE_SCHEMA_SAID" \
  "$INVOICE_DATA" \
  "$OUTPUT_PATH" \
  "/task-data"

echo ""
echo -e "${GREEN}✓ Self-attested invoice credential issued successfully${NC}"
echo "  Output: ./task-data/${ISSUER_AGENT}-self-invoice-credential-info.json"
echo "  Note: Credential stored in ${ISSUER_AGENT}'s KERIA"
echo "  Note: Use invoice-ipex-grant.sh to grant to another agent"
