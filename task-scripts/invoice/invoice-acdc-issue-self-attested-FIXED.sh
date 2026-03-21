#!/bin/bash
################################################################################
# invoice-acdc-issue-self-attested.sh
# Issue a self-attested invoice credential (issuer = issuee)
################################################################################

set -e

AGENT_ALIAS="${1:-jupiterSellerAgent}"
INVOICE_CONFIG="${2:-./appconfig/invoiceConfig.json}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Source environment variables
source ./task-scripts/workshop-env-vars.sh

echo "Issuing self-attested invoice credential..."
echo "  Issuer Agent: $AGENT_ALIAS"
echo ""

# Check if agent info exists
if [ ! -f "./task-data/${AGENT_ALIAS}-info.json" ]; then
    echo -e "${RED}ERROR: Agent info file not found: ./task-data/${AGENT_ALIAS}-info.json${NC}"
    exit 1
fi

# Check if BRAN file exists
BRAN_FILE="./task-data/${AGENT_ALIAS}-bran.txt"
if [ ! -f "$BRAN_FILE" ]; then
    echo -e "${RED}ERROR: Agent BRAN file not found: $BRAN_FILE${NC}"
    exit 1
fi

AGENT_BRAN=$(cat "$BRAN_FILE")
echo "Using agent's unique BRAN: ${AGENT_BRAN:0:20}..."

# Get agent AID
AGENT_AID=$(cat "./task-data/${AGENT_ALIAS}-info.json" | jq -r '.aid')
echo "  Issuer Agent AID: $AGENT_AID"
echo "  Self-Attested: YES (issuer = issuee = $AGENT_ALIAS)"
echo "  Registry: ${AGENT_ALIAS}_INVOICE_REGISTRY"
echo "  Edge: NONE (no OOR chain)"
echo ""

# Wait for KERIA to be healthy (important after registry creation!)
echo -e "${BLUE}→ Checking KERIA health...${NC}"
MAX_HEALTH_CHECKS=10
for i in $(seq 1 $MAX_HEALTH_CHECKS); do
    if docker compose exec -T keria wget --spider --tries=1 --no-verbose http://127.0.0.1:3902/spec.yaml 2>/dev/null; then
        echo -e "${GREEN}  ✓ KERIA is healthy${NC}"
        break
    else
        echo "  KERIA health check $i/$MAX_HEALTH_CHECKS failed, waiting..."
        sleep 3
        if [ $i -eq $MAX_HEALTH_CHECKS ]; then
            echo -e "${RED}ERROR: KERIA is not healthy after $MAX_HEALTH_CHECKS checks${NC}"
            docker compose logs keria --tail=20
            exit 1
        fi
    fi
done

# Add a small delay after registry creation for KERIA to settle
echo "  Waiting 5s for KERIA to settle after registry creation..."
sleep 5

# Get invoice data from config
SELLER_LEI=$(jq -r '.invoice.issuer.lei' "$INVOICE_CONFIG")
BUYER_LEI=$(jq -r '.invoice.holder.lei' "$INVOICE_CONFIG")
INVOICE_NUMBER=$(jq -r '.invoice.sampleInvoice.invoiceNumber' "$INVOICE_CONFIG")
TOTAL_AMOUNT=$(jq -r '.invoice.sampleInvoice.totalAmount' "$INVOICE_CONFIG")
CURRENCY=$(jq -r '.invoice.sampleInvoice.currency' "$INVOICE_CONFIG")
DUE_DATE=$(jq -r '.invoice.sampleInvoice.dueDate' "$INVOICE_CONFIG")
PAYMENT_METHOD=$(jq -r '.invoice.sampleInvoice.paymentMethod' "$INVOICE_CONFIG")
PAYMENT_CHAIN_ID=$(jq -r '.invoice.sampleInvoice.paymentChainID' "$INVOICE_CONFIG")
PAYMENT_WALLET=$(jq -r '.invoice.sampleInvoice.paymentWalletAddress' "$INVOICE_CONFIG")
REF_URI=$(jq -r '.invoice.sampleInvoice.ref_uri // ""' "$INVOICE_CONFIG")
PAYMENT_TERMS=$(jq -r '.invoice.sampleInvoice.paymentTerms // ""' "$INVOICE_CONFIG")

echo -e "${BLUE}Invoice Details:${NC}"
echo "    Invoice Number: $INVOICE_NUMBER"
echo "    Total Amount: $TOTAL_AMOUNT $CURRENCY"
echo "    Seller LEI: $SELLER_LEI"
echo "    Buyer LEI: $BUYER_LEI"
echo "    Due Date: $DUE_DATE"
echo "    Payment Method: $PAYMENT_METHOD"
echo ""

# Prepare invoice data JSON
DT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
INVOICE_DATA=$(jq -c --arg issuerAid "$AGENT_AID" \
  --arg sellerLEI "$SELLER_LEI" \
  --arg buyerLEI "$BUYER_LEI" \
  --arg dt "$DT" \
  '.invoice.sampleInvoice + {i: $issuerAid, sellerLEI: $sellerLEI, buyerLEI: $buyerLEI, dt: $dt}' "$INVOICE_CONFIG")

# Invoice Schema SAID - must match the $id in the SAIDified schema file
# Schema file: ./schemas/self-attested-invoice.json
INVOICE_SCHEMA_SAID="EEwSXh_s-i7NBmFrNSTJDC5K9Xw6W-YvEi-Cl9-JaAFb"

REGISTRY_NAME="${AGENT_ALIAS}_INVOICE_REGISTRY"
CRED_OUTPUT_PATH="/task-data/${AGENT_ALIAS}-self-invoice-credential-info.json"

echo -e "${BLUE}→ Issuing self-attested invoice credential...${NC}"

# Use the FIXED version with retry logic if it exists
SCRIPT_PATH="sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only.ts"
if [ -f "./sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only-FIXED.ts" ]; then
    SCRIPT_PATH="sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only-FIXED.ts"
    echo "  Using FIXED script with retry logic"
fi

docker compose exec -T tsx-shell tsx \
  "$SCRIPT_PATH" \
  docker \
  "$AGENT_ALIAS" \
  "" \
  "$REGISTRY_NAME" \
  "$INVOICE_SCHEMA_SAID" \
  "$INVOICE_DATA" \
  "$CRED_OUTPUT_PATH" \
  "/task-data"

# Verify credential was created
LOCAL_OUTPUT_PATH="./task-data/${AGENT_ALIAS}-self-invoice-credential-info.json"
if [ -f "$LOCAL_OUTPUT_PATH" ]; then
    CRED_SAID=$(cat "$LOCAL_OUTPUT_PATH" | jq -r '.said')
    echo ""
    echo -e "${GREEN}✓ Self-attested invoice credential issued${NC}"
    echo "    Credential SAID: $CRED_SAID"
    echo "    Output: $LOCAL_OUTPUT_PATH"
else
    echo -e "${YELLOW}Note: Credential output file may be inside container at $CRED_OUTPUT_PATH${NC}"
fi
