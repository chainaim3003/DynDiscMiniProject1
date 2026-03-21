#!/bin/bash
################################################################################
# invoice-self-attested-to-agent.sh
#
# Issues a SELF-ATTESTED invoice credential to a delegated agent.
# The agent is BOTH the issuer AND the issuee.
#
# Usage:
#   ./invoice-self-attested-to-agent.sh <agent_name> <registry_name> <invoice_json>
#
# Example:
#   ./invoice-self-attested-to-agent.sh jupiterSellerAgent jupiterSellerAgent-registry '{"invoiceNumber":"INV-001",...}'
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Load environment
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../workshop-env-vars.sh" 2>/dev/null || true

# Default values
ENV="${ENV:-docker}"
TASK_DATA_DIR="${TASK_DATA_DIR:-/task-data}"
INVOICE_SCHEMA_SAID="${INVOICE_SCHEMA_SAID:-EHFQviJgaf6YTHbMPLheau8H1s3v31-ByTER_D49DLHY}"

# Arguments
AGENT_NAME="${1:-jupiterSellerAgent}"
REGISTRY_NAME="${2:-${AGENT_NAME}-registry}"
INVOICE_DATA_JSON="${3}"
OUTPUT_PATH="${4:-${TASK_DATA_DIR}/invoice-credential-info.json}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   SELF-ATTESTED INVOICE CREDENTIAL TO DELEGATED AGENT     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Validate arguments
if [ -z "$INVOICE_DATA_JSON" ]; then
    echo -e "${YELLOW}No invoice data provided. Using default test data...${NC}"
    INVOICE_DATA_JSON='{
        "invoiceNumber": "INV-SELF-001",
        "invoiceDate": "2025-01-15",
        "dueDate": "2025-02-15",
        "sellerLEI": "5493001KJTIIGC8Y1R17",
        "buyerLEI": "254900OPPU84GM83MG36",
        "currency": "USD",
        "totalAmount": 5000.00,
        "paymentMethod": "USDC",
        "paymentChainID": "stellar:testnet",
        "paymentWalletAddress": "GBXYZ...",
        "ref_uri": "ipfs://QmTest...",
        "paymentTerms": "Net 30"
    }'
fi

echo "Configuration:"
echo "  Agent Name: $AGENT_NAME"
echo "  Registry: $REGISTRY_NAME"
echo "  Schema SAID: $INVOICE_SCHEMA_SAID"
echo "  Output: $OUTPUT_PATH"
echo ""

# Check if BRAN file exists
BRAN_FILE="${TASK_DATA_DIR}/${AGENT_NAME}-bran.txt"
if [ ! -f "$BRAN_FILE" ]; then
    echo -e "${RED}ERROR: Agent BRAN file not found: $BRAN_FILE${NC}"
    echo "  The agent must be created first with a unique BRAN."
    echo "  Run the agent creation step before issuing credentials."
    exit 1
fi
echo -e "${GREEN}✓ Agent BRAN file found${NC}"

# Run the TypeScript
echo ""
echo -e "${YELLOW}Issuing self-attested credential...${NC}"

docker compose exec -T tsx-shell npx tsx \
    /vlei/sig-wallet/src/tasks/invoice/invoice-self-attested-to-agent.ts \
    "$ENV" \
    "$AGENT_NAME" \
    "" \
    "$REGISTRY_NAME" \
    "$INVOICE_SCHEMA_SAID" \
    "$INVOICE_DATA_JSON" \
    "$OUTPUT_PATH" \
    "$TASK_DATA_DIR"

# Check result
if [ -f "$OUTPUT_PATH" ] || docker compose exec -T tsx-shell test -f "$OUTPUT_PATH" 2>/dev/null; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              SELF-ATTESTED CREDENTIAL ISSUED               ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    
    # Try to show the credential info
    if [ -f "$OUTPUT_PATH" ]; then
        CRED_SAID=$(jq -r '.said' "$OUTPUT_PATH" 2>/dev/null || echo "unknown")
        ISSUER=$(jq -r '.issuer' "$OUTPUT_PATH" 2>/dev/null || echo "unknown")
        ISSUEE=$(jq -r '.issuee' "$OUTPUT_PATH" 2>/dev/null || echo "unknown")
        SELF_ATTESTED=$(jq -r '.selfAttested' "$OUTPUT_PATH" 2>/dev/null || echo "unknown")
    else
        CRED_SAID=$(docker compose exec -T tsx-shell cat "$OUTPUT_PATH" 2>/dev/null | jq -r '.said' || echo "unknown")
        ISSUER=$(docker compose exec -T tsx-shell cat "$OUTPUT_PATH" 2>/dev/null | jq -r '.issuer' || echo "unknown")
        ISSUEE=$(docker compose exec -T tsx-shell cat "$OUTPUT_PATH" 2>/dev/null | jq -r '.issuee' || echo "unknown")
        SELF_ATTESTED=$(docker compose exec -T tsx-shell cat "$OUTPUT_PATH" 2>/dev/null | jq -r '.selfAttested' || echo "unknown")
    fi
    
    echo ""
    echo "  Credential SAID: $CRED_SAID"
    echo "  Issuer: $ISSUER"
    echo "  Issuee: $ISSUEE"
    echo "  Self-Attested: $SELF_ATTESTED"
    echo ""
    echo "  The agent ($AGENT_NAME) now holds a self-attested invoice credential."
    echo "  Trust derives from the delegation chain to GLEIF root."
else
    echo -e "${RED}ERROR: Credential issuance may have failed${NC}"
    echo "  Check the output above for errors."
    exit 1
fi
