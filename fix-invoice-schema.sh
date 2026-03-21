#!/bin/bash
################################################################################
# fix-invoice-schema.sh
#
# This script ensures the invoice schema is properly set up for credential issuance:
# 1. Verifies the schema file exists with correct SAID
# 2. Restarts the schema container to load the schema
# 3. Tests that the schema is accessible via OOBI
#
# Run this BEFORE running the 4C workflow with invoice credentials
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

cd "$(dirname "$0")"

# The correct SAID from the self-attested-invoice.json schema
INVOICE_SCHEMA_SAID="EHFQviJgaf6YTHbMPLheau8H1s3v31-ByTER_D49DLHY"
SCHEMA_FILE="./schemas/self-attested-invoice.json"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         INVOICE SCHEMA FIX SCRIPT                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

################################################################################
# Step 1: Verify Schema File
################################################################################

echo -e "${YELLOW}[1/4] Checking schema file...${NC}"

if [ ! -f "$SCHEMA_FILE" ]; then
    echo -e "${RED}ERROR: Schema file not found: $SCHEMA_FILE${NC}"
    exit 1
fi

# Check that the SAID in the file matches expected
FILE_SAID=$(jq -r '.["$id"]' "$SCHEMA_FILE")
if [ "$FILE_SAID" != "$INVOICE_SCHEMA_SAID" ]; then
    echo -e "${YELLOW}WARNING: Schema SAID mismatch!${NC}"
    echo "  Expected: $INVOICE_SCHEMA_SAID"
    echo "  Found:    $FILE_SAID"
    echo ""
    echo "Updating schema file with correct SAID..."
    
    # Backup original
    cp "$SCHEMA_FILE" "${SCHEMA_FILE}.bak"
    
    # Update the SAID
    jq --arg said "$INVOICE_SCHEMA_SAID" '.["$id"] = $said' "${SCHEMA_FILE}.bak" > "$SCHEMA_FILE"
    
    echo -e "${GREEN}✓ Schema file updated${NC}"
else
    echo -e "${GREEN}✓ Schema file SAID is correct: $INVOICE_SCHEMA_SAID${NC}"
fi

################################################################################
# Step 2: Restart Schema Container
################################################################################

echo ""
echo -e "${YELLOW}[2/4] Restarting schema container...${NC}"

# Check if docker compose is running
if ! docker compose ps | grep -q "schema"; then
    echo -e "${RED}ERROR: Schema container not found. Is docker compose running?${NC}"
    echo "  Run: docker compose up -d"
    exit 1
fi

docker compose restart schema

echo "Waiting for schema container to be healthy..."
MAX_WAIT=30
for i in $(seq 1 $MAX_WAIT); do
    if docker compose exec -T schema curl -sf http://127.0.0.1:7723/oobi/EBfdlu8R27Fbx-ehrqwImnK-8Cm79sqbAQ4MmvEAYqao > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Schema container is healthy${NC}"
        break
    fi
    if [ $i -eq $MAX_WAIT ]; then
        echo -e "${RED}ERROR: Schema container did not become healthy in ${MAX_WAIT}s${NC}"
        docker compose logs schema --tail=20
        exit 1
    fi
    sleep 1
    echo -n "."
done
echo ""

################################################################################
# Step 3: Test Schema OOBI Access
################################################################################

echo -e "${YELLOW}[3/4] Testing schema OOBI access...${NC}"

# Test access to the invoice schema OOBI
SCHEMA_OOBI="http://127.0.0.1:7723/oobi/$INVOICE_SCHEMA_SAID"

echo "Testing: $SCHEMA_OOBI"

# Try external access first
if curl -sf --max-time 5 "$SCHEMA_OOBI" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Invoice schema accessible via host${NC}"
else
    echo -e "${YELLOW}  Host access failed, testing from container...${NC}"
    
    # Test from inside the container
    INTERNAL_OOBI="http://127.0.0.1:7723/oobi/$INVOICE_SCHEMA_SAID"
    if docker compose exec -T schema curl -sf "$INTERNAL_OOBI" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Invoice schema accessible from container${NC}"
    else
        echo -e "${RED}✗ Invoice schema NOT accessible!${NC}"
        echo ""
        echo "The vLEI server may not be loading custom schemas correctly."
        echo "Checking available schemas..."
        echo ""
        
        # List schemas that ARE available
        echo "Standard vLEI schemas:"
        docker compose exec -T schema curl -sf "http://127.0.0.1:7723/oobi/EBfdlu8R27Fbx-ehrqwImnK-8Cm79sqbAQ4MmvEAYqao" > /dev/null 2>&1 && echo "  ✓ QVI Schema" || echo "  ✗ QVI Schema"
        docker compose exec -T schema curl -sf "http://127.0.0.1:7723/oobi/ENPXp1vQzRF6JwIuS-mp2U8Uf1MoADoP_GqQ62VsDZWY" > /dev/null 2>&1 && echo "  ✓ LE Schema" || echo "  ✗ LE Schema"
        docker compose exec -T schema curl -sf "http://127.0.0.1:7723/oobi/EBNaNu-M9P5cgrnfl2Fvymy4E_jvxxyjb70PRtiANlJy" > /dev/null 2>&1 && echo "  ✓ OOR Schema" || echo "  ✗ OOR Schema"
        
        echo ""
        echo "The schema directory may not be mounted correctly."
        echo "Checking docker-compose volume mount..."
        docker compose exec -T schema ls -la /vLEI/custom-schema/ 2>/dev/null || echo "  ✗ /vLEI/custom-schema not accessible"
        
        echo ""
        echo "WORKAROUND: Use alternative approach below"
        exit 1
    fi
fi

################################################################################
# Step 4: Summary
################################################################################

echo ""
echo -e "${YELLOW}[4/4] Summary${NC}"
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Invoice Schema Setup Complete!${NC}"
echo ""
echo -e "  Schema File: ${YELLOW}$SCHEMA_FILE${NC}"
echo -e "  Schema SAID: ${YELLOW}$INVOICE_SCHEMA_SAID${NC}"
echo -e "  Schema OOBI: ${YELLOW}http://schema:7723/oobi/$INVOICE_SCHEMA_SAID${NC}"
echo ""
echo -e "The invoice credential workflow should now work correctly."
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Run the 4C workflow: ./run-all-buyerseller-4C-with-agents.sh"
echo "  2. Or run just the invoice part manually:"
echo "     ./task-scripts/invoice/invoice-registry-create-agent.sh jupiterSellerAgent"
echo "     ./task-scripts/invoice/invoice-acdc-issue-self-attested.sh jupiterSellerAgent"
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
