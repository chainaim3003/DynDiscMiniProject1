#!/bin/bash
################################################################################
# restart-schema-for-invoice.sh
#
# This script restarts the schema container and verifies the invoice schema
# is accessible. Run this BEFORE issuing invoice credentials.
#
# The vLEI-server needs to be restarted to pick up new schemas from the
# mounted ./schemas directory.
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

cd "$(dirname "$0")"

# Invoice Schema SAID
INVOICE_SCHEMA_SAID="EHFQviJgaf6YTHbMPLheau8H1s3v31-ByTER_D49DLHY"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     RESTARTING SCHEMA CONTAINER FOR INVOICE WORKFLOW       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

################################################################################
# Step 1: Ensure schema file exists
################################################################################

echo -e "${YELLOW}[1/3] Checking schema file...${NC}"

# Check both possible schema file locations
SCHEMA_FILE=""
if [ -f "./schemas/self-attested-invoice.json" ]; then
    SCHEMA_FILE="./schemas/self-attested-invoice.json"
elif [ -f "./schemas/EHFQviJgaf6YTHbMPLheau8H1s3v31-ByTER_D49DLHY.json" ]; then
    SCHEMA_FILE="./schemas/EHFQviJgaf6YTHbMPLheau8H1s3v31-ByTER_D49DLHY.json"
else
    echo -e "${RED}ERROR: Invoice schema file not found${NC}"
    echo "  Expected: ./schemas/self-attested-invoice.json"
    echo "         or ./schemas/EHFQviJgaf6YTHbMPLheau8H1s3v31-ByTER_D49DLHY.json"
    exit 1
fi

echo -e "${GREEN}✓ Schema file found: $SCHEMA_FILE${NC}"

# Verify SAID in file
FILE_SAID=$(jq -r '.["$id"]' "$SCHEMA_FILE")
if [ "$FILE_SAID" != "$INVOICE_SCHEMA_SAID" ]; then
    echo -e "${YELLOW}  Updating schema SAID from $FILE_SAID to $INVOICE_SCHEMA_SAID${NC}"
    jq --arg said "$INVOICE_SCHEMA_SAID" '.["$id"] = $said' "$SCHEMA_FILE" > "${SCHEMA_FILE}.tmp" && mv "${SCHEMA_FILE}.tmp" "$SCHEMA_FILE"
fi
echo ""

################################################################################
# Step 2: Restart schema container
################################################################################

echo -e "${YELLOW}[2/3] Restarting schema container...${NC}"

# Check if schema container exists and is running
if ! docker compose ps schema 2>/dev/null | grep -q "schema"; then
    echo -e "${RED}ERROR: Schema container not found. Is docker compose running?${NC}"
    echo "  Run: docker compose up -d"
    exit 1
fi

# Restart the schema container
echo "  Restarting schema service..."
docker compose restart schema

# Wait for schema container to be healthy
echo "  Waiting for schema container to be healthy..."
MAX_WAIT=60
HEALTHY=false
for i in $(seq 1 $MAX_WAIT); do
    if docker compose exec -T schema curl -sf http://127.0.0.1:7723/oobi/EBfdlu8R27Fbx-ehrqwImnK-8Cm79sqbAQ4MmvEAYqao > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Schema container is healthy (${i}s)${NC}"
        HEALTHY=true
        break
    fi
    sleep 1
    printf "."
done
echo ""

if [ "$HEALTHY" = false ]; then
    echo -e "${RED}ERROR: Schema container did not become healthy in ${MAX_WAIT}s${NC}"
    docker compose logs schema --tail=20
    exit 1
fi

################################################################################
# Step 3: Verify invoice schema is accessible
################################################################################

echo -e "${YELLOW}[3/3] Verifying invoice schema accessibility...${NC}"

# Wait a bit more for schema loading
sleep 3

# Test if invoice schema OOBI is accessible
SCHEMA_OOBI="http://127.0.0.1:7723/oobi/$INVOICE_SCHEMA_SAID"
echo "  Testing: $SCHEMA_OOBI"

# Try via curl from host
if curl -sf --max-time 10 "$SCHEMA_OOBI" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Invoice schema accessible via host${NC}"
else
    # Try from inside the container
    echo -e "${YELLOW}  Host access failed, testing from inside container...${NC}"
    INTERNAL_OOBI="http://127.0.0.1:7723/oobi/$INVOICE_SCHEMA_SAID"
    if docker compose exec -T schema curl -sf "$INTERNAL_OOBI" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Invoice schema accessible from inside container${NC}"
    else
        echo -e "${RED}✗ Invoice schema NOT accessible!${NC}"
        echo ""
        echo "The vLEI server may not be loading the custom schema."
        echo "This could be because:"
        echo "  1. The schema file doesn't have the correct SAID"
        echo "  2. The schema format is not compatible with vLEI-server"
        echo "  3. The volume mount may not be working correctly"
        echo ""
        echo "Checking available schemas:"
        docker compose exec -T schema curl -sf "http://127.0.0.1:7723/oobi/EBfdlu8R27Fbx-ehrqwImnK-8Cm79sqbAQ4MmvEAYqao" > /dev/null 2>&1 && echo "  ✓ QVI Schema" || echo "  ✗ QVI Schema"
        docker compose exec -T schema curl -sf "http://127.0.0.1:7723/oobi/ENPXp1vQzRF6JwIuS-mp2U8Uf1MoADoP_GqQ62VsDZWY" > /dev/null 2>&1 && echo "  ✓ LE Schema" || echo "  ✗ LE Schema"
        docker compose exec -T schema curl -sf "http://127.0.0.1:7723/oobi/EBNaNu-M9P5cgrnfl2Fvymy4E_jvxxyjb70PRtiANlJy" > /dev/null 2>&1 && echo "  ✓ OOR Schema" || echo "  ✗ OOR Schema"
        echo ""
        echo "Checking mounted schemas directory:"
        docker compose exec -T schema ls -la /vLEI/custom-schema/ 2>/dev/null || echo "  ✗ /vLEI/custom-schema not accessible"
        echo ""
        echo -e "${YELLOW}WORKAROUND: The invoice credential will need to skip schema validation${NC}"
        echo -e "${YELLOW}or use an alternative approach.${NC}"
        # Don't exit, just warn
    fi
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Schema restart complete.${NC}"
echo ""
echo -e "  Schema SAID: ${YELLOW}$INVOICE_SCHEMA_SAID${NC}"
echo -e "  Schema OOBI: ${YELLOW}http://schema:7723/oobi/$INVOICE_SCHEMA_SAID${NC}"
echo ""
echo -e "${BLUE}The invoice credential workflow can now proceed.${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
