#!/bin/bash
################################################################################
# diagnose-schema-issue.sh - Diagnose why invoice schema OOBI resolution fails
#
# Run this script when invoice credential issuance fails with "fetch failed"
################################################################################

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

INVOICE_SCHEMA_SAID="EHFQviJgaf6YTHbMPLheau8H1s3v31-ByTER_D49DLHY"

echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Schema Issue Diagnostic Tool${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""

################################################################################
# Step 1: Check container status
################################################################################

echo -e "${YELLOW}[1/6] Checking container status...${NC}"

echo "Schema container:"
docker compose ps schema 2>/dev/null || docker-compose ps schema 2>/dev/null || echo "  ✗ Could not check schema container"

echo ""
echo "KERIA container:"
docker compose ps keria 2>/dev/null || docker-compose ps keria 2>/dev/null || echo "  ✗ Could not check keria container"

echo ""

################################################################################
# Step 2: Check schema container logs
################################################################################

echo -e "${YELLOW}[2/6] Checking schema container logs (last 20 lines)...${NC}"
docker compose logs schema --tail=20 2>/dev/null || docker-compose logs schema --tail=20 2>/dev/null || echo "  ✗ Could not get logs"
echo ""

################################################################################
# Step 3: Check if schema files are mounted
################################################################################

echo -e "${YELLOW}[3/6] Checking schema files in container...${NC}"

echo "Custom schemas mounted at /vLEI/custom-schema/:"
docker compose exec -T schema ls -la /vLEI/custom-schema/ 2>/dev/null || echo "  ✗ Could not list custom-schema"

echo ""
echo "Schemas in /vLEI/schema/ (where server reads from):"
docker compose exec -T schema ls -la /vLEI/schema/ 2>/dev/null | grep -E "invoice|${INVOICE_SCHEMA_SAID}" || echo "  ✗ Invoice schema not found in /vLEI/schema/"

echo ""

################################################################################
# Step 4: Test schema endpoint from host
################################################################################

echo -e "${YELLOW}[4/6] Testing schema endpoint from host...${NC}"

echo "Testing: curl http://localhost:7723/oobi/${INVOICE_SCHEMA_SAID}"
if curl -s -f "http://localhost:7723/oobi/${INVOICE_SCHEMA_SAID}" | head -c 100; then
    echo ""
    echo -e "${GREEN}  ✓ Schema accessible from host${NC}"
else
    echo -e "${RED}  ✗ Schema NOT accessible from host${NC}"
fi
echo ""

################################################################################
# Step 5: Test schema endpoint from KERIA container
################################################################################

echo -e "${YELLOW}[5/6] Testing schema endpoint from KERIA container (this is the critical test)...${NC}"

echo "Testing: docker compose exec keria wget -qO- http://schema:7723/oobi/${INVOICE_SCHEMA_SAID}"
if docker compose exec -T keria wget -qO- "http://schema:7723/oobi/${INVOICE_SCHEMA_SAID}" 2>/dev/null | head -c 100; then
    echo ""
    echo -e "${GREEN}  ✓ Schema accessible from KERIA container${NC}"
else
    echo -e "${RED}  ✗ Schema NOT accessible from KERIA container${NC}"
    echo ""
    echo "  This is the ROOT CAUSE of your issue!"
    echo "  KERIA cannot reach the schema container."
fi
echo ""

################################################################################
# Step 6: Test Docker network DNS
################################################################################

echo -e "${YELLOW}[6/6] Testing Docker network DNS...${NC}"

echo "Testing DNS resolution of 'schema' hostname from KERIA:"
if docker compose exec -T keria ping -c 1 schema 2>/dev/null; then
    echo -e "${GREEN}  ✓ DNS resolution works${NC}"
else
    echo -e "${RED}  ✗ DNS resolution FAILED${NC}"
    echo "  Docker network DNS may be broken"
fi
echo ""

################################################################################
# Recommendations
################################################################################

echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  RECOMMENDATIONS${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${YELLOW}If schema is not accessible from KERIA, try these fixes:${NC}"
echo ""
echo "1. Restart schema container:"
echo "   docker compose restart schema"
echo "   sleep 15"
echo ""
echo "2. Verify schema file has correct SAID:"
echo "   cat ./schemas/self-attested-invoice.json | jq '.\"\$id\"'"
echo "   # Should show: ${INVOICE_SCHEMA_SAID}"
echo ""
echo "3. Manually copy schema into container:"
echo "   docker compose exec schema cp /vLEI/custom-schema/self-attested-invoice.json /vLEI/schema/"
echo "   docker compose exec schema cp /vLEI/custom-schema/${INVOICE_SCHEMA_SAID}.json /vLEI/schema/"
echo ""
echo "4. Rebuild and restart all containers:"
echo "   docker compose down"
echo "   docker compose up -d"
echo "   sleep 30"
echo ""
echo "5. Check Docker network:"
echo "   docker network inspect vlei_workshop"
echo ""

echo -e "${GREEN}Script completed.${NC}"
