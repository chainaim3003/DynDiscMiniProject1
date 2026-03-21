#!/bin/bash
################################################################################
# deploy.sh - vLEI Workshop Infrastructure Deployment
#
# PURPOSE: Start all vLEI services and verify they are healthy
#
# SERVICES STARTED:
#   • Schema server (vLEI schemas)     - Port 7723
#   • Witnesses (6x)                   - Ports 5642-5647
#   • KERIA (agent server)             - Ports 3901-3903
#   • Verifier (Sally)                 - Port 9723
#   • Webhook                          - Port 9923
#   • tsx-shell (TypeScript runner)    - No port (utility)
#   • vlei-verification                - Port 9724
#
# USAGE:
#   ./deploy.sh
#
# PREREQUISITES:
#   Run ./setup.sh first to:
#   - Fix line endings
#   - Build Docker images
#   - Verify schema configuration
#
# DATE: November 30, 2025
# VERSION: 2.0 - Added tsx-shell verification
################################################################################

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         vLEI Workshop Infrastructure Deployment              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

################################################################################
# PRE-FLIGHT CHECKS
################################################################################
echo -e "${YELLOW}[0/5] Pre-flight checks...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}  ✗ Docker not found. Please install Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Docker available${NC}"

# Check docker compose
if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}  ✗ docker compose not found. Please install docker compose first.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ docker compose available${NC}"

# Check if tsx-shell image exists (should be built by setup.sh)
if ! docker images | grep -q "gleif/wkshp-tsx-shell"; then
    echo -e "${YELLOW}  ⚠ tsx-shell image not found - building now...${NC}"
    docker compose build tsx-shell
fi
echo -e "${GREEN}  ✓ tsx-shell image ready${NC}"

echo ""

################################################################################
# STEP 1: Create Docker Network
################################################################################
echo -e "${YELLOW}[1/5] Creating Docker network...${NC}"

docker network create vlei_workshop --driver bridge 2>/dev/null && \
    echo -e "${GREEN}  ✓ Network vlei_workshop created${NC}" || \
    echo -e "${YELLOW}  - Network vlei_workshop already exists${NC}"
echo ""

################################################################################
# STEP 2: Clean Up Existing Containers
################################################################################
echo -e "${YELLOW}[2/5] Cleaning up existing containers...${NC}"

docker compose down --remove-orphans --volumes 2>/dev/null || true
echo -e "${GREEN}  ✓ Existing containers removed${NC}"
echo ""

################################################################################
# STEP 3: Start Services
################################################################################
echo -e "${YELLOW}[3/5] Starting services...${NC}"

docker compose up --wait -d

echo -e "${GREEN}  ✓ All services started${NC}"
echo ""

################################################################################
# STEP 4: Health Checks
################################################################################
echo -e "${YELLOW}[4/5] Running health checks...${NC}"
echo ""

# Check schema server
echo -e "${BLUE}  Schema Server:${NC}"
if curl -sf http://127.0.0.1:7723/oobi/EBfdlu8R27Fbx-ehrqwImnK-8Cm79sqbAQ4MmvEAYqao > /dev/null 2>&1; then
    echo -e "${GREEN}    ✓ Schema server healthy${NC}"
else
    echo -e "${RED}    ✗ Schema server not responding${NC}"
fi

# Check invoice schema
if [ -f "./appconfig/schemaSaids.json" ]; then
    INVOICE_SCHEMA_SAID=$(grep -o '"said": *"[^"]*"' ./appconfig/schemaSaids.json | head -1 | sed 's/.*"\([^"]*\)"/\1/')
    if curl -sf "http://127.0.0.1:7723/oobi/${INVOICE_SCHEMA_SAID}" > /dev/null 2>&1; then
        echo -e "${GREEN}    ✓ Invoice schema accessible (${INVOICE_SCHEMA_SAID})${NC}"
    else
        echo -e "${YELLOW}    ⚠ Invoice schema may need loading${NC}"
    fi
fi

# Check witnesses
echo -e "${BLUE}  Witnesses:${NC}"
WITNESS_PORTS=(5642 5643 5644 5645 5646 5647)
WITNESS_NAMES=(wan wil wes wit wub wyz)
for i in ${!WITNESS_PORTS[@]}; do
    if curl -sf "http://127.0.0.1:${WITNESS_PORTS[$i]}/oobi" > /dev/null 2>&1; then
        echo -e "${GREEN}    ✓ Witness ${WITNESS_NAMES[$i]} (port ${WITNESS_PORTS[$i]})${NC}"
    else
        echo -e "${RED}    ✗ Witness ${WITNESS_NAMES[$i]} not responding${NC}"
    fi
done

# Check KERIA
echo -e "${BLUE}  KERIA:${NC}"
if curl -sf http://127.0.0.1:3902/spec.yaml > /dev/null 2>&1; then
    echo -e "${GREEN}    ✓ KERIA healthy${NC}"
else
    echo -e "${RED}    ✗ KERIA not responding${NC}"
fi

# Check verifier
echo -e "${BLUE}  Verifier (Sally):${NC}"
if curl -sf http://127.0.0.1:9723/health > /dev/null 2>&1; then
    echo -e "${GREEN}    ✓ Verifier healthy${NC}"
else
    echo -e "${RED}    ✗ Verifier not responding${NC}"
fi

# Check webhook
echo -e "${BLUE}  Webhook:${NC}"
if curl -sf http://127.0.0.1:9923/health > /dev/null 2>&1; then
    echo -e "${GREEN}    ✓ Webhook healthy${NC}"
else
    echo -e "${RED}    ✗ Webhook not responding${NC}"
fi

# Check tsx-shell
echo -e "${BLUE}  tsx-shell:${NC}"
if docker compose ps tsx-shell 2>/dev/null | grep -q "running"; then
    echo -e "${GREEN}    ✓ tsx-shell running${NC}"
    
    # Verify TypeScript files are mounted
    if docker compose exec -T tsx-shell ls /vlei/sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only.ts > /dev/null 2>&1; then
        echo -e "${GREEN}    ✓ TypeScript files mounted correctly${NC}"
    else
        echo -e "${RED}    ✗ TypeScript files not mounted - check docker-compose.yml${NC}"
    fi
else
    echo -e "${RED}    ✗ tsx-shell not running${NC}"
fi

echo ""

################################################################################
# STEP 5: Summary
################################################################################
echo -e "${YELLOW}[5/5] Deployment summary...${NC}"
echo ""

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              ✓ DEPLOYMENT COMPLETE                           ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}Service URLs:${NC}"
echo "  Schema server:    http://127.0.0.1:7723"
echo "  Witnesses:        http://127.0.0.1:5642 - 5647"
echo "  KERIA Admin:      http://127.0.0.1:3901"
echo "  KERIA HTTP:       http://127.0.0.1:3902"
echo "  KERIA Boot:       http://127.0.0.1:3903"
echo "  Verifier:         http://127.0.0.1:9723"
echo "  Webhook:          http://127.0.0.1:9923"
echo ""

echo -e "${YELLOW}Invoice Schema:${NC}"
if [ -n "$INVOICE_SCHEMA_SAID" ]; then
    echo "  SAID: $INVOICE_SCHEMA_SAID"
    echo "  OOBI: http://127.0.0.1:7723/oobi/$INVOICE_SCHEMA_SAID"
fi
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo "  Run the vLEI workflow:"
echo "    ./run-all-buyerseller-4C-with-agents.sh"
echo ""
echo "  Or run individual test:"
echo "    ./test-agent-verification.sh"
echo ""

echo -e "${YELLOW}To stop:${NC}"
echo "    ./stop.sh"
echo ""
