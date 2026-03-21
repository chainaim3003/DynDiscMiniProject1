#!/bin/bash
################################################################################
# saidify-and-restart.sh - SAIDify Schema and Restart Container
#
# PURPOSE: Convenience script to SAIDify the invoice schema and restart
#          the schema container in one step.
#
# WHAT IT DOES:
#   1. Runs ./saidify-with-docker.sh to compute schema SAID using keripy
#   2. Restarts the schema container to load the new schema
#   3. Waits for the container to be ready
#
# USAGE:
#   ./saidify-and-restart.sh
#
# PREREQUISITES:
#   - Docker must be running
#   - ./deploy.sh must have been run first (containers must be up)
#
# DATE: December 1, 2025
# VERSION: 1.0
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
echo -e "${CYAN}║         SAIDify Schema and Restart Container                 ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

################################################################################
# PRE-FLIGHT CHECKS
################################################################################
echo -e "${YELLOW}[0/3] Pre-flight checks...${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Docker is not running${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Docker is running${NC}"

# Check if schema container exists
if ! docker compose ps schema 2>/dev/null | grep -q "Up\|running"; then
    echo -e "${RED}ERROR: Schema container is not running${NC}"
    echo -e "${YELLOW}  Please run ./deploy.sh first to start the containers${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Schema container is running${NC}"

# Check if saidify script exists
if [ ! -f "./saidify-with-docker.sh" ]; then
    echo -e "${RED}ERROR: ./saidify-with-docker.sh not found${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ saidify-with-docker.sh exists${NC}"

echo ""

################################################################################
# STEP 1: SAIDify the Schema
################################################################################
echo -e "${YELLOW}[1/3] SAIDifying invoice schema...${NC}"
echo ""

chmod +x ./saidify-with-docker.sh
./saidify-with-docker.sh

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: SAIDification failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Schema SAIDified successfully${NC}"
echo ""

################################################################################
# STEP 2: Restart Schema Container
################################################################################
echo -e "${YELLOW}[2/3] Restarting schema container...${NC}"

docker compose restart schema

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to restart schema container${NC}"
    exit 1
fi

echo -e "${GREEN}  ✓ Schema container restarted${NC}"
echo ""

################################################################################
# STEP 3: Wait for Schema Container to be Ready
################################################################################
echo -e "${YELLOW}[3/3] Waiting for schema container to be ready...${NC}"

# Wait 15 seconds for container to fully start
echo -e "${BLUE}  Waiting 15 seconds for container to initialize...${NC}"
sleep 15

# Verify schema is accessible
if [ -f "./appconfig/schemaSaids.json" ]; then
    SCHEMA_SAID=$(jq -r '.invoiceSchema.said' "./appconfig/schemaSaids.json" 2>/dev/null)
    if [ -n "$SCHEMA_SAID" ] && [ "$SCHEMA_SAID" != "null" ]; then
        echo -e "${BLUE}  Verifying schema accessibility...${NC}"
        
        if curl -sf "http://127.0.0.1:7723/oobi/${SCHEMA_SAID}" > /dev/null 2>&1; then
            echo -e "${GREEN}  ✓ Invoice schema is accessible${NC}"
            echo -e "${GREEN}    SAID: ${SCHEMA_SAID}${NC}"
        else
            echo -e "${YELLOW}  ⚠ Schema may not be immediately accessible${NC}"
            echo -e "${YELLOW}    This is usually fine - KERIA will resolve it on first use${NC}"
        fi
    fi
fi

echo ""

################################################################################
# COMPLETION
################################################################################
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              ✓ SAIDIFY AND RESTART COMPLETE                  ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}Schema is ready for use!${NC}"
echo ""

echo -e "${YELLOW}Next step:${NC}"
echo "  ./run-all-buyerseller-4C-with-agents.sh"
echo ""
