#!/bin/bash
################################################################################
# stop.sh - vLEI Workshop Environment Teardown
#
# PURPOSE: Completely stop and clean up the vLEI environment
#
# WHAT IT DOES:
#   1. Removes all task data files (credentials, AIDs, etc.)
#   2. Stops and removes all Docker containers
#   3. Removes Docker volumes (KERIA cache, etc.)
#   4. Removes the Docker network
#
# USAGE:
#   ./stop.sh
#
# IMPORTANT:
#   This completely resets the environment. All AIDs, credentials,
#   and cached data will be lost. Use this before a fresh start.
#
# DATE: November 30, 2025
# VERSION: 2.0
################################################################################

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║         Stopping vLEI Workshop Environment                   ║${NC}"
echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

################################################################################
# STEP 1: Clean Task Data
################################################################################
echo -e "${BLUE}[1/4] Cleaning task data...${NC}"

if [ -d "./task-data" ]; then
    rm -fv task-data/*.json 2>/dev/null || true
    rm -fv task-data/*.txt 2>/dev/null || true
    echo -e "${GREEN}  ✓ Task data cleaned${NC}"
else
    echo -e "${YELLOW}  - No task-data directory found${NC}"
fi
echo ""

################################################################################
# STEP 2: Check Docker Availability
################################################################################
echo -e "${BLUE}[2/4] Checking Docker...${NC}"

if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}  ✗ docker compose not found${NC}"
    echo -e "${YELLOW}  Please install docker compose first.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Docker compose available${NC}"
echo ""

################################################################################
# STEP 3: Stop and Remove Containers
################################################################################
echo -e "${BLUE}[3/4] Stopping and removing containers...${NC}"

# Stop all containers and remove volumes
docker compose down --remove-orphans --volumes 2>/dev/null || true

echo -e "${GREEN}  ✓ Containers stopped and removed${NC}"
echo -e "${GREEN}  ✓ Volumes removed (including KERIA cache)${NC}"
echo ""

################################################################################
# STEP 4: Remove Docker Network
################################################################################
echo -e "${BLUE}[4/4] Removing Docker network...${NC}"

docker network rm vlei_workshop 2>/dev/null && \
    echo -e "${GREEN}  ✓ Network vlei_workshop removed${NC}" || \
    echo -e "${YELLOW}  - Network already removed or doesn't exist${NC}"
echo ""

################################################################################
# COMPLETION
################################################################################
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              ✓ ENVIRONMENT STOPPED                           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}What was cleaned:${NC}"
echo "  • All Docker containers"
echo "  • All Docker volumes (KERIA cache cleared)"
echo "  • Docker network (vlei_workshop)"
echo "  • Task data files (*.json, *.txt)"
echo ""
echo -e "${YELLOW}To start fresh:${NC}"
echo "  1. ./setup.sh    # Prepare environment & build Docker images"
echo "  2. ./deploy.sh   # Start all services"
echo "  3. ./run-all-buyerseller-4C-with-agents.sh  # Run workflow"
echo ""
