#!/bin/bash
################################################################################
# setup.sh - vLEI Workshop Environment Setup
#
# PURPOSE: Prepare the vLEI environment for deployment
#
# WHAT IT DOES:
#   1. Copies project files from Windows to Linux (WSL)
#   2. Installs required system dependencies (dos2unix, python3)
#   3. Fixes Windows line endings (CRLF → LF) on all files
#   4. Makes all shell scripts executable
#   5. Builds Docker images (including tsx-shell)
#
# USAGE:
#   ./setup.sh
#
# WORKFLOW (4-Step Process):
#   Step 1: ./stop.sh              # Clean up any existing environment
#   Step 2: ./setup.sh             # THIS SCRIPT - prepare everything
#   Step 3: ./deploy.sh            # Start all services
#   Step 4: ./run-all-buyerseller-4C-with-agents.sh  # Run the workflow
#
# NOTE: Schema SAIDification happens automatically after deploy.sh starts
#       the schema container (run ./saidify-with-docker.sh after deploy.sh)
#
# DATE: December 1, 2025
# VERSION: 2.1 - Fixed schema SAID handling
################################################################################

set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         vLEI Workshop Environment Setup                      ║${NC}"
echo -e "${CYAN}║         Version 2.1 - With Schema SAIDification              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

################################################################################
# STEP 1: Copy Project Files from Windows to Linux
################################################################################
echo -e "${YELLOW}[1/6] Copying project files from Windows to Linux...${NC}"

# # Create project directories
# mkdir -p ~/projects/algoTitanV51/LegentvLEI
# mkdir -p ~/projects/algoTitanV61/LegentvLEI

# # Copy from Windows mount (adjust paths as needed)
# if [ -d "/mnt/c/SATHYA/CHAINAIM3003/mcp-servers/stellarboston/LegentAlgoTITANV51" ]; then
#     cp -r /mnt/c/SATHYA/CHAINAIM3003/mcp-servers/stellarboston/LegentAlgoTITANV51/algoTITANV5/LegentvLEI/* ~/projects/algoTitanV51/LegentvLEI/ 2>/dev/null || true
#     echo -e "${GREEN}  ✓ Copied algoTitanV51${NC}"
# fi

# if [ -d "/mnt/c/SATHYA/CHAINAIM3003/mcp-servers/stellarboston/LegentAlgoTITANV61" ]; then
#     cp -r /mnt/c/SATHYA/CHAINAIM3003/mcp-servers/stellarboston/LegentAlgoTITANV61/algoTITANV6/LegentvLEI/* ~/projects/algoTitanV61/LegentvLEI/ 2>/dev/null || true
#     echo -e "${GREEN}  ✓ Copied algoTitanV61${NC}"
# fi

echo ""

################################################################################
# STEP 2: Install System Dependencies
################################################################################
echo -e "${YELLOW}[2/6] Installing system dependencies...${NC}"

sudo apt update -qq
sudo apt install -y dos2unix python3 jq curl > /dev/null 2>&1

echo -e "${GREEN}  ✓ Installed: dos2unix, python3, jq, curl${NC}"
echo ""

################################################################################
# STEP 3: Fix Windows Line Endings (CRLF → LF)
################################################################################
echo -e "${YELLOW}[3/6] Fixing Windows line endings...${NC}"

# Fix all shell scripts
find . -type f -name "*.sh" -exec dos2unix -q {} \; 2>/dev/null || true
echo -e "${GREEN}  ✓ Fixed .sh files${NC}"

# Fix TypeScript files
find . -type f -name "*.ts" -exec dos2unix -q {} \; 2>/dev/null || true
echo -e "${GREEN}  ✓ Fixed .ts files${NC}"

# Fix JSON files
find . -type f -name "*.json" -exec dos2unix -q {} \; 2>/dev/null || true
echo -e "${GREEN}  ✓ Fixed .json files${NC}"

# Fix YAML/YML files
find . -type f \( -name "*.yml" -o -name "*.yaml" \) -exec dos2unix -q {} \; 2>/dev/null || true
echo -e "${GREEN}  ✓ Fixed .yml/.yaml files${NC}"

# Fix Dockerfile
find . -type f -name "Dockerfile*" -exec dos2unix -q {} \; 2>/dev/null || true
echo -e "${GREEN}  ✓ Fixed Dockerfile files${NC}"

# Fix Python files
find . -type f -name "*.py" -exec dos2unix -q {} \; 2>/dev/null || true
echo -e "${GREEN}  ✓ Fixed .py files${NC}"

echo ""

################################################################################
# STEP 4: Make Shell Scripts Executable
################################################################################
echo -e "${YELLOW}[4/6] Making shell scripts executable...${NC}"

chmod +x *.sh 2>/dev/null || true
chmod +x scripts/*.sh 2>/dev/null || true
chmod +x */*.sh 2>/dev/null || true
chmod +x task-scripts/*.sh 2>/dev/null || true
chmod +x task-scripts/*/*.sh 2>/dev/null || true

echo -e "${GREEN}  ✓ All .sh files are now executable${NC}"
echo ""

################################################################################
# STEP 5: Verify Schema Configuration
################################################################################
echo -e "${YELLOW}[5/6] Verifying and fixing schema configuration...${NC}"

mkdir -p ./task-data

# Check schema file
if [ -f "./schemas/self-attested-invoice.json" ]; then
    SCHEMA_ID=$(jq -r '."$id" // ""' ./schemas/self-attested-invoice.json 2>/dev/null)
    if [ -z "$SCHEMA_ID" ] || [ "$SCHEMA_ID" = "" ] || [ "$SCHEMA_ID" = "null" ]; then
        echo -e "${YELLOW}  ⚠ Invoice schema needs SAIDification (empty \$id)${NC}"
        echo -e "${BLUE}    → Will be SAIDified after ./deploy.sh starts containers${NC}"
        NEEDS_SAIDIFY=true
    else
        echo -e "${GREEN}  ✓ Invoice schema has \$id: ${SCHEMA_ID}${NC}"
        
        # Check if SAID-named file exists
        if [ -f "./schemas/${SCHEMA_ID}.json" ]; then
            echo -e "${GREEN}  ✓ SAID-named schema file exists${NC}"
        else
            echo -e "${YELLOW}  ⚠ SAID-named schema file missing (will be created by SAIDification)${NC}"
        fi
    fi
else
    echo -e "${RED}  ✗ Invoice schema file not found: ./schemas/self-attested-invoice.json${NC}"
fi

# Check config file
if [ -f "./appconfig/schemaSaids.json" ]; then
    CONFIG_SAID=$(jq -r '.invoiceSchema.said // ""' ./appconfig/schemaSaids.json 2>/dev/null)
    if [ -n "$CONFIG_SAID" ] && [ "$CONFIG_SAID" != "null" ]; then
        echo -e "${GREEN}  ✓ Config SAID: ${CONFIG_SAID}${NC}"
        
        # Check if config SAID matches actual schema
        if [ -n "$SCHEMA_ID" ] && [ "$SCHEMA_ID" != "" ] && [ "$SCHEMA_ID" != "null" ] && [ "$SCHEMA_ID" != "$CONFIG_SAID" ]; then
            echo -e "${YELLOW}  ⚠ Config SAID differs from schema \$id - will be updated after SAIDification${NC}"
        fi
    else
        echo -e "${YELLOW}  ⚠ Config SAID is empty - will be set after SAIDification${NC}"
    fi
else
    echo -e "${YELLOW}  ⚠ schemaSaids.json not found - will be created${NC}"
fi

echo ""

################################################################################
# STEP 6: Build Docker Images
################################################################################
echo -e "${YELLOW}[6/6] Building Docker images...${NC}"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}  ✗ Docker not found. Please install Docker first.${NC}"
    exit 1
fi

# Check if docker compose is available
if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}  ✗ docker compose not found. Please install docker compose first.${NC}"
    exit 1
fi

echo -e "${BLUE}  Building tsx-shell container (with --no-cache)...${NC}"
docker compose build --no-cache tsx-shell 2>&1 | tail -5

echo -e "${BLUE}  Building vlei-verification service...${NC}"
docker compose build vlei-verification 2>&1 | tail -3 || true

echo -e "${GREEN}  ✓ Docker images built successfully${NC}"
echo ""

################################################################################
# COMPLETION SUMMARY
################################################################################
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    ✓ SETUP COMPLETE                          ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}All preparations complete! Your environment is ready.${NC}"
echo ""
echo -e "${WHITE}NEXT STEPS:${NC}"
echo ""
echo -e "  ${BLUE}1.${NC} Deploy the services:"
echo -e "     ${WHITE}./deploy.sh${NC}"
echo ""

if [ "$NEEDS_SAIDIFY" = true ]; then
echo -e "  ${BLUE}2.${NC} ${YELLOW}SAIDify the invoice schema (REQUIRED - schema \$id is empty):${NC}"
echo -e "     ${WHITE}./saidify-with-docker.sh${NC}"
echo ""
echo -e "  ${BLUE}3.${NC} Restart schema container to load new schema:"
echo -e "     ${WHITE}docker compose restart schema && sleep 15${NC}"
echo ""
echo -e "  ${BLUE}4.${NC} Run the vLEI workflow:"
echo -e "     ${WHITE}./run-all-buyerseller-4C-with-agents.sh${NC}"
else
echo -e "  ${BLUE}2.${NC} Wait for services to be healthy (~30 seconds)"
echo ""
echo -e "  ${BLUE}3.${NC} Run the vLEI workflow:"
echo -e "     ${WHITE}./run-all-buyerseller-4C-with-agents.sh${NC}"
fi

echo ""
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
echo -e "${YELLOW}QUICK REFERENCE:${NC}"
echo ""
echo -e "  Stop everything:      ${WHITE}./stop.sh${NC}"
echo -e "  Fresh start:          ${WHITE}./stop.sh && ./setup.sh && ./deploy.sh${NC}"
echo -e "  SAIDify schema:       ${WHITE}./saidify-with-docker.sh${NC}"
echo -e "  Run workflow:         ${WHITE}./run-all-buyerseller-4C-with-agents.sh${NC}"
echo ""
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
echo -e "${YELLOW}IMPORTANT FILES:${NC}"
echo ""
echo -e "  Schema config:      ${WHITE}./appconfig/schemaSaids.json${NC}"
echo -e "  Invoice schema:     ${WHITE}./schemas/self-attested-invoice.json${NC}"
echo -e "  Workflow script:    ${WHITE}./run-all-buyerseller-4C-with-agents.sh${NC}"
echo -e "  Task data:          ${WHITE}./task-data/${NC}"
echo ""
