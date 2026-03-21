#!/bin/bash
################################################################################
# generateAgentCards.sh
#
# Purpose: Generate agent cards from 4C workflow task-data
#          Copies data from WSL native path to Windows and generates cards
#
# Usage: Run from anywhere in WSL:
#   ./generateAgentCards.sh
#
# Output: Agent cards will be written to:
#   C:\SATHYA\CHAINAIM3003\mcp-servers\stellarboston\LegentAlgoTITANV61\algoTITANV6\Legent\A2A\agent-cards\
#
# Prerequisites:
#   - 4C workflow must have been run (creates task-data files)
#   - Node.js must be installed
#
# Date: December 2025
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Paths
WSL_NATIVE_TASK_DATA="$HOME/projects/algoTitanV61/LegentvLEI/task-data"
WINDOWS_MOUNTED_BASE="/mnt/c/SATHYA/CHAINAIM3003/mcp-servers/stellarboston/LegentAlgoTITANV61/algoTITANV6"
WINDOWS_LEGENDVLEI="${WINDOWS_MOUNTED_BASE}/LegentvLEI"
WINDOWS_TASK_DATA="${WINDOWS_LEGENDVLEI}/task-data"
WINDOWS_AGENT_CARDS="${WINDOWS_MOUNTED_BASE}/Legent/A2A/agent-cards"

echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Agent Card Generator${NC}"
echo -e "${CYAN}  Generates vLEI agent cards from 4C workflow data${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""

# Step 1: Check if WSL task-data exists
echo -e "${YELLOW}[1/5] Checking WSL task-data directory...${NC}"

if [ ! -d "$WSL_NATIVE_TASK_DATA" ]; then
    echo -e "${RED}✗ WSL task-data directory not found: $WSL_NATIVE_TASK_DATA${NC}"
    echo -e "${YELLOW}  Have you run the 4C workflow yet?${NC}"
    echo -e "${YELLOW}  Expected path: ~/projects/algoTitanV61/LegentvLEI/task-data${NC}"
    exit 1
fi

# Check for required files
REQUIRED_FILES=(
    "jupiterSellerAgent-info.json"
    "tommyBuyerAgent-info.json"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${WSL_NATIVE_TASK_DATA}/${file}" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo -e "${RED}✗ Missing required files in task-data:${NC}"
    for file in "${MISSING_FILES[@]}"; do
        echo -e "${RED}    - $file${NC}"
    done
    echo -e "${YELLOW}  Run the 4C workflow first to generate these files.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ WSL task-data found with required files${NC}"
echo ""

# Step 2: Ensure Windows directories exist
echo -e "${YELLOW}[2/5] Ensuring Windows directories exist...${NC}"

mkdir -p "$WINDOWS_TASK_DATA"
mkdir -p "$WINDOWS_AGENT_CARDS"

echo -e "${GREEN}✓ Windows directories ready${NC}"
echo "    Task-data: $WINDOWS_TASK_DATA"
echo "    Agent cards: $WINDOWS_AGENT_CARDS"
echo ""

# Step 3: Copy task-data from WSL to Windows
echo -e "${YELLOW}[3/5] Copying task-data from WSL to Windows...${NC}"

cp -r "${WSL_NATIVE_TASK_DATA}"/* "${WINDOWS_TASK_DATA}/"

# Count files copied
FILE_COUNT=$(ls -1 "${WINDOWS_TASK_DATA}" | wc -l)
echo -e "${GREEN}✓ Copied $FILE_COUNT files to Windows task-data${NC}"
echo ""

# Step 4: Run the Node.js generator script
echo -e "${YELLOW}[4/5] Running agent card generator...${NC}"

# Change to Windows-mounted LegentvLEI directory
cd "$WINDOWS_LEGENDVLEI"

# Check if node is available
if ! command -v node &> /dev/null; then
    echo -e "${RED}✗ Node.js is not installed or not in PATH${NC}"
    exit 1
fi

# Check if generator script exists
if [ ! -f "generate-agent-cards.js" ]; then
    echo -e "${RED}✗ generate-agent-cards.js not found in $WINDOWS_LEGENDVLEI${NC}"
    exit 1
fi

# Run the generator
node generate-agent-cards.js

echo ""

# Step 5: Verify output
echo -e "${YELLOW}[5/5] Verifying generated agent cards...${NC}"

if [ -f "${WINDOWS_AGENT_CARDS}/jupiterSellerAgent-card.json" ] && \
   [ -f "${WINDOWS_AGENT_CARDS}/tommyBuyerAgent-card.json" ]; then
    echo -e "${GREEN}✓ Agent cards generated successfully!${NC}"
    echo ""
    echo -e "${CYAN}Generated files:${NC}"
    ls -la "${WINDOWS_AGENT_CARDS}"/*.json 2>/dev/null || true
    echo ""
    
    # Show key vLEI metadata from generated cards
    echo -e "${CYAN}vLEI Metadata (for DEEP-EXT verification):${NC}"
    echo ""
    
    echo -e "${BLUE}Jupiter Seller Agent:${NC}"
    if command -v jq &> /dev/null; then
        jq -r '.extensions.vLEImetadata | "  agentName: \(.agentName)\n  oorHolderName: \(.oorHolderName)"' \
            "${WINDOWS_AGENT_CARDS}/jupiterSellerAgent-card.json" 2>/dev/null || echo "  (install jq to see details)"
    else
        echo "  (install jq to see details)"
    fi
    echo ""
    
    echo -e "${BLUE}Tommy Buyer Agent:${NC}"
    if command -v jq &> /dev/null; then
        jq -r '.extensions.vLEImetadata | "  agentName: \(.agentName)\n  oorHolderName: \(.oorHolderName)"' \
            "${WINDOWS_AGENT_CARDS}/tommyBuyerAgent-card.json" 2>/dev/null || echo "  (install jq to see details)"
    else
        echo "  (install jq to see details)"
    fi
    echo ""
else
    echo -e "${RED}✗ Agent card generation may have failed${NC}"
    echo -e "${YELLOW}  Check the output above for errors${NC}"
    exit 1
fi

echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Agent Card Generation Complete!${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Output location:${NC}"
echo "  Windows: C:\\SATHYA\\CHAINAIM3003\\mcp-servers\\stellarboston\\LegentAlgoTITANV61\\algoTITANV6\\Legent\\A2A\\agent-cards\\"
echo "  WSL:     $WINDOWS_AGENT_CARDS"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Agent cards are ready for the A2A agents to serve via well-known URLs"
echo "  2. The vLEImetadata.agentName and oorHolderName will be used for DEEP-EXT verification"
echo ""

exit 0
