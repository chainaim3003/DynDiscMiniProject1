#!/bin/bash
################################################################################
# subagent-delegate-with-unique-bran.sh (FIXED)
#
# Purpose: Create and delegate sub-agent in 3 steps to avoid libsodium conflicts
#
# Steps:
#   1. Sub-agent creates AID and requests delegation
#   2. Parent agent approves delegation (creates seal)
#   3. Sub-agent completes delegation (resolves OOBI, adds endpoint)
#
# Usage: ./task-scripts/subagent/subagent-delegate-with-unique-bran.sh <sub_agent> <parent_agent>
################################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SUB_AGENT_ALIAS=$1
PARENT_AGENT_ALIAS=$2

if [ -z "$SUB_AGENT_ALIAS" ] || [ -z "$PARENT_AGENT_ALIAS" ]; then
    echo -e "${RED}ERROR: Missing required arguments${NC}"
    echo "Usage: $0 <sub_agent_alias> <parent_agent_alias>"
    exit 1
fi

echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Sub-Agent Delegation (3-Step Process)${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Sub-Agent: $SUB_AGENT_ALIAS"
echo "  Parent Agent: $PARENT_AGENT_ALIAS"
echo ""

# Verify BRANs exist
SUB_BRAN_FILE="./task-data/${SUB_AGENT_ALIAS}-bran.txt"
PARENT_BRAN_FILE="./task-data/${PARENT_AGENT_ALIAS}-bran.txt"

if [ ! -f "$SUB_BRAN_FILE" ]; then
    echo -e "${RED}ERROR: Sub-agent BRAN not found: $SUB_BRAN_FILE${NC}"
    exit 1
fi

if [ ! -f "$PARENT_BRAN_FILE" ]; then
    echo -e "${RED}ERROR: Parent agent BRAN not found: $PARENT_BRAN_FILE${NC}"
    exit 1
fi

PARENT_INFO_FILE="./task-data/${PARENT_AGENT_ALIAS}-info.json"
if [ ! -f "$PARENT_INFO_FILE" ]; then
    echo -e "${RED}ERROR: Parent agent info not found: $PARENT_INFO_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}✓ Sub-agent BRAN verified${NC}"
echo -e "${BLUE}✓ Parent agent BRAN verified${NC}"
echo -e "${BLUE}✓ Parent agent info verified${NC}"
echo ""

################################################################################
# STEP 1: Create Sub-Agent AID and Request Delegation
################################################################################

echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  STEP 1: Create Sub-Agent AID and Request Delegation${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo ""

docker compose exec -T tsx-shell tsx \
    sig-wallet/src/tasks/subagent/subagent-delegate-inception.ts \
    docker \
    "$SUB_AGENT_ALIAS" \
    "$PARENT_AGENT_ALIAS" \
    "/task-data"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Step 1 failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Step 1 complete: Sub-agent AID created and delegation requested${NC}"
echo ""
sleep 2

################################################################################
# STEP 2: Parent Agent Approves Delegation
################################################################################

echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  STEP 2: Parent Agent Approves Delegation${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo ""

docker compose exec -T tsx-shell tsx \
    sig-wallet/src/tasks/subagent/subagent-delegate-approve.ts \
    docker \
    "$PARENT_AGENT_ALIAS" \
    "$SUB_AGENT_ALIAS" \
    "/task-data"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Step 2 failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Step 2 complete: Parent agent approved delegation${NC}"
echo ""
sleep 2

################################################################################
# STEP 3: Sub-Agent Completes Delegation
################################################################################

echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  STEP 3: Sub-Agent Completes Delegation${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo ""

docker compose exec -T tsx-shell tsx \
    sig-wallet/src/tasks/subagent/subagent-delegate-finish.ts \
    docker \
    "$SUB_AGENT_ALIAS" \
    "$PARENT_AGENT_ALIAS" \
    "/task-data"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Step 3 failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Step 3 complete: Sub-agent delegation finished${NC}"
echo ""

################################################################################
# Summary
################################################################################

if [ -f "./task-data/${SUB_AGENT_ALIAS}-info.json" ]; then
    SUB_AID=$(jq -r '.aid' "./task-data/${SUB_AGENT_ALIAS}-info.json")
    SUB_OOBI=$(jq -r '.oobi' "./task-data/${SUB_AGENT_ALIAS}-info.json")
    PARENT_AID=$(jq -r '.aid' "$PARENT_INFO_FILE")
    
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  ✅ Sub-Agent Delegation Complete${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Sub-Agent: $SUB_AGENT_ALIAS"
    echo "  Sub-Agent AID: $SUB_AID"
    echo "  Sub-Agent OOBI: $SUB_OOBI"
    echo ""
    echo "  Parent Agent: $PARENT_AGENT_ALIAS"
    echo "  Parent AID: $PARENT_AID"
    echo ""
    echo "  Trust Chain:"
    echo "    OOR Holder → $PARENT_AGENT_ALIAS → $SUB_AGENT_ALIAS"
    echo ""
    echo "  Output Files:"
    echo "    • ./task-data/${SUB_AGENT_ALIAS}-info.json"
    echo "    • ./task-data/${SUB_AGENT_ALIAS}-delegate-info.json"
    echo ""
fi

exit 0