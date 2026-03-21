#!/bin/bash
################################################################################
# generate-subagent-brans.sh - Generate Unique BRANs for Sub-Agents
#
# Purpose: Pre-generate cryptographic seeds (BRANs) for all sub-agents
#          before creating them, ensuring each has a unique identity
#
# Usage: ./generate-subagent-brans.sh
################################################################################

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

CONFIG_FILE="./appconfig/configBuyerSellerAIAgent1-with-subdelegation.json"
OUTPUT_FILE="./task-data/subagent-brans.json"
LEGENT_BASE="../Legent/A2A/js/src/agents"

echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Generating Unique Sub-Agent BRANs${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Create task-data directory if it doesn't exist
mkdir -p ./task-data

# Initialize output JSON
echo '{"subAgents": []}' > "$OUTPUT_FILE"

# Extract sub-agents from config
SUB_AGENT_COUNT=0

# Loop through organizations
ORG_COUNT=$(jq -r '.organizations | length' "$CONFIG_FILE")

for ((org_idx=0; org_idx<$ORG_COUNT; org_idx++)); do
    PERSON_COUNT=$(jq -r ".organizations[$org_idx].persons | length" "$CONFIG_FILE")
    
    for ((person_idx=0; person_idx<$PERSON_COUNT; person_idx++)); do
        AGENT_COUNT=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents | length" "$CONFIG_FILE")
        
        for ((agent_idx=0; agent_idx<$AGENT_COUNT; agent_idx++)); do
            # Check if this agent has sub-agents
            SUB_AGENT_LIST=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents[$agent_idx].subAgents // []" "$CONFIG_FILE")
            
            if [ "$SUB_AGENT_LIST" != "[]" ]; then
                PARENT_AGENT=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents[$agent_idx].alias" "$CONFIG_FILE")
                PARENT_PERSON=$(jq -r ".organizations[$org_idx].persons[$person_idx].alias" "$CONFIG_FILE")
                
                SUB_COUNT=$(echo "$SUB_AGENT_LIST" | jq 'length')
                
                for ((sub_idx=0; sub_idx<$SUB_COUNT; sub_idx++)); do
                    SUB_ALIAS=$(echo "$SUB_AGENT_LIST" | jq -r ".[$sub_idx].alias")
                    SUB_TYPE=$(echo "$SUB_AGENT_LIST" | jq -r ".[$sub_idx].agentType")
                    SUB_SCOPE=$(echo "$SUB_AGENT_LIST" | jq -r ".[$sub_idx].permissions.scope")
                    
                    echo -e "${BLUE}→ Processing: ${SUB_ALIAS}${NC}"
                    
                    # Generate unique 256-bit BRAN (44 base64 characters)
                    BRAN=$(openssl rand -base64 32)
                    
                    # Save BRAN to individual file
                    echo "$BRAN" > "./task-data/${SUB_ALIAS}-bran.txt"
                    
                    # Add to JSON output
                    jq --arg alias "$SUB_ALIAS" \
                       --arg bran "$BRAN" \
                       --arg parent "$PARENT_AGENT" \
                       --arg person "$PARENT_PERSON" \
                       --arg type "$SUB_TYPE" \
                       --arg scope "$SUB_SCOPE" \
                       '.subAgents += [{
                         alias: $alias,
                         bran: $bran,
                         parentAgent: $parent,
                         parentPerson: $person,
                         agentType: $type,
                         scope: $scope,
                         isSubAgent: true
                       }]' "$OUTPUT_FILE" > "$OUTPUT_FILE.tmp" && mv "$OUTPUT_FILE.tmp" "$OUTPUT_FILE"
                    
                    # Create .env file for Legent A2A agent if directory exists
                    AGENT_DIR="$LEGENT_BASE/$SUB_ALIAS"
                    if [ ! -d "$AGENT_DIR" ]; then
                        mkdir -p "$AGENT_DIR"
                    fi
                    
                    cat > "$AGENT_DIR/.env" << EOF
# Sub-Agent Configuration for $SUB_ALIAS
# Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
# Parent Agent: $PARENT_AGENT
# Parent Person: $PARENT_PERSON

# Unique BRAN (256-bit cryptographic seed)
BRAN=$BRAN

# Agent Identity
AGENT_NAME=$SUB_ALIAS
AGENT_TYPE=$SUB_TYPE
IS_SUB_AGENT=true

# Delegation Chain
PARENT_AGENT=$PARENT_AGENT
PARENT_PERSON=$PARENT_PERSON

# Permissions
SCOPE=$SUB_SCOPE
CAN_DELEGATE=false

# KERIA Connection
KERIA_URL=http://127.0.0.1:3902
KERIA_BOOT_URL=http://127.0.0.1:3903
EOF
                    
                    echo -e "${GREEN}  ✓ BRAN: ${BRAN:0:20}...${NC}"
                    
                    SUB_AGENT_COUNT=$((SUB_AGENT_COUNT + 1))
                done
            fi
        done
    done
done

echo ""
echo -e "${GREEN}✅ Complete! Generated $SUB_AGENT_COUNT unique sub-agent BRANs${NC}"
echo ""

# Verify all BRANs are unique
UNIQUE_COUNT=$(jq -r '.subAgents[].bran' "$OUTPUT_FILE" | sort -u | wc -l)
TOTAL_COUNT=$(jq -r '.subAgents | length' "$OUTPUT_FILE")

if [ "$UNIQUE_COUNT" -eq "$TOTAL_COUNT" ]; then
    echo -e "${GREEN}✓ All BRANs are unique${NC}"
else
    echo -e "${RED}✗ WARNING: Duplicate BRANs detected!${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Configuration saved to:${NC}"
echo "  • $OUTPUT_FILE"
echo "  • ./task-data/[SUB_AGENT_NAME]-bran.txt (individual files)"
echo "  • $LEGENT_BASE/*/. env (agent directories)"
echo ""

exit 0