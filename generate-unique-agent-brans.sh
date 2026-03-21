#!/bin/bash
################################################################################
# generate-unique-agent-brans.sh
# Purpose: Generate unique cryptographic BRANs for ALL agents before creation
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
CONFIG_FILE="./appconfig/configBuyerSellerAIAgent1.json"
TASK_DATA_DIR="./task-data"
AGENTS_BASE_DIR="../Legent/A2A/js/src/agents"
OUTPUT_FILE="${TASK_DATA_DIR}/agent-brans.json"

mkdir -p "$TASK_DATA_DIR"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo -e "${RED}ERROR: jq is not installed${NC}"
    exit 1
fi

if ! command -v openssl &> /dev/null; then
    echo -e "${RED}ERROR: openssl is not installed${NC}"
    exit 1
fi

generate_bran() {
    openssl rand -base64 32 | tr -d '\n='
}

get_keria_alias() {
    local agent_type=$1
    local org_name=$2
    
    case "$agent_type" in
        "AI Agent")
            if [[ "$org_name" == *"Jupiter"* ]]; then
                echo "jupiterSellerAgent"
            elif [[ "$org_name" == *"Tommy"* ]]; then
                echo "tommyBuyerAgent"
            else
                echo "unknownAgent"
            fi
            ;;
        *)
            echo "unknownAgent"
            ;;
    esac
}

get_agent_role() {
    local org_name=$1
    
    if [[ "$org_name" == *"Jupiter"* ]]; then
        echo "seller"
    elif [[ "$org_name" == *"Tommy"* ]]; then
        echo "buyer"
    else
        echo "unknown"
    fi
}

echo -e "${BLUE}Generating Unique Agent BRANs...${NC}"

echo "{" > "$OUTPUT_FILE"
echo "  \"generated_at\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"," >> "$OUTPUT_FILE"
echo "  \"agents\": [" >> "$OUTPUT_FILE"

ORG_COUNT=$(jq -r '.organizations | length' "$CONFIG_FILE")
TOTAL_AGENTS=0

# Count agents
for ((org_idx=0; org_idx<$ORG_COUNT; org_idx++)); do
    PERSON_COUNT=$(jq -r ".organizations[$org_idx].persons | length" "$CONFIG_FILE")
    for ((person_idx=0; person_idx<$PERSON_COUNT; person_idx++)); do
        AGENT_COUNT=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents | length" "$CONFIG_FILE")
        TOTAL_AGENTS=$((TOTAL_AGENTS + AGENT_COUNT))
    done
done

echo -e "${GREEN}✓ Found $TOTAL_AGENTS agent(s)${NC}"

declare -a generated_brans=()
CURRENT_AGENT=0

for ((org_idx=0; org_idx<$ORG_COUNT; org_idx++)); do
    ORG_NAME=$(jq -r ".organizations[$org_idx].name" "$CONFIG_FILE")
    ORG_ALIAS=$(jq -r ".organizations[$org_idx].alias" "$CONFIG_FILE")
    ORG_LEI=$(jq -r ".organizations[$org_idx].lei" "$CONFIG_FILE")
    PERSON_COUNT=$(jq -r ".organizations[$org_idx].persons | length" "$CONFIG_FILE")
    
    for ((person_idx=0; person_idx<$PERSON_COUNT; person_idx++)); do
        PERSON_ALIAS=$(jq -r ".organizations[$org_idx].persons[$person_idx].alias" "$CONFIG_FILE")
        PERSON_NAME=$(jq -r ".organizations[$org_idx].persons[$person_idx].legalName" "$CONFIG_FILE")
        PERSON_ROLE=$(jq -r ".organizations[$org_idx].persons[$person_idx].officialRole" "$CONFIG_FILE")
        AGENT_COUNT=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents | length" "$CONFIG_FILE")
        
        if [ "$AGENT_COUNT" -gt 0 ]; then
            for ((agent_idx=0; agent_idx<$AGENT_COUNT; agent_idx++)); do
                AGENT_ALIAS=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents[$agent_idx].alias" "$CONFIG_FILE")
                AGENT_TYPE=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents[$agent_idx].agentType" "$CONFIG_FILE")
                
                echo -e "${BLUE}→ Processing: ${AGENT_ALIAS}${NC}"
                
                BRAN=$(generate_bran)
                while [[ " ${generated_brans[*]} " =~ " ${BRAN} " ]]; do
                    BRAN=$(generate_bran)
                done
                generated_brans+=("$BRAN")
                
                KERIA_ALIAS=$(get_keria_alias "$AGENT_TYPE" "$ORG_NAME")
                AGENT_ROLE=$(get_agent_role "$ORG_NAME")
                
                echo -e "${GREEN}  ✓ BRAN: ${BRAN:0:20}...${NC}"
                
                BRAN_FILE="${TASK_DATA_DIR}/${AGENT_ALIAS}-bran.txt"
                echo "$BRAN" > "$BRAN_FILE"
                
                if [ $CURRENT_AGENT -gt 0 ]; then
                    echo "," >> "$OUTPUT_FILE"
                fi
                
                # Write JSON entry directly without heredoc
                echo "    {" >> "$OUTPUT_FILE"
                echo "      \"alias\": \"$AGENT_ALIAS\"," >> "$OUTPUT_FILE"
                echo "      \"keriaAlias\": \"$KERIA_ALIAS\"," >> "$OUTPUT_FILE"
                echo "      \"agentType\": \"$AGENT_TYPE\"," >> "$OUTPUT_FILE"
                echo "      \"role\": \"$AGENT_ROLE\"," >> "$OUTPUT_FILE"
                echo "      \"organization\": \"$ORG_NAME\"," >> "$OUTPUT_FILE"
                echo "      \"orgAlias\": \"$ORG_ALIAS\"," >> "$OUTPUT_FILE"
                echo "      \"orgLEI\": \"$ORG_LEI\"," >> "$OUTPUT_FILE"
                echo "      \"oorHolder\": \"$PERSON_NAME\"," >> "$OUTPUT_FILE"
                echo "      \"oorHolderAlias\": \"$PERSON_ALIAS\"," >> "$OUTPUT_FILE"
                echo "      \"oorRole\": \"$PERSON_ROLE\"," >> "$OUTPUT_FILE"
                echo "      \"bran\": \"$BRAN\"," >> "$OUTPUT_FILE"
                echo "      \"aid\": null," >> "$OUTPUT_FILE"
                echo "      \"envFile\": \"${AGENTS_BASE_DIR}/${AGENT_ROLE}-agent/.env\"," >> "$OUTPUT_FILE"
                echo "      \"branFile\": \"$BRAN_FILE\"" >> "$OUTPUT_FILE"
                echo "    }" >> "$OUTPUT_FILE"
                
                CURRENT_AGENT=$((CURRENT_AGENT + 1))
                
                ENV_FILE="${AGENTS_BASE_DIR}/${AGENT_ROLE}-agent/.env"
                
                if [ -f "$ENV_FILE" ]; then
                    if grep -q "^AGENT_BRAN=" "$ENV_FILE"; then
                        sed -i "s|^AGENT_BRAN=.*|AGENT_BRAN=$BRAN|" "$ENV_FILE"
                    else
                        sed -i "/^KERIA_URL=/a\\
\\
# Unique Agent BRAN\\
AGENT_BRAN=$BRAN" "$ENV_FILE"
                    fi
                    
                    if grep -q "^AGENT_NAME=" "$ENV_FILE"; then
                        sed -i "s|^AGENT_NAME=.*|AGENT_NAME=$KERIA_ALIAS|" "$ENV_FILE"
                    fi
                    
                    chmod 600 "$ENV_FILE"
                    echo -e "${GREEN}  ✓ Updated: $ENV_FILE${NC}"
                fi
            done
        fi
    done
done

echo "" >> "$OUTPUT_FILE"
echo "  ]" >> "$OUTPUT_FILE"
echo "}" >> "$OUTPUT_FILE"

echo -e "${GREEN}✅ Complete! Generated $TOTAL_AGENTS unique BRANs${NC}"

UNIQUE_COUNT=$(cat "$OUTPUT_FILE" | jq -r '.agents[].bran' | sort | uniq | wc -l)
if [ "$UNIQUE_COUNT" -eq "$TOTAL_AGENTS" ]; then
    echo -e "${GREEN}✓ All BRANs are unique${NC}"
else
    echo -e "${RED}✗ ERROR: BRAN collision detected!${NC}"
    exit 1
fi

exit 0
