#!/bin/bash
###############################################################################
# fix-invoice-schema-said.sh
#
# This script:
# 1. Runs SAIDify using signify-ts in the tsx-shell container
# 2. Extracts the computed SAID
# 3. Updates the schema file with the SAID in $id
# 4. Creates schema file named with the SAID
# 5. Updates all TypeScript and shell references
# 6. Rebuilds and restarts containers
###############################################################################

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Invoice Schema SAID Fix Script${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""

# Paths
SCHEMA_DIR="./schemas"
SCHEMA_FILE="$SCHEMA_DIR/self-attested-invoice.json"
TS_FILE="./sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only.ts"
TS_FIXED_FILE="./sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only-FIXED.ts"
SHELL_SCRIPT="./run-all-buyerseller-4C-with-agents.sh"
SHELL_SCRIPT_FIXED="./run-all-buyerseller-4C-with-agents-FIXED.sh"
OLD_SAID="EHFQviJgaf6YTHbMPLheau8H1s3v31-ByTER_D49DLHY"

###############################################################################
# Step 1: Ensure schema file has empty $id for SAIDification
###############################################################################
echo -e "${YELLOW}[1/8] Preparing schema file for SAIDification...${NC}"

# Read current schema and ensure $id is empty
if [ -f "$SCHEMA_FILE" ]; then
    # Use jq to set $id to empty string
    jq '."$id" = ""' "$SCHEMA_FILE" > "$SCHEMA_FILE.tmp" && mv "$SCHEMA_FILE.tmp" "$SCHEMA_FILE"
    echo -e "${GREEN}✓ Schema $id set to empty string${NC}"
else
    echo -e "${RED}✗ Schema file not found: $SCHEMA_FILE${NC}"
    exit 1
fi

###############################################################################
# Step 2: Copy schema to tsx-shell container
###############################################################################
echo -e "${YELLOW}[2/8] Copying schema to tsx-shell container...${NC}"

docker compose cp "$SCHEMA_FILE" tsx-shell:/vlei/schema-to-saidify.json
echo -e "${GREEN}✓ Schema copied to container${NC}"

###############################################################################
# Step 3: Run SAIDify in container
###############################################################################
echo -e "${YELLOW}[3/8] Running SAIDification using signify-ts...${NC}"

# Run the saidify task
SAIDIFY_OUTPUT=$(docker compose exec -T tsx-shell npx tsx src/tasks/saidify-schema.ts /vlei/schema-to-saidify.json 2>&1)
echo "$SAIDIFY_OUTPUT"

# Extract SAID from output
NEW_SAID=$(echo "$SAIDIFY_OUTPUT" | grep -oE 'E[A-Za-z0-9_-]{43}' | head -1)

if [ -z "$NEW_SAID" ]; then
    # Try alternate pattern
    NEW_SAID=$(echo "$SAIDIFY_OUTPUT" | grep -oE 'SAID.*E[A-Za-z0-9_-]{43}' | grep -oE 'E[A-Za-z0-9_-]{43}' | head -1)
fi

if [ -z "$NEW_SAID" ] || [ ${#NEW_SAID} -ne 44 ]; then
    echo -e "${RED}✗ Failed to extract SAID from SAIDify output${NC}"
    echo "Output was: $SAIDIFY_OUTPUT"
    
    # Try to get the SAID from the file in container
    echo -e "${YELLOW}Attempting to read SAID from saidified file...${NC}"
    NEW_SAID=$(docker compose exec -T tsx-shell cat /vlei/schema-to-saidify.json | jq -r '."$id"')
    
    if [ -z "$NEW_SAID" ] || [ "$NEW_SAID" == "" ] || [ "$NEW_SAID" == "null" ]; then
        echo -e "${RED}✗ Could not get SAID from container file${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Computed SAID: $NEW_SAID${NC}"

###############################################################################
# Step 4: Copy SAIDified schema back to host
###############################################################################
echo -e "${YELLOW}[4/8] Copying SAIDified schema back to host...${NC}"

docker compose cp tsx-shell:/vlei/schema-to-saidify.json "$SCHEMA_FILE"
echo -e "${GREEN}✓ SAIDified schema copied back${NC}"

# Verify $id is set
SCHEMA_ID=$(jq -r '."$id"' "$SCHEMA_FILE")
if [ "$SCHEMA_ID" != "$NEW_SAID" ]; then
    echo -e "${YELLOW}  Updating $id in schema file...${NC}"
    jq --arg said "$NEW_SAID" '."$id" = $said' "$SCHEMA_FILE" > "$SCHEMA_FILE.tmp" && mv "$SCHEMA_FILE.tmp" "$SCHEMA_FILE"
fi

echo -e "${GREEN}✓ Schema $id is: $(jq -r '."$id"' "$SCHEMA_FILE")${NC}"

###############################################################################
# Step 5: Create schema file with SAID as filename
###############################################################################
echo -e "${YELLOW}[5/8] Creating schema file with SAID filename...${NC}"

SAID_SCHEMA_FILE="$SCHEMA_DIR/${NEW_SAID}.json"
cp "$SCHEMA_FILE" "$SAID_SCHEMA_FILE"
echo -e "${GREEN}✓ Created: $SAID_SCHEMA_FILE${NC}"

# Remove old SAID file if different
if [ "$OLD_SAID" != "$NEW_SAID" ] && [ -f "$SCHEMA_DIR/${OLD_SAID}.json" ]; then
    rm -f "$SCHEMA_DIR/${OLD_SAID}.json"
    echo -e "${GREEN}✓ Removed old schema file: $SCHEMA_DIR/${OLD_SAID}.json${NC}"
fi

###############################################################################
# Step 6: Update TypeScript files
###############################################################################
echo -e "${YELLOW}[6/8] Updating TypeScript files...${NC}"

# Update invoice-acdc-issue-self-attested-only.ts
if [ -f "$TS_FILE" ]; then
    sed -i "s/const INVOICE_SCHEMA_SAID = \"$OLD_SAID\"/const INVOICE_SCHEMA_SAID = \"$NEW_SAID\"/" "$TS_FILE"
    sed -i "s/const INVOICE_SCHEMA_SAID = \"E[A-Za-z0-9_-]\{43\}\"/const INVOICE_SCHEMA_SAID = \"$NEW_SAID\"/" "$TS_FILE"
    echo -e "${GREEN}✓ Updated: $TS_FILE${NC}"
fi

if [ -f "$TS_FIXED_FILE" ]; then
    sed -i "s/const INVOICE_SCHEMA_SAID = \"$OLD_SAID\"/const INVOICE_SCHEMA_SAID = \"$NEW_SAID\"/" "$TS_FIXED_FILE"
    sed -i "s/const INVOICE_SCHEMA_SAID = \"E[A-Za-z0-9_-]\{43\}\"/const INVOICE_SCHEMA_SAID = \"$NEW_SAID\"/" "$TS_FIXED_FILE"
    echo -e "${GREEN}✓ Updated: $TS_FIXED_FILE${NC}"
fi

###############################################################################
# Step 7: Update shell scripts
###############################################################################
echo -e "${YELLOW}[7/8] Updating shell scripts...${NC}"

# Update run-all-buyerseller-4C-with-agents.sh
if [ -f "$SHELL_SCRIPT" ]; then
    sed -i "s/INVOICE_SCHEMA_SAID=\"$OLD_SAID\"/INVOICE_SCHEMA_SAID=\"$NEW_SAID\"/" "$SHELL_SCRIPT"
    sed -i "s/INVOICE_SCHEMA_SAID=\"E[A-Za-z0-9_-]\{43\}\"/INVOICE_SCHEMA_SAID=\"$NEW_SAID\"/" "$SHELL_SCRIPT"
    echo -e "${GREEN}✓ Updated: $SHELL_SCRIPT${NC}"
fi

if [ -f "$SHELL_SCRIPT_FIXED" ]; then
    sed -i "s/INVOICE_SCHEMA_SAID=\"$OLD_SAID\"/INVOICE_SCHEMA_SAID=\"$NEW_SAID\"/" "$SHELL_SCRIPT_FIXED"
    sed -i "s/INVOICE_SCHEMA_SAID=\"E[A-Za-z0-9_-]\{43\}\"/INVOICE_SCHEMA_SAID=\"$NEW_SAID\"/" "$SHELL_SCRIPT_FIXED"
    echo -e "${GREEN}✓ Updated: $SHELL_SCRIPT_FIXED${NC}"
fi

# Update deploy.sh
if grep -q "$OLD_SAID" deploy.sh 2>/dev/null; then
    sed -i "s/$OLD_SAID/$NEW_SAID/g" deploy.sh
    echo -e "${GREEN}✓ Updated: deploy.sh${NC}"
fi

###############################################################################
# Step 8: Rebuild and restart containers
###############################################################################
echo -e "${YELLOW}[8/8] Rebuilding tsx-shell and restarting schema container...${NC}"

# Rebuild tsx-shell to pick up updated TypeScript
docker compose build tsx-shell --no-cache

# Restart schema container to pick up new schema files
docker compose restart schema

# Wait for schema container to be healthy
echo -e "${YELLOW}Waiting for schema container to be healthy...${NC}"
for i in {1..30}; do
    STATUS=$(docker compose ps schema --format "{{.Status}}" 2>/dev/null | grep -o "healthy" || true)
    if [ "$STATUS" == "healthy" ]; then
        echo -e "${GREEN}✓ Schema container is healthy${NC}"
        break
    fi
    sleep 1
done

# Verify schema is being cached
echo -e "${YELLOW}Verifying schema is cached by vLEI server...${NC}"
sleep 5
CACHED=$(docker compose logs schema 2>&1 | grep "caching schema $NEW_SAID" || true)
if [ -n "$CACHED" ]; then
    echo -e "${GREEN}✓ Schema $NEW_SAID is being cached by vLEI server${NC}"
else
    echo -e "${YELLOW}⚠ Schema caching not confirmed in logs yet${NC}"
    echo "  Check manually: docker compose logs schema | grep 'caching schema'"
fi

# Test schema accessibility
echo -e "${YELLOW}Testing schema accessibility from KERIA...${NC}"
SCHEMA_TEST=$(docker compose exec keria wget -qO- "http://schema:7723/oobi/$NEW_SAID" 2>&1 || true)
if [ -n "$SCHEMA_TEST" ] && echo "$SCHEMA_TEST" | grep -q "$NEW_SAID"; then
    echo -e "${GREEN}✓ Schema accessible from KERIA${NC}"
else
    echo -e "${YELLOW}⚠ Schema not immediately accessible. May need container restart.${NC}"
fi

###############################################################################
# Summary
###############################################################################
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  INVOICE SCHEMA SAID FIX COMPLETE${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Old SAID: ${RED}$OLD_SAID${NC}"
echo -e "  New SAID: ${GREEN}$NEW_SAID${NC}"
echo ""
echo "  Files updated:"
echo "    - $SCHEMA_FILE"
echo "    - $SAID_SCHEMA_FILE"
echo "    - $TS_FILE"
echo "    - $SHELL_SCRIPT"
echo ""
echo "  Next steps:"
echo "    1. If schema still not accessible, run: ./stop.sh && ./deploy.sh"
echo "    2. Then run: ./run-all-buyerseller-4C-with-agents.sh"
echo ""
