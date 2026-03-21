#!/bin/bash
###############################################################################
# saidify-with-keripy.sh
#
# SAIDifies the invoice schema using keripy (the official Python library)
# inside the vlei_shell container which has keripy installed.
#
# This follows the official GLEIF vLEI training approach:
# - Uses keri.core.coring.Saider with MtrDex.Blake3_256
# - Processes nested $id fields in oneOf blocks first
# - Then computes the top-level $id
###############################################################################

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  SAIDify Invoice Schema using keripy (Official Method)${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""

# Schema paths
SCHEMA_INPUT="./schemas/self-attested-invoice.json"
SCHEMA_OUTPUT="./schemas/self-attested-invoice-saidified.json"

# Check vlei_shell container is running
if ! docker compose ps vlei-shell 2>/dev/null | grep -q "Up"; then
    echo -e "${RED}ERROR: vlei_shell container is not running${NC}"
    echo "Run ./deploy.sh first"
    exit 1
fi

# Copy schema to vlei_shell container
echo -e "${YELLOW}[1/4] Copying schema to vlei_shell container...${NC}"
docker compose cp "$SCHEMA_INPUT" vlei-shell:/tmp/schema-to-saidify.json
echo -e "${GREEN}✓ Schema copied${NC}"

# Run SAIDify using keripy inside the container
echo -e "${YELLOW}[2/4] Running SAIDification using keripy...${NC}"

docker compose exec -T vlei-shell python3 << 'PYTHON_SCRIPT'
import json
import copy
from keri.core import coring, scheming

# Constants from official GLEIF saidify.py
DEFAULT_SAID_KEY = coring.Saids.dollar  # '$id'
DEFAULT_HASH_CODE = coring.MtrDex.Blake3_256
JSON_SCHEMA_ID_KEY = '$id'
PROPERTY_KEYS_TO_PROCESS = ["a", "e", "r"]

def _try_calculate_and_set_said(item, hash_code):
    """Calculate and set SAID for a dictionary item with $id field."""
    if not isinstance(item, dict) or JSON_SCHEMA_ID_KEY not in item:
        return
    
    # Check if there's content other than $id
    temp_copy = item.copy()
    del temp_copy[JSON_SCHEMA_ID_KEY]
    if not temp_copy:
        return
    
    try:
        item_for_saider = item.copy()
        said = coring.Saider(sad=item_for_saider,
                             code=hash_code,
                             label=JSON_SCHEMA_ID_KEY).qb64
        item[JSON_SCHEMA_ID_KEY] = said
        print(f"  Calculated nested SAID: {said}")
    except Exception as e:
        print(f"  Warning: Error generating SAID: {e}")

def add_saids_to_data(data_dict, said_key=DEFAULT_SAID_KEY, hash_code=DEFAULT_HASH_CODE):
    """Add SAIDs to schema following official GLEIF method."""
    processed_dict = copy.deepcopy(data_dict)
    
    # Process nested $id fields in a, e, r properties
    if 'properties' in processed_dict:
        properties = processed_dict['properties']
        for prop_key in PROPERTY_KEYS_TO_PROCESS:
            if prop_key in properties:
                prop_value = properties[prop_key]
                
                # Direct property value
                if isinstance(prop_value, dict):
                    _try_calculate_and_set_said(prop_value, hash_code)
                
                # oneOf list
                if isinstance(prop_value, dict) and 'oneOf' in prop_value:
                    if isinstance(prop_value['oneOf'], list):
                        for item in prop_value['oneOf']:
                            _try_calculate_and_set_said(item, hash_code)
    
    # Calculate top-level SAID
    dict_copy = copy.deepcopy(processed_dict)
    try:
        processed_dict[said_key] = coring.Saider(
            sad=dict_copy,
            code=hash_code,
            label=said_key
        ).qb64
        print(f"  Calculated top-level SAID: {processed_dict[said_key]}")
    except Exception as e:
        print(f"  Warning: Error generating top-level SAID: {e}")
    
    return processed_dict

# Read input schema
print("Reading schema file...")
with open('/tmp/schema-to-saidify.json', 'r') as f:
    schema = json.load(f)

print(f"Original $id: '{schema.get('$id', '')}'")

# Ensure $id is empty for SAID computation
schema['$id'] = ''

# Also clear nested $id fields
if 'properties' in schema:
    for prop_key in PROPERTY_KEYS_TO_PROCESS:
        if prop_key in schema['properties']:
            prop_value = schema['properties'][prop_key]
            if isinstance(prop_value, dict) and '$id' in prop_value:
                prop_value['$id'] = ''
            if isinstance(prop_value, dict) and 'oneOf' in prop_value:
                for item in prop_value.get('oneOf', []):
                    if isinstance(item, dict) and '$id' in item:
                        item['$id'] = ''

print("\nCalculating SAIDs...")
saidified = add_saids_to_data(schema)

# Use Schemer for final validation/formatting
try:
    schemer = scheming.Schemer(sed=saidified)
    final_schema = schemer.sed
    print("\n✓ Schema validated by Schemer")
except Exception as e:
    print(f"\nWarning: Schemer validation failed: {e}")
    final_schema = saidified

# Write output
with open('/tmp/schema-saidified.json', 'w') as f:
    json.dump(final_schema, f, indent=2)

print(f"\n✓ SAIDified schema written to /tmp/schema-saidified.json")
print(f"  Top-level SAID: {final_schema.get('$id', 'NOT SET')}")
PYTHON_SCRIPT

# Copy SAIDified schema back to host
echo -e "${YELLOW}[3/4] Copying SAIDified schema back to host...${NC}"
docker compose cp vlei-shell:/tmp/schema-saidified.json "$SCHEMA_OUTPUT"

# Extract the new SAID
NEW_SAID=$(jq -r '."$id"' "$SCHEMA_OUTPUT")
echo -e "${GREEN}✓ New SAID: $NEW_SAID${NC}"

# Create schema file with SAID as filename
echo -e "${YELLOW}[4/4] Creating schema files and updating references...${NC}"

SAID_SCHEMA_FILE="./schemas/${NEW_SAID}.json"
cp "$SCHEMA_OUTPUT" "$SAID_SCHEMA_FILE"
echo -e "${GREEN}✓ Created: $SAID_SCHEMA_FILE${NC}"

# Also update the main schema file
cp "$SCHEMA_OUTPUT" "$SCHEMA_INPUT"
echo -e "${GREEN}✓ Updated: $SCHEMA_INPUT${NC}"

# Update TypeScript files
TS_FILE="./sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only.ts"
if [ -f "$TS_FILE" ]; then
    # Get current SAID in file
    OLD_SAID=$(grep -oP 'const INVOICE_SCHEMA_SAID = "E[A-Za-z0-9_-]{43}"' "$TS_FILE" | grep -oP 'E[A-Za-z0-9_-]{43}' || echo "")
    if [ -n "$OLD_SAID" ] && [ "$OLD_SAID" != "$NEW_SAID" ]; then
        sed -i "s/const INVOICE_SCHEMA_SAID = \"$OLD_SAID\"/const INVOICE_SCHEMA_SAID = \"$NEW_SAID\"/" "$TS_FILE"
        echo -e "${GREEN}✓ Updated TypeScript: $OLD_SAID -> $NEW_SAID${NC}"
    fi
fi

# Update shell scripts
SHELL_SCRIPT="./run-all-buyerseller-4C-with-agents.sh"
if [ -f "$SHELL_SCRIPT" ]; then
    OLD_SAID=$(grep -oP 'INVOICE_SCHEMA_SAID="E[A-Za-z0-9_-]{43}"' "$SHELL_SCRIPT" | grep -oP 'E[A-Za-z0-9_-]{43}' || echo "")
    if [ -n "$OLD_SAID" ] && [ "$OLD_SAID" != "$NEW_SAID" ]; then
        sed -i "s/INVOICE_SCHEMA_SAID=\"$OLD_SAID\"/INVOICE_SCHEMA_SAID=\"$NEW_SAID\"/" "$SHELL_SCRIPT"
        echo -e "${GREEN}✓ Updated shell script: $OLD_SAID -> $NEW_SAID${NC}"
    fi
fi

# Update deploy.sh
if [ -f "deploy.sh" ]; then
    OLD_SAID=$(grep -oP 'INVOICE_SCHEMA_SAID="E[A-Za-z0-9_-]{43}"' "deploy.sh" | grep -oP 'E[A-Za-z0-9_-]{43}' || echo "")
    if [ -n "$OLD_SAID" ] && [ "$OLD_SAID" != "$NEW_SAID" ]; then
        sed -i "s/INVOICE_SCHEMA_SAID=\"$OLD_SAID\"/INVOICE_SCHEMA_SAID=\"$NEW_SAID\"/" "deploy.sh"
        sed -i "s/Schema SAID: $OLD_SAID/Schema SAID: $NEW_SAID/" "deploy.sh"
        echo -e "${GREEN}✓ Updated deploy.sh${NC}"
    fi
fi

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  SAIDification Complete!${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Schema SAID: ${GREEN}$NEW_SAID${NC}"
echo ""
echo "  Next steps:"
echo "    1. Rebuild tsx-shell: docker compose build tsx-shell --no-cache"
echo "    2. Restart schema container: docker compose restart schema"
echo "    3. Verify schema is cached: docker compose logs schema | grep 'caching schema'"
echo "    4. Run 4C script: ./run-all-buyerseller-4C-with-agents.sh"
echo ""
