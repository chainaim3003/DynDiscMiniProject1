#!/bin/bash
#
# saidify-and-prepare-schema.sh
#
# This script SAIDifies the invoice schema using keripy's official Schemer class.
# It runs the SAIDification inside the Docker container to ensure compatibility
# with the vLEI-server's SAID verification.
#
# Usage: ./saidify-and-prepare-schema.sh
#
# This should be called from setup.sh or run manually after copying files from Windows.

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

cd "$(dirname "$0")"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         SAIDify Invoice Schema (Pre-Deploy Step)           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

SCHEMA_FILE="./schemas/self-attested-invoice.json"
SCHEMA_DIR="./schemas"

# Create task-data directory if needed
mkdir -p ./task-data

# Check if schema file exists
if [ ! -f "$SCHEMA_FILE" ]; then
    echo -e "${RED}ERROR: Schema file not found: $SCHEMA_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found schema file: $SCHEMA_FILE${NC}"

# Step 1: SAIDify using keripy's Schemer class (official method)
echo ""
echo -e "${BLUE}[1/3] Computing SAID for schema using keripy Schemer...${NC}"

# Use Python with keripy to SAIDify (the official way per WebOfTrust/vLEI)
python3 << 'PYTHON_SCRIPT'
import json
import sys
import os

SCHEMA_FILE = "./schemas/self-attested-invoice.json"
SCHEMA_DIR = "./schemas"
TASK_DATA_DIR = "./task-data"

try:
    from keri.core import scheming
    from keri.core import coring
    HAS_KERIPY = True
    print("Using keripy's Schemer for SAIDification (official method)")
except ImportError:
    HAS_KERIPY = False
    print("keripy not available, using fallback method")

def saidify_with_keripy(schema_data):
    """Use keripy's official Schemer class to SAIDify"""
    # Use the generating module's populateSAIDS function
    from keri.core import scheming, generating
    
    # First populate nested SAIDs
    schema_with_saids = generating.populateSAIDS(schema_data)
    
    # Create a Schemer to get the final SAID
    schemer = scheming.Schemer(sed=schema_with_saids)
    
    return schemer.sed, schemer.said

def saidify_nested_blocks(schema):
    """SAIDify nested blocks ($id in 'a' and 'r' properties) using keripy"""
    from keri.core import scheming
    
    # Handle nested $id in 'a' (attributes) block
    if 'properties' in schema and 'a' in schema['properties']:
        a_prop = schema['properties']['a']
        if 'oneOf' in a_prop:
            for i, item in enumerate(a_prop['oneOf']):
                if isinstance(item, dict) and '$id' in item:
                    # Use Schemer for nested block
                    nested_schemer = scheming.Schemer(sed=item)
                    a_prop['oneOf'][i] = nested_schemer.sed
                    print(f"  Nested 'a' block SAID: {nested_schemer.said}")
    
    # Handle nested $id in 'r' (rules) block
    if 'properties' in schema and 'r' in schema['properties']:
        r_prop = schema['properties']['r']
        if 'oneOf' in r_prop:
            for i, item in enumerate(r_prop['oneOf']):
                if isinstance(item, dict) and '$id' in item:
                    # Use Schemer for nested block
                    nested_schemer = scheming.Schemer(sed=item)
                    r_prop['oneOf'][i] = nested_schemer.sed
                    print(f"  Nested 'r' block SAID: {nested_schemer.said}")
    
    # Now compute the main schema SAID
    main_schemer = scheming.Schemer(sed=schema)
    print(f"  Main schema SAID: {main_schemer.said}")
    
    return main_schemer.sed, main_schemer.said

# Read schema
print(f"Reading schema from {SCHEMA_FILE}...")
with open(SCHEMA_FILE, 'r') as f:
    schema = json.load(f)

original_id = schema.get('$id', '(empty)')
print(f"Original $id: {original_id}")
print("")

# SAIDify using keripy if available
print("Computing SAIDs...")
if HAS_KERIPY:
    try:
        schema_saidified, said = saidify_nested_blocks(schema)
    except Exception as e:
        print(f"Error with nested SAIDification: {e}")
        print("Trying simpler approach...")
        schema_saidified, said = saidify_with_keripy(schema)
else:
    # Fallback - use simple hash (may not match vLEI-server)
    print("WARNING: keripy not available. Schema SAID may not match vLEI-server!")
    import hashlib
    import base64
    
    # Simple fallback
    schema['$id'] = ""
    canonical = json.dumps(schema, separators=(',', ':'), sort_keys=True)
    digest = hashlib.blake2b(canonical.encode('utf-8'), digest_size=32).digest()
    b64 = base64.urlsafe_b64encode(digest).decode('ascii').rstrip('=')
    said = 'E' + b64[:43]
    schema['$id'] = said
    schema_saidified = schema

# Write updated schema
print(f"\nWriting updated schema to {SCHEMA_FILE}...")
with open(SCHEMA_FILE, 'w') as f:
    json.dump(schema_saidified, f, indent=2)

# Create a copy with the SAID as filename (required by vLEI-server)
said_filename = f"{SCHEMA_DIR}/{said}.json"
print(f"Creating SAID-named copy: {said_filename}")
with open(said_filename, 'w') as f:
    json.dump(schema_saidified, f, indent=2)

# Save SAID for other scripts to use
os.makedirs(TASK_DATA_DIR, exist_ok=True)
with open(f'{TASK_DATA_DIR}/invoice-schema-said.txt', 'w') as f:
    f.write(said)

print(f"\n✓ Schema SAIDified successfully!")
print(f"  SAID: {said}")
print(f"  Saved to: {TASK_DATA_DIR}/invoice-schema-said.txt")

PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to SAIDify schema${NC}"
    echo -e "${YELLOW}Note: keripy may not be installed. Install with: pip install keri${NC}"
    exit 1
fi

# Step 2: Read the computed SAID
SCHEMA_SAID=$(cat ./task-data/invoice-schema-said.txt)
echo ""
echo -e "${GREEN}✓ Schema SAID computed: ${YELLOW}$SCHEMA_SAID${NC}"

# Step 3: Update deploy.sh with the new SAID
echo ""
echo -e "${BLUE}[2/3] Updating deploy.sh with new SAID...${NC}"

if [ -f "./deploy.sh" ]; then
    # Update the INVOICE_SCHEMA_SAID variable in deploy.sh
    sed -i "s/INVOICE_SCHEMA_SAID=\"E[A-Za-z0-9_-]*\"/INVOICE_SCHEMA_SAID=\"$SCHEMA_SAID\"/" ./deploy.sh
    echo -e "${GREEN}✓ Updated deploy.sh${NC}"
else
    echo -e "${YELLOW}⚠ deploy.sh not found, skipping update${NC}"
fi

# Step 4: Update any TypeScript files that reference the schema SAID
echo ""
echo -e "${BLUE}[3/3] Updating TypeScript files with new SAID...${NC}"

# Find and update TypeScript files that have the INVOICE_SCHEMA_SAID constant
INVOICE_TS_FILE="./sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only.ts"
if [ -f "$INVOICE_TS_FILE" ]; then
    # Update the INVOICE_SCHEMA_SAID constant
    sed -i "s/const INVOICE_SCHEMA_SAID = \"E[A-Za-z0-9_-]\{43\}\"/const INVOICE_SCHEMA_SAID = \"$SCHEMA_SAID\"/" "$INVOICE_TS_FILE" 2>/dev/null || true
    echo -e "  Updated: $INVOICE_TS_FILE"
    echo -e "${GREEN}✓ TypeScript invoice file updated${NC}"
else
    echo -e "${YELLOW}⚠ Invoice TypeScript file not found${NC}"
fi

# Also update any other TypeScript files that reference old SAIDs
echo "  Searching for other TypeScript files with old SAID..."
for ts_file in $(find ./sig-wallet/src -name "*.ts" -type f 2>/dev/null); do
    # Look for any SAID pattern and update if found
    if grep -q "EEwSXh_s-i7NBmFrNSTJDC5K9Xw6W-YvEi-Cl9-JaAFb" "$ts_file" 2>/dev/null; then
        sed -i "s/EEwSXh_s-i7NBmFrNSTJDC5K9Xw6W-YvEi-Cl9-JaAFb/$SCHEMA_SAID/g" "$ts_file" 2>/dev/null || true
        echo -e "  Updated: $ts_file"
    fi
    if grep -q "EtX9ETBh-yqstm0t6Otzl3P4WZkPlZQMTt1cVGGa1Bhk" "$ts_file" 2>/dev/null; then
        sed -i "s/EtX9ETBh-yqstm0t6Otzl3P4WZkPlZQMTt1cVGGa1Bhk/$SCHEMA_SAID/g" "$ts_file" 2>/dev/null || true
        echo -e "  Updated: $ts_file"
    fi
done

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Schema SAIDification Complete!${NC}"
echo ""
echo -e "Schema SAID: ${YELLOW}$SCHEMA_SAID${NC}"
echo -e "Schema file: ${YELLOW}./schemas/self-attested-invoice.json${NC}"
echo -e "SAID file:   ${YELLOW}./schemas/${SCHEMA_SAID}.json${NC}"
echo ""
echo -e "Next: Run ${YELLOW}./deploy.sh${NC} to start Docker containers"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
