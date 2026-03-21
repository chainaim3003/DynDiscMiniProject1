#!/bin/bash
#
# saidify-with-docker.sh
#
# This script SAIDifies the invoice schema by running keripy inside a Docker container.
# This ensures we use the exact same SAIDification as the vLEI-server.
#
# Usage: ./saidify-with-docker.sh
#
# Prerequisites: Docker must be running

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

cd "$(dirname "$0")"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    SAIDify Invoice Schema using Docker (keripy method)     ║${NC}"
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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Docker is not running${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is running${NC}"

# Get the actual schema image being used
SCHEMA_IMAGE=$(docker inspect legentvlei-schema-1 --format='{{.Config.Image}}' 2>/dev/null || echo "gleif/vlei:1.0.1")
echo -e "${GREEN}✓ Using image: $SCHEMA_IMAGE${NC}"

# Step 1: Run SAIDification inside the running schema container
echo ""
echo -e "${BLUE}[1/4] SAIDifying schema using keripy in Docker...${NC}"

# Create a temporary Python script using the CORRECT keripy API
# NOTE: /vLEI/custom-schema is read-only, so we write output to /tmp
cat > /tmp/saidify_schema.py << 'PYTHON_SCRIPT'
import json
import sys
import os

# Paths inside container - read from mounted volume, write to /tmp
INPUT_SCHEMA = "/vLEI/custom-schema/self-attested-invoice.json"
OUTPUT_SCHEMA = "/tmp/self-attested-invoice-saidified.json"
OUTPUT_SAID_FILE = "/tmp/invoice-schema-said.txt"

from keri.core import scheming

def saidify_schema(schema_path):
    """SAIDify a schema using keripy's Schemer class"""
    
    print(f"Reading schema from {schema_path}...")
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    original_id = schema.get('$id', '(empty)')
    print(f"Original $id: {original_id}")
    
    # First, SAIDify any nested blocks with $id (like 'a' attributes block)
    print("\nProcessing nested $id fields...")
    
    # Check for nested $id in properties.a (attributes block)
    if 'properties' in schema and 'a' in schema['properties']:
        a_block = schema['properties']['a']
        if isinstance(a_block, dict) and '$id' in a_block:
            print("  Found $id in 'a' (attributes) block, SAIDifying...")
            try:
                a_schemer = scheming.Schemer(sed=a_block)
                schema['properties']['a'] = a_schemer.sed
                print(f"    Attributes block SAID: {a_schemer.said}")
            except Exception as e:
                print(f"    Warning: Could not SAIDify 'a' block: {e}")
    
    # Check for nested $id in properties.r (rules block)
    if 'properties' in schema and 'r' in schema['properties']:
        r_block = schema['properties']['r']
        if isinstance(r_block, dict) and '$id' in r_block:
            print("  Found $id in 'r' (rules) block, SAIDifying...")
            try:
                r_schemer = scheming.Schemer(sed=r_block)
                schema['properties']['r'] = r_schemer.sed
                print(f"    Rules block SAID: {r_schemer.said}")
            except Exception as e:
                print(f"    Warning: Could not SAIDify 'r' block: {e}")
    
    # Now SAIDify the top-level schema
    print("\nComputing main schema SAID using keripy Schemer...")
    schemer = scheming.Schemer(sed=schema)
    
    print(f"  Main schema SAID: {schemer.said}")
    
    # schemer.sed contains the schema with $id populated
    return schemer.sed, schemer.said

try:
    # SAIDify the schema
    schema_saidified, said = saidify_schema(INPUT_SCHEMA)
    
    # Write updated schema to /tmp (writable)
    print(f"\nWriting SAIDified schema to {OUTPUT_SCHEMA}...")
    with open(OUTPUT_SCHEMA, 'w') as f:
        json.dump(schema_saidified, f, indent=2)
    
    # Also write SAID-named version
    said_filename = f"/tmp/{said}.json"
    print(f"Creating SAID-named copy: {said_filename}")
    with open(said_filename, 'w') as f:
        json.dump(schema_saidified, f, indent=2)
    
    # Write SAID to output file for shell script to read
    with open(OUTPUT_SAID_FILE, 'w') as f:
        f.write(said)
    
    print(f"\n✓ Schema SAIDified successfully!")
    print(f"  SAID: {said}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_SCRIPT

# Copy the script into the running container and execute it
docker cp /tmp/saidify_schema.py legentvlei-schema-1:/tmp/saidify_schema.py

# Run the SAIDification inside the existing container
docker exec legentvlei-schema-1 python3 /tmp/saidify_schema.py

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to SAIDify schema in Docker${NC}"
    exit 1
fi

# Read the computed SAID from container
SCHEMA_SAID=$(docker exec legentvlei-schema-1 cat /tmp/invoice-schema-said.txt)
echo "$SCHEMA_SAID" > ./task-data/invoice-schema-said.txt

echo ""
echo -e "${GREEN}✓ Schema SAID computed: ${YELLOW}$SCHEMA_SAID${NC}"

# Step 2: Copy SAIDified schema from container back to host
echo ""
echo -e "${BLUE}[2/4] Copying SAIDified schema from container to host...${NC}"

docker cp legentvlei-schema-1:/tmp/self-attested-invoice-saidified.json ./schemas/self-attested-invoice.json
docker cp "legentvlei-schema-1:/tmp/${SCHEMA_SAID}.json" "./schemas/${SCHEMA_SAID}.json"

echo -e "${GREEN}✓ Updated ./schemas/self-attested-invoice.json${NC}"
echo -e "${GREEN}✓ Created ./schemas/${SCHEMA_SAID}.json${NC}"

# Step 3: Update deploy.sh with the new SAID
echo ""
echo -e "${BLUE}[3/4] Updating deploy.sh and TypeScript files...${NC}"

if [ -f "./deploy.sh" ]; then
    sed -i "s/INVOICE_SCHEMA_SAID=\"E[A-Za-z0-9_-]*\"/INVOICE_SCHEMA_SAID=\"$SCHEMA_SAID\"/" ./deploy.sh
    echo -e "  Updated: deploy.sh"
fi

# Update TypeScript files
INVOICE_TS_FILE="./sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only.ts"
if [ -f "$INVOICE_TS_FILE" ]; then
    sed -i "s/const INVOICE_SCHEMA_SAID = \"E[A-Za-z0-9_-]\{43\}\"/const INVOICE_SCHEMA_SAID = \"$SCHEMA_SAID\"/" "$INVOICE_TS_FILE" 2>/dev/null || true
    echo -e "  Updated: $INVOICE_TS_FILE"
fi

# Update old SAIDs in all TypeScript files
for ts_file in $(find ./sig-wallet/src -name "*.ts" -type f 2>/dev/null); do
    if grep -q "EEwSXh_s-i7NBmFrNSTJDC5K9Xw6W-YvEi-Cl9-JaAFb" "$ts_file" 2>/dev/null; then
        sed -i "s/EEwSXh_s-i7NBmFrNSTJDC5K9Xw6W-YvEi-Cl9-JaAFb/$SCHEMA_SAID/g" "$ts_file" 2>/dev/null || true
        echo -e "  Updated: $ts_file"
    fi
    if grep -q "EtX9ETBh-yqstm0t6Otzl3P4WZkPlZQMTt1cVGGa1Bhk" "$ts_file" 2>/dev/null; then
        sed -i "s/EtX9ETBh-yqstm0t6Otzl3P4WZkPlZQMTt1cVGGa1Bhk/$SCHEMA_SAID/g" "$ts_file" 2>/dev/null || true
        echo -e "  Updated: $ts_file"
    fi
done

# Also update invoiceConfig.json
if [ -f "./appconfig/invoiceConfig.json" ]; then
    sed -i "s/\"schemaSAID\": \"E[A-Za-z0-9_-]*\"/\"schemaSAID\": \"$SCHEMA_SAID\"/" ./appconfig/invoiceConfig.json 2>/dev/null || true
    echo -e "  Updated: appconfig/invoiceConfig.json"
fi

# UPDATE THE SINGLE SOURCE OF TRUTH CONFIG FILE
if [ -f "./appconfig/schemaSaids.json" ]; then
    # Use jq if available, otherwise use sed
    if command -v jq &> /dev/null; then
        jq --arg said "$SCHEMA_SAID" '.invoiceSchema.said = $said' ./appconfig/schemaSaids.json > /tmp/schemaSaids.tmp && mv /tmp/schemaSaids.tmp ./appconfig/schemaSaids.json
    else
        sed -i "s/\"said\": \"E[A-Za-z0-9_-]*\"/\"said\": \"$SCHEMA_SAID\"/" ./appconfig/schemaSaids.json 2>/dev/null || true
    fi
    echo -e "  ${GREEN}✓ Updated: appconfig/schemaSaids.json (SINGLE SOURCE OF TRUTH)${NC}"
else
    # Create the config file if it doesn't exist
    cat > ./appconfig/schemaSaids.json << EOF
{
  "comment": "THIS FILE IS THE SINGLE SOURCE OF TRUTH FOR SCHEMA SAIDs",
  "invoiceSchema": {
    "said": "$SCHEMA_SAID",
    "file": "./schemas/self-attested-invoice.json"
  }
}
EOF
    echo -e "  ${GREEN}✓ Created: appconfig/schemaSaids.json (SINGLE SOURCE OF TRUTH)${NC}"
fi

echo -e "${GREEN}✓ Configuration files updated${NC}"

# Step 4: Copy schemas into running container's schema directory
echo ""
echo -e "${BLUE}[4/4] Installing schemas into container...${NC}"

# The /vLEI/schema directory should be writable (it's the server's schema cache)
docker exec legentvlei-schema-1 cp /tmp/self-attested-invoice-saidified.json /vLEI/schema/self-attested-invoice.json 2>/dev/null || true
docker exec legentvlei-schema-1 cp "/tmp/${SCHEMA_SAID}.json" "/vLEI/schema/${SCHEMA_SAID}.json" 2>/dev/null || true

# Verify
if docker exec legentvlei-schema-1 ls "/vLEI/schema/${SCHEMA_SAID}.json" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Schema installed in container at /vLEI/schema/${SCHEMA_SAID}.json${NC}"
else
    echo -e "${YELLOW}⚠ Could not install in /vLEI/schema/ - will need container restart${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Schema SAIDification Complete!${NC}"
echo ""
echo -e "Schema SAID: ${YELLOW}$SCHEMA_SAID${NC}"
echo -e "Schema file: ${YELLOW}./schemas/self-attested-invoice.json${NC}"
echo -e "SAID file:   ${YELLOW}./schemas/${SCHEMA_SAID}.json${NC}"
echo ""
echo -e "${YELLOW}IMPORTANT: Next steps to complete the process:${NC}"
echo ""
echo -e "  ${BLUE}# 1. Restart schema container to reload schemas${NC}"
echo -e "  docker compose restart schema && sleep 15"
echo ""
echo -e "  ${BLUE}# 2. Verify schema is accessible${NC}"
echo -e "  curl -s http://127.0.0.1:7723/oobi/$SCHEMA_SAID | head -c 200"
echo ""
echo -e "  ${BLUE}# 3. Rebuild tsx-shell with new SAID${NC}"
echo -e "  docker compose build --no-cache tsx-shell"
echo ""
echo -e "  ${BLUE}# 4. Restart tsx-shell${NC}"
echo -e "  docker compose up -d tsx-shell && sleep 10"
echo ""
echo -e "  ${BLUE}# 5. Run the workflow${NC}"
echo -e "  ./run-all-buyerseller-4C-with-agents.sh"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
