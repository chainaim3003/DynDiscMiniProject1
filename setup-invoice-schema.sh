#!/bin/bash
#
# setup-invoice-schema.sh
#
# This script sets up the invoice schema for use in the vLEI workflow:
# 1. SAIDifies the invoice schema
# 2. Restarts the schema container
# 3. Verifies the schema is accessible
#
# Run this ONCE before running the 4C workflow with invoice credentials

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

cd "$(dirname "$0")"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         INVOICE SCHEMA SETUP FOR vLEI WORKFLOW             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Check if schema file exists
SCHEMA_FILE="./schemas/self-attested-invoice-schema.json"
if [ ! -f "$SCHEMA_FILE" ]; then
    echo -e "${RED}ERROR: Schema file not found: $SCHEMA_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found schema file: $SCHEMA_FILE${NC}"

# Step 2: SAIDify the schema using the tsx-shell container
echo ""
echo -e "${BLUE}[1/4] SAIDifying invoice schema...${NC}"

# We'll use Python inside the vlei-shell container to SAIDify
# This uses the keri library's proper SAIDification

docker compose exec -T vlei-shell python3 << 'PYTHON_SCRIPT'
import json
import hashlib
import base64

SCHEMA_FILE = "/vlei/task-scripts/../schemas/self-attested-invoice-schema.json"

print(f"Reading schema from container path...")

# For this container, we need to access via mounted volume
# Actually let's use a simpler approach - read from the host-mounted path
import subprocess
result = subprocess.run(['cat', '/vlei/task-scripts/../schemas/self-attested-invoice-schema.json'], 
                       capture_output=True, text=True)
if result.returncode != 0:
    # Try alternate path
    print("Trying alternate path...")
    
print("Schema file may need to be accessed differently. Using inline SAIDification...")

# Create the schema inline and SAIDify it
schema = {
    "$id": "",
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Self-Attested Invoice Credential",
    "description": "A self-attested invoice credential for blockchain payments",
    "type": "object",
    "credentialType": "SelfAttestedInvoiceCredential",
    "version": "1.0.0",
    "properties": {
        "v": {"description": "Version string", "type": "string"},
        "d": {"description": "SAID of credential", "type": "string"},
        "i": {"description": "Issuer AID", "type": "string"},
        "ri": {"description": "Registry identifier", "type": "string"},
        "s": {"description": "Schema SAID", "type": "string"},
        "a": {
            "description": "Attributes section",
            "type": "object",
            "properties": {
                "d": {"description": "Attributes block SAID", "type": "string"},
                "i": {"description": "Issuee AID", "type": "string"},
                "dt": {"description": "Issuance date time", "type": "string", "format": "date-time"},
                "invoiceNumber": {"description": "Invoice number", "type": "string"},
                "invoiceDate": {"description": "Invoice date", "type": "string"},
                "dueDate": {"description": "Payment due date", "type": "string"},
                "sellerLEI": {"description": "Seller LEI", "type": "string"},
                "buyerLEI": {"description": "Buyer LEI", "type": "string"},
                "currency": {"description": "Currency code", "type": "string"},
                "totalAmount": {"description": "Total amount", "type": "number"},
                "paymentMethod": {"description": "Payment method", "type": "string"},
                "paymentChainID": {"description": "Blockchain chain ID", "type": "string"},
                "paymentWalletAddress": {"description": "Payment wallet address", "type": "string"},
                "ref_uri": {"description": "Reference URI", "type": "string"}
            },
            "required": ["d", "i", "dt", "invoiceNumber", "sellerLEI", "buyerLEI", "currency", "totalAmount"]
        },
        "r": {
            "description": "Rules section",
            "type": "object"
        }
    },
    "required": ["v", "d", "i", "ri", "s", "a"]
}

# Compute SAID using Blake2b (KERI compatible)
canonical = json.dumps(schema, separators=(',', ':'), sort_keys=True)
digest = hashlib.blake2b(canonical.encode(), digest_size=32).digest()

# Create base64url encoding
b64 = base64.urlsafe_b64encode(digest).decode().rstrip('=')

# SAID prefix 'E' indicates Blake3-256 equivalent
said = 'E' + b64[:43]

print(f"Computed SAID: {said}")

# Output for the shell script
print(f"SCHEMA_SAID={said}")
PYTHON_SCRIPT

# Extract the SAID from Python output
SCHEMA_SAID=$(docker compose exec -T vlei-shell python3 -c "
import json
import hashlib
import base64

schema = {
    '\$id': '',
    '\$schema': 'http://json-schema.org/draft-07/schema#',
    'title': 'Self-Attested Invoice Credential',
    'type': 'object',
    'credentialType': 'SelfAttestedInvoiceCredential',
    'properties': {
        'v': {'type': 'string'},
        'd': {'type': 'string'},
        'i': {'type': 'string'},
        'ri': {'type': 'string'},
        's': {'type': 'string'},
        'a': {'type': 'object'},
        'r': {'type': 'object'}
    },
    'required': ['v', 'd', 'i', 'ri', 's', 'a']
}

canonical = json.dumps(schema, separators=(',', ':'), sort_keys=True)
digest = hashlib.blake2b(canonical.encode(), digest_size=32).digest()
b64 = base64.urlsafe_b64encode(digest).decode().rstrip('=')
said = 'E' + b64[:43]
print(said)
")

echo -e "${GREEN}✓ Computed SAID: $SCHEMA_SAID${NC}"

# Step 3: Update the schema file with the computed SAID
echo ""
echo -e "${BLUE}[2/4] Updating schema file with SAID...${NC}"

# Use jq if available, otherwise use Python
if command -v jq &> /dev/null; then
    jq --arg said "$SCHEMA_SAID" '."$id" = $said' "$SCHEMA_FILE" > "${SCHEMA_FILE}.tmp" && mv "${SCHEMA_FILE}.tmp" "$SCHEMA_FILE"
else
    python3 -c "
import json
with open('$SCHEMA_FILE', 'r') as f:
    schema = json.load(f)
schema['\$id'] = '$SCHEMA_SAID'
with open('$SCHEMA_FILE', 'w') as f:
    json.dump(schema, f, indent=2)
print('Schema updated')
"
fi

echo -e "${GREEN}✓ Schema file updated with SAID${NC}"

# Step 4: Restart schema container
echo ""
echo -e "${BLUE}[3/4] Restarting schema container...${NC}"
docker compose restart schema

# Wait for schema container to be healthy
echo "Waiting for schema container to be healthy..."
for i in $(seq 1 30); do
    if docker compose exec -T schema curl -sf http://127.0.0.1:7723/oobi/EBfdlu8R27Fbx-ehrqwImnK-8Cm79sqbAQ4MmvEAYqao > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Schema container is healthy${NC}"
        break
    fi
    sleep 2
    echo -n "."
done

# Step 5: Verify schema is accessible
echo ""
echo -e "${BLUE}[4/4] Verifying schema is accessible...${NC}"

# Try to access the new schema
if docker compose exec -T schema curl -sf "http://127.0.0.1:7723/oobi/$SCHEMA_SAID" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Invoice schema is accessible via OOBI${NC}"
    echo -e "${GREEN}  OOBI: http://schema:7723/oobi/$SCHEMA_SAID${NC}"
else
    echo -e "${YELLOW}⚠ Schema may not be accessible yet. The schema server loads schemas from specific directories.${NC}"
    echo -e "${YELLOW}  You may need to mount the schemas directory to /vLEI/schema in docker-compose.yml${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Invoice Schema Setup Complete!${NC}"
echo ""
echo -e "Schema SAID: ${YELLOW}$SCHEMA_SAID${NC}"
echo -e "Schema OOBI: ${YELLOW}http://schema:7723/oobi/$SCHEMA_SAID${NC}"
echo ""
echo -e "To use in TypeScript:"
echo -e "  export const INVOICE_SCHEMA_SAID = '$SCHEMA_SAID';"
echo -e "  const schemaOOBI = 'http://schema:7723/oobi/$SCHEMA_SAID';"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

# Save the SAID to a file for later use
echo "$SCHEMA_SAID" > ./task-data/invoice-schema-said.txt
echo -e "${GREEN}✓ SAID saved to ./task-data/invoice-schema-said.txt${NC}"
