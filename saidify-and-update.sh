#!/bin/bash
# saidify-and-update.sh
# 
# This script:
# 1. Runs the saidify script inside the tsx-shell container to compute the correct SAID
# 2. Updates the TypeScript code with the new SAID
# 3. Copies the SAIDified schema to the correct location
# 4. Restarts the schema container

set -e

cd "$(dirname "$0")"

echo "========================================"
echo "SAIDify Invoice Schema and Update Code"
echo "========================================"

# Make sure containers are running (service name is tsx-shell, container name is tsx_shell)
if ! docker compose ps tsx-shell | grep -q "Up"; then
    echo "ERROR: tsx-shell container is not running"
    echo "Please run ./deploy.sh first"
    exit 1
fi

# Create schemas directory in container if it doesn't exist
echo ""
echo "1. Setting up container..."
docker compose exec -T tsx-shell mkdir -p /vlei/sig-wallet/schemas

# Copy the schema file to the container for SAIDification
echo ""
echo "2. Copying schema to container..."
docker compose cp ./schemas/self-attested-invoice.json tsx-shell:/vlei/sig-wallet/schemas/

# Run the saidify script and capture the output
echo ""
echo "3. Running SAIDify script..."
OUTPUT=$(docker compose exec -T tsx-shell sh -c 'cd /vlei/sig-wallet && npx tsx src/tasks/saidify-schema.ts schemas/self-attested-invoice.json 2>&1' || true)

echo "$OUTPUT"

# Extract SAID from output
SAID=$(echo "$OUTPUT" | grep -E "^\s*SAID:" | head -1 | awk '{print $NF}')

# If that didn't work, try another pattern
if [ -z "$SAID" ]; then
    SAID=$(echo "$OUTPUT" | grep -oE 'E[A-Za-z0-9_-]{43}' | head -1)
fi

if [ -z "$SAID" ]; then
    echo ""
    echo "ERROR: Failed to extract SAID from output"
    echo "Trying to read from updated schema file..."
    
    # Try to get SAID from the file that was updated
    SAID=$(docker compose exec -T tsx-shell sh -c 'cat /vlei/sig-wallet/schemas/self-attested-invoice.json | grep "\$id" | head -1 | sed "s/.*\"\\\$id\": \"\([^\"]*\)\".*/\1/"')
fi

if [ -z "$SAID" ] || [ "$SAID" = "" ]; then
    echo "ERROR: Could not determine SAID"
    exit 1
fi

echo ""
echo "Computed SAID: $SAID"

# Copy the SAIDified schema back
echo ""
echo "4. Copying SAIDified schema back..."
docker compose cp tsx-shell:/vlei/sig-wallet/schemas/self-attested-invoice.json ./schemas/

# Also create the file with the SAID as filename
echo ""
echo "5. Creating schema file with SAID filename..."
cp ./schemas/self-attested-invoice.json "./schemas/${SAID}.json"

# Update the TypeScript code with the new SAID
echo ""
echo "6. Updating TypeScript code with new SAID..."
INVOICE_TS="./sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only.ts"

if [ -f "$INVOICE_TS" ]; then
    # Replace the old SAID with the new one
    sed -i "s/const INVOICE_SCHEMA_SAID = \"[^\"]*\"/const INVOICE_SCHEMA_SAID = \"$SAID\"/" "$INVOICE_TS"
    echo "  ✓ Updated $INVOICE_TS"
    grep "INVOICE_SCHEMA_SAID" "$INVOICE_TS" | head -1
else
    echo "  WARNING: $INVOICE_TS not found"
fi

# Also update the FIXED version if it exists
INVOICE_TS_FIXED="./sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only-FIXED.ts"
if [ -f "$INVOICE_TS_FIXED" ]; then
    sed -i "s/const INVOICE_SCHEMA_SAID = \"[^\"]*\"/const INVOICE_SCHEMA_SAID = \"$SAID\"/" "$INVOICE_TS_FIXED"
    echo "  ✓ Updated $INVOICE_TS_FIXED"
fi

# Rebuild the tsx-shell container with updated code
echo ""
echo "7. Rebuilding tsx-shell container..."
docker compose build --no-cache tsx-shell

# Restart containers to pick up changes
echo ""
echo "8. Restarting containers..."
docker compose up -d tsx-shell
docker compose restart schema
sleep 10

# Verify the schema is now cached
echo ""
echo "9. Verifying schema is cached..."
if docker compose logs schema 2>&1 | grep -q "caching schema $SAID"; then
    echo "  ✓ Schema $SAID is now cached!"
else
    echo "  Checking schema logs..."
    docker compose logs schema 2>&1 | grep -i "caching" | tail -20
fi

echo ""
echo "========================================"
echo "DONE!"
echo "========================================"
echo ""
echo "New SAID: $SAID"
echo ""
echo "Files updated:"
echo "  - ./schemas/self-attested-invoice.json"
echo "  - ./schemas/${SAID}.json"
echo "  - $INVOICE_TS"
echo ""
echo "Next steps:"
echo "  1. Run the 4C script: ./run-all-buyerseller-4C-with-agents.sh"
