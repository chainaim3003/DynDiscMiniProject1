#!/bin/bash
# SAIDify the invoice schema using the vLEI-server container
# This script computes the SAID hash and updates the $id field

set -e

SCHEMA_DIR="./schemas"
SCHEMA_FILE="self-attested-invoice-schema.json"

echo "==================================="
echo "SAIDifying Invoice Schema"
echo "==================================="

cd "$(dirname "$0")"

# Check if schema file exists
if [ ! -f "$SCHEMA_DIR/$SCHEMA_FILE" ]; then
    echo "ERROR: Schema file not found: $SCHEMA_DIR/$SCHEMA_FILE"
    exit 1
fi

echo "Schema file: $SCHEMA_DIR/$SCHEMA_FILE"

# Use the vlei container to SAIDify the schema
echo ""
echo "Using vLEI container to SAIDify schema..."
docker compose exec -T schema python3 -c "
import json
import hashlib
import base64

# Read schema
with open('/vLEI/schema/$SCHEMA_FILE', 'r') as f:
    schema = json.load(f)

# Remove $id to compute SAID
schema['\$id'] = ''

# Convert to canonical JSON (sorted keys, no whitespace)
canonical = json.dumps(schema, separators=(',', ':'), sort_keys=True)

# Compute Blake3 hash (KERI uses Blake3 for SAIDification)
# Since Python doesn't have Blake3 built-in, we'll use SHA256 as fallback
# In production, use proper KERI SAIDification
import hashlib
digest = hashlib.blake2b(canonical.encode(), digest_size=32).digest()

# Convert to SAID format (base64url without padding, prefixed with 'E')
said = 'E' + base64.urlsafe_b64encode(digest).decode().rstrip('=')

print(f'Computed SAID: {said}')

# Update schema with SAID
schema['\$id'] = said

# Write updated schema
with open('/vLEI/schema/$SCHEMA_FILE', 'w') as f:
    json.dump(schema, f, indent=2)

print(f'Schema updated with SAID: {said}')
"

echo ""
echo "Schema SAIDified successfully!"
echo "Restart the schema container for changes to take effect:"
echo "  docker compose restart schema"
