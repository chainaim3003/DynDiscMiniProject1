#!/bin/bash
################################################################################
# fix-invoice-credential-complete.sh
# 
# COMPLETE FIX for Invoice Credential Issuance
#
# Based on official vLEI documentation analysis, this script:
# 1. SAIDifies the invoice schema properly
# 2. Copies schema to the correct location with SAID-based filename
# 3. Restarts schema container to load the new schema
# 4. Verifies schema is accessible from both host AND KERIA
# 5. Provides diagnostic information
#
# Run this BEFORE attempting invoice credential issuance
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

cd "$(dirname "$0")"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     COMPLETE FIX FOR INVOICE CREDENTIAL ISSUANCE          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

################################################################################
# STEP 1: Check/Create Schema File
################################################################################

echo -e "${YELLOW}[Step 1/6] Checking schema files...${NC}"

SCHEMA_DIR="./schemas"
SCHEMA_SOURCE="$SCHEMA_DIR/self-attested-invoice.json"

if [ ! -d "$SCHEMA_DIR" ]; then
    echo -e "${RED}  ERROR: schemas directory not found${NC}"
    mkdir -p "$SCHEMA_DIR"
    echo -e "${GREEN}  Created schemas directory${NC}"
fi

if [ ! -f "$SCHEMA_SOURCE" ]; then
    echo -e "${YELLOW}  Creating default invoice schema...${NC}"
    cat > "$SCHEMA_SOURCE" << 'EOF'
{
  "$id": "",
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Self-Attested Invoice Credential",
  "description": "vLEI Credential for self-attested business invoice",
  "type": "object",
  "credentialType": "SelfAttestedInvoiceCredential",
  "version": "1.0.0",
  "properties": {
    "v": {
      "description": "Version string using ACDC conventions",
      "type": "string"
    },
    "d": {
      "description": "Credential SAID",
      "type": "string"
    },
    "u": {
      "description": "One time use nonce",
      "type": "string"
    },
    "i": {
      "description": "Issuer AID",
      "type": "string"
    },
    "ri": {
      "description": "Credential Registry SAID",
      "type": "string"
    },
    "s": {
      "description": "Schema SAID",
      "type": "string"
    },
    "a": {
      "oneOf": [
        {
          "description": "Attributes block SAID",
          "type": "string"
        },
        {
          "$id": "",
          "description": "Attributes block",
          "type": "object",
          "properties": {
            "d": {
              "description": "Attributes block SAID",
              "type": "string"
            },
            "i": {
              "description": "Issuee AID",
              "type": "string"
            },
            "dt": {
              "description": "Issuance date time ISO-8601",
              "type": "string",
              "format": "date-time"
            },
            "invoiceNumber": {
              "description": "Unique invoice identifier",
              "type": "string"
            },
            "invoiceDate": {
              "description": "Invoice date",
              "type": "string"
            },
            "dueDate": {
              "description": "Payment due date",
              "type": "string"
            },
            "sellerLEI": {
              "description": "Seller Legal Entity Identifier",
              "type": "string"
            },
            "buyerLEI": {
              "description": "Buyer Legal Entity Identifier",
              "type": "string"
            },
            "currency": {
              "description": "Currency code",
              "type": "string"
            },
            "totalAmount": {
              "description": "Total invoice amount",
              "type": "number"
            },
            "paymentMethod": {
              "description": "Payment method",
              "type": "string"
            },
            "paymentChainID": {
              "description": "Blockchain chain ID",
              "type": "string"
            },
            "paymentWalletAddress": {
              "description": "Payment wallet address",
              "type": "string"
            },
            "ref_uri": {
              "description": "Reference URI",
              "type": "string"
            },
            "paymentTerms": {
              "description": "Payment terms",
              "type": "string"
            }
          },
          "additionalProperties": false,
          "required": [
            "d",
            "i",
            "dt",
            "invoiceNumber",
            "sellerLEI",
            "buyerLEI",
            "currency",
            "totalAmount"
          ]
        }
      ]
    },
    "r": {
      "oneOf": [
        {
          "description": "Rules block SAID",
          "type": "string"
        },
        {
          "$id": "",
          "description": "Rules block",
          "type": "object",
          "properties": {
            "d": {
              "description": "Rules block SAID",
              "type": "string"
            },
            "usageDisclaimer": {
              "description": "Usage Disclaimer",
              "type": "object",
              "properties": {
                "l": {
                  "description": "Associated legal language",
                  "type": "string"
                }
              }
            },
            "selfAttestation": {
              "description": "Self attestation notice",
              "type": "object",
              "properties": {
                "l": {
                  "description": "Self attestation text",
                  "type": "string"
                }
              }
            }
          },
          "additionalProperties": false,
          "required": [
            "d",
            "usageDisclaimer",
            "selfAttestation"
          ]
        }
      ]
    }
  },
  "additionalProperties": false,
  "required": [
    "v",
    "d",
    "i",
    "ri",
    "s",
    "a"
  ]
}
EOF
    echo -e "${GREEN}  ✓ Created default schema${NC}"
fi

echo -e "${GREEN}  ✓ Schema source file exists${NC}"

################################################################################
# STEP 2: SAIDify the Schema using KLI
################################################################################

echo ""
echo -e "${YELLOW}[Step 2/6] SAIDifying schema...${NC}"

# Check if kli is available in vlei_shell container
if docker compose exec -T vlei-shell which kli > /dev/null 2>&1; then
    echo "  Using KLI in vlei-shell container..."
    
    # Copy schema to container and saidify
    docker compose cp "$SCHEMA_SOURCE" vlei-shell:/tmp/invoice-schema.json
    
    # Run kli saidify (note: this may need adjustment based on KLI version)
    SAIDIFY_OUTPUT=$(docker compose exec -T vlei-shell kli saidify --file /tmp/invoice-schema.json --label '$id' 2>&1) || true
    
    if echo "$SAIDIFY_OUTPUT" | grep -q "^E"; then
        NEW_SAID=$(echo "$SAIDIFY_OUTPUT" | grep "^E" | head -1 | tr -d '[:space:]')
        echo -e "${GREEN}  ✓ Schema SAIDified: $NEW_SAID${NC}"
        
        # Copy back the SAIDified schema
        docker compose cp vlei-shell:/tmp/invoice-schema.json "$SCHEMA_SOURCE"
    else
        echo -e "${YELLOW}  KLI saidify may not have worked, using existing SAID...${NC}"
        NEW_SAID=$(jq -r '.["$id"]' "$SCHEMA_SOURCE")
    fi
else
    echo "  KLI not available, checking existing schema SAID..."
    NEW_SAID=$(jq -r '.["$id"]' "$SCHEMA_SOURCE")
    
    if [ "$NEW_SAID" == "" ] || [ "$NEW_SAID" == "null" ]; then
        echo -e "${YELLOW}  Schema needs SAIDification. Using TypeScript SAIDify...${NC}"
        
        # Use the TypeScript saidify if available
        if docker compose exec -T tsx-shell test -f /vlei/sig-wallet/src/tasks/saidify-schema.ts; then
            docker compose exec -T tsx-shell npx tsx /vlei/sig-wallet/src/tasks/saidify-schema.ts "$SCHEMA_SOURCE" "$SCHEMA_SOURCE" 2>/dev/null || true
            NEW_SAID=$(jq -r '.["$id"]' "$SCHEMA_SOURCE")
        fi
        
        if [ "$NEW_SAID" == "" ] || [ "$NEW_SAID" == "null" ]; then
            # Fallback: use the known working SAID
            NEW_SAID="EHFQviJgaf6YTHbMPLheau8H1s3v31-ByTER_D49DLHY"
            echo -e "${YELLOW}  Using fallback SAID: $NEW_SAID${NC}"
            
            # Update the schema file with the fallback SAID
            jq --arg said "$NEW_SAID" '.["$id"] = $said' "$SCHEMA_SOURCE" > "${SCHEMA_SOURCE}.tmp" && mv "${SCHEMA_SOURCE}.tmp" "$SCHEMA_SOURCE"
        fi
    fi
fi

echo -e "${CYAN}  Schema SAID: $NEW_SAID${NC}"

################################################################################
# STEP 3: Create SAID-named copy of schema
################################################################################

echo ""
echo -e "${YELLOW}[Step 3/6] Creating SAID-named schema file...${NC}"

SAID_SCHEMA_FILE="$SCHEMA_DIR/${NEW_SAID}.json"
cp "$SCHEMA_SOURCE" "$SAID_SCHEMA_FILE"
echo -e "${GREEN}  ✓ Created: $SAID_SCHEMA_FILE${NC}"

# List all schema files
echo "  Schema files in $SCHEMA_DIR:"
ls -la "$SCHEMA_DIR"/*.json 2>/dev/null | while read line; do
    echo "    $line"
done

################################################################################
# STEP 4: Restart Schema Container
################################################################################

echo ""
echo -e "${YELLOW}[Step 4/6] Restarting schema container...${NC}"

# Check if schema container is running
if ! docker compose ps schema 2>/dev/null | grep -q "Up"; then
    echo -e "${RED}  ERROR: Schema container not running${NC}"
    echo "  Run: docker compose up -d"
    exit 1
fi

echo "  Restarting schema service..."
docker compose restart schema

echo "  Waiting for schema container to be healthy..."
MAX_WAIT=60
for i in $(seq 1 $MAX_WAIT); do
    if docker compose exec -T schema curl -sf http://127.0.0.1:7723/oobi/EBfdlu8R27Fbx-ehrqwImnK-8Cm79sqbAQ4MmvEAYqao > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ Schema container healthy (${i}s)${NC}"
        break
    fi
    sleep 1
    printf "."
done
echo ""

################################################################################
# STEP 5: Verify Schema Accessibility
################################################################################

echo ""
echo -e "${YELLOW}[Step 5/6] Verifying schema accessibility...${NC}"

# Check from host
echo "  Checking from host (localhost:7723)..."
HOST_CHECK=$(curl -sf "http://localhost:7723/oobi/$NEW_SAID" 2>/dev/null | head -c 100) || HOST_CHECK=""
if [ -n "$HOST_CHECK" ]; then
    echo -e "${GREEN}    ✓ Accessible from host${NC}"
else
    echo -e "${RED}    ✗ NOT accessible from host${NC}"
fi

# Check from KERIA container (CRITICAL)
echo "  Checking from KERIA container (http://schema:7723)..."
KERIA_CHECK=$(docker compose exec -T keria wget -qO- --timeout=10 "http://schema:7723/oobi/$NEW_SAID" 2>/dev/null | head -c 100) || KERIA_CHECK=""
if [ -n "$KERIA_CHECK" ]; then
    echo -e "${GREEN}    ✓ Accessible from KERIA (CRITICAL)${NC}"
else
    echo -e "${RED}    ✗ NOT accessible from KERIA (PROBLEM!)${NC}"
    
    # Diagnostic
    echo ""
    echo -e "${YELLOW}  Running diagnostics...${NC}"
    echo "    Schema container logs:"
    docker compose logs schema --tail=20 2>/dev/null | head -10
    
    echo ""
    echo "    Files in /vLEI/schema inside container:"
    docker compose exec -T schema ls -la /vLEI/schema/*.json 2>/dev/null | head -5 || echo "    Could not list"
    
    echo ""
    echo "    Files in /vLEI/custom-schema inside container:"
    docker compose exec -T schema ls -la /vLEI/custom-schema/*.json 2>/dev/null | head -5 || echo "    Could not list"
fi

# Compare with working schemas
echo ""
echo "  Comparing with known working schemas:"
for KNOWN_SAID in "EBNaNu-M9P5cgrnfl2Fvymy4E_jvxxyjb70PRtiANlJy" "ENPXp1vQzRF6JwIuS-mp2U8Uf1MoADoP_GqQ62VsDZWY"; do
    KNOWN_CHECK=$(docker compose exec -T keria wget -qO- --timeout=5 "http://schema:7723/oobi/$KNOWN_SAID" 2>/dev/null | head -c 50) || KNOWN_CHECK=""
    if [ -n "$KNOWN_CHECK" ]; then
        echo -e "${GREEN}    ✓ $KNOWN_SAID (working)${NC}"
    else
        echo -e "${RED}    ✗ $KNOWN_SAID (not working!)${NC}"
    fi
done

################################################################################
# STEP 6: Summary and Next Steps
################################################################################

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                        SUMMARY                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Schema SAID: ${CYAN}$NEW_SAID${NC}"
echo -e "  Schema OOBI: ${CYAN}http://schema:7723/oobi/$NEW_SAID${NC}"
echo ""

if [ -n "$KERIA_CHECK" ]; then
    echo -e "${GREEN}  ✓ Schema is accessible from KERIA - credential issuance should work${NC}"
    echo ""
    echo -e "  To issue invoice credentials, update your script to use:"
    echo -e "    INVOICE_SCHEMA_SAID=\"${CYAN}$NEW_SAID${NC}\""
    echo ""
    echo -e "  Or run the V2 issuance script:"
    echo -e "    ${CYAN}invoice-acdc-issue-v2.ts${NC}"
else
    echo -e "${RED}  ✗ Schema NOT accessible from KERIA - issuance will fail${NC}"
    echo ""
    echo -e "  Possible fixes:"
    echo -e "  1. Check Docker network: docker network inspect vlei_workshop"
    echo -e "  2. Restart all services: docker compose down && docker compose up -d"
    echo -e "  3. Verify schema file format matches official vLEI schemas"
    echo -e "  4. Check schema container logs: docker compose logs schema"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

# Export the SAID for use in other scripts
echo "$NEW_SAID" > /tmp/invoice-schema-said.txt
