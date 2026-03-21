#!/bin/bash
################################################################################
# invoice-acdc-issue.sh
# Issue invoice credential from OOR holder to buyer agent
################################################################################

set -e

ISSUER_ALIAS="${1:-Jupiter_Chief_Sales_Officer}"
HOLDER_ALIAS="${2:-tommyBuyerAgent}"
INVOICE_CONFIG_FILE="${3:-./appconfig/invoiceConfig.json}"

# Source environment variables
source ./task-scripts/workshop-env-vars.sh

# Get passcode for issuer
PASSCODE=$(get_passcode "$ISSUER_ALIAS")

# Load configuration
REGISTRY_NAME=$(jq -r '.invoice.registryName' "$INVOICE_CONFIG_FILE")
OOR_SCHEMA_SAID=$(jq -r '.invoice.oorSchemaSaid' "$INVOICE_CONFIG_FILE")

# Get OOR credential SAID from issuer's credential info
OOR_CRED_FILE="./task-data/${ISSUER_ALIAS}-oor-credential-info.json"
if [ ! -f "$OOR_CRED_FILE" ]; then
    echo "ERROR: OOR credential file not found: $OOR_CRED_FILE"
    echo "The issuer must have an OOR credential before issuing invoices"
    exit 1
fi
OOR_CREDENTIAL_SAID=$(cat "$OOR_CRED_FILE" | jq -r '.said')

# Get holder AID
HOLDER_INFO_FILE="./task-data/${HOLDER_ALIAS}-info.json"
if [ ! -f "$HOLDER_INFO_FILE" ]; then
    echo "ERROR: Holder info file not found: $HOLDER_INFO_FILE"
    echo "The holder AID must exist before issuing invoices"
    exit 1
fi
HOLDER_AID=$(cat "$HOLDER_INFO_FILE" | jq -r '.aid')

# Get invoice data from config
SELLER_LEI=$(jq -r '.invoice.issuer.lei' "$INVOICE_CONFIG_FILE")
BUYER_LEI=$(jq -r '.invoice.holder.lei' "$INVOICE_CONFIG_FILE")
INVOICE_DATA=$(jq -c '.invoice.sampleInvoice' "$INVOICE_CONFIG_FILE")

# Add holder AID and LEIs to invoice data
INVOICE_DATA=$(echo "$INVOICE_DATA" | jq -c \
  --arg holder "$HOLDER_AID" \
  --arg sellerLEI "$SELLER_LEI" \
  --arg buyerLEI "$BUYER_LEI" \
  --arg dt "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  '. + {i: $holder, sellerLEI: $sellerLEI, buyerLEI: $buyerLEI, dt: $dt}')

# Note: Invoice schema SAID will need to be set after schema is published
# For now, using a placeholder that must be updated
INVOICE_SCHEMA_SAID="EInvoiceSchemaPlaceholder"

OUTPUT_PATH="./task-data/${ISSUER_ALIAS}-invoice-credential-info.json"

echo "Issuing invoice credential..."
echo "  Issuer: $ISSUER_ALIAS"
echo "  Holder: $HOLDER_ALIAS (AID: $HOLDER_AID)"
echo "  OOR Credential SAID: $OOR_CREDENTIAL_SAID"
echo "  Registry: $REGISTRY_NAME"
echo ""

docker compose exec -T tsx-shell tsx \
  sig-wallet/src/tasks/invoice/invoice-acdc-issue.ts \
  docker \
  "$ISSUER_ALIAS" \
  "$PASSCODE" \
  "$REGISTRY_NAME" \
  "$INVOICE_SCHEMA_SAID" \
  "$HOLDER_AID" \
  "$OOR_CREDENTIAL_SAID" \
  "$OOR_SCHEMA_SAID" \
  "$INVOICE_DATA" \
  "$OUTPUT_PATH"

echo ""
echo "✓ Invoice credential issued successfully"
echo "  Output: $OUTPUT_PATH"
