#!/bin/bash
################################################################################
# invoice-acdc-present.sh
# Present invoice credential to Sally verifier
################################################################################

set -e

HOLDER_ALIAS="${1:-tommyBuyerAgent}"
ISSUER_ALIAS="${2:-Jupiter_Chief_Sales_Officer}"

# Source environment variables
source ./task-scripts/workshop-env-vars.sh

# Get passcode for holder
PASSCODE=$(get_passcode "$HOLDER_ALIAS")

# Get credential SAID from issuer's invoice credential info
INVOICE_CRED_FILE="./task-data/${ISSUER_ALIAS}-invoice-credential-info.json"
if [ ! -f "$INVOICE_CRED_FILE" ]; then
    echo "ERROR: Invoice credential file not found: $INVOICE_CRED_FILE"
    exit 1
fi
CREDENTIAL_SAID=$(cat "$INVOICE_CRED_FILE" | jq -r '.said')

echo "Presenting invoice credential to Sally verifier..."
echo "  Holder: $HOLDER_ALIAS"
echo "  Credential SAID: $CREDENTIAL_SAID"
echo ""

docker compose exec -T tsx-shell tsx \
  sig-wallet/src/tasks/invoice/invoice-acdc-present.ts \
  docker \
  "$HOLDER_ALIAS" \
  "$PASSCODE" \
  "$CREDENTIAL_SAID"

echo ""
echo "✓ Invoice credential presented successfully"
