#!/bin/bash
# agent-verify-delegation.sh - Verify agent delegation via Sally
# Usage: agent-verify-delegation.sh <agentName> <oorHolderName>

set -e

AGENT_NAME=$1
OOR_HOLDER_NAME=$2

if [ -z "$AGENT_NAME" ] || [ -z "$OOR_HOLDER_NAME" ]; then
  echo "Usage: agent-verify-delegation.sh <agentName> <oorHolderName>"
  echo "Example: agent-verify-delegation.sh jupiterSellerAgent Jupiter_Chief_Sales_Officer"
  exit 1
fi

echo "Verifying agent delegation via Sally"
echo "  Agent: ${AGENT_NAME}"
echo "  OOR Holder: ${OOR_HOLDER_NAME}"

docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh agent/agent-verify-delegation.ts \
    "/task-data" \
    "${AGENT_NAME}" \
    "${OOR_HOLDER_NAME}"
