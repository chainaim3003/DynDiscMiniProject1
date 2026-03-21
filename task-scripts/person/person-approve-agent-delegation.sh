#!/bin/bash
# person-approve-agent-delegation.sh - Person/OOR Holder approves agent delegation
# Usage: person-approve-agent-delegation.sh <oorHolderName> <agentName>

set -e

OOR_HOLDER_NAME=$1
AGENT_NAME=$2

if [ -z "$OOR_HOLDER_NAME" ] || [ -z "$AGENT_NAME" ]; then
  echo "Usage: person-approve-agent-delegation.sh <oorHolderName> <agentName>"
  echo "Example: person-approve-agent-delegation.sh Jupiter_Chief_Sales_Officer jupiterSellerAgent"
  exit 1
fi

echo "Approving agent delegation: ${AGENT_NAME} by ${OOR_HOLDER_NAME}"

source ./task-scripts/workshop-env-vars.sh

docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh person/person-approve-agent-delegation.ts \
    'docker' \
    "${PERSON_SALT}" \
    "${OOR_HOLDER_NAME}" \
    "/task-data/${AGENT_NAME}-delegate-info.json"

echo "✓ Agent delegation approved by ${OOR_HOLDER_NAME}"
