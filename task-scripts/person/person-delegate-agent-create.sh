#!/bin/bash
# person-delegate-agent-create.sh - Person/OOR Holder initiates agent delegation
# Usage: person-delegate-agent-create.sh <oorHolderName> <agentName>

set -e

OOR_HOLDER_NAME=$1
AGENT_NAME=$2

if [ -z "$OOR_HOLDER_NAME" ] || [ -z "$AGENT_NAME" ]; then
  echo "Usage: person-delegate-agent-create.sh <oorHolderName> <agentName>"
  echo "Example: person-delegate-agent-create.sh Jupiter_Chief_Sales_Officer jupiterSellerAgent"
  exit 1
fi

echo "Creating agent delegation request: ${AGENT_NAME} from ${OOR_HOLDER_NAME}"

source ./task-scripts/workshop-env-vars.sh

docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh person/person-delegate-agent-create.ts \
    'docker' \
    "${AGENT_SALT:-AgentPass123}" \
    "/task-data" \
    "${OOR_HOLDER_NAME}" \
    "${AGENT_NAME}"

echo "✓ Agent delegation request created"
echo "   File: ./task-data/${AGENT_NAME}-delegate-info.json"
