#!/bin/bash
# agent-oobi-resolve-le.sh - Agent resolves LE OOBI
# Usage: agent-oobi-resolve-le.sh <agentName> <leName>

set -e

AGENT_NAME=$1
LE_NAME=$2

if [ -z "$AGENT_NAME" ] || [ -z "$LE_NAME" ]; then
  echo "Usage: agent-oobi-resolve-le.sh <agentName> <leName>"
  echo "Example: agent-oobi-resolve-le.sh jupiterSellerAgent Jupiter_Knitting"
  exit 1
fi

echo "Agent ${AGENT_NAME} resolving LE OOBI for ${LE_NAME}"

source ./task-scripts/workshop-env-vars.sh

LE_OOBI=$(cat ./task-data/${LE_NAME}-info.json | jq -r .oobi)

docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh agent/agent-oobi-resolve-le.ts \
    'docker' \
    "${AGENT_SALT:-AgentPass123}" \
    "${AGENT_NAME}" \
    "${LE_OOBI}"

echo "✓ LE OOBI resolved by agent ${AGENT_NAME}"
