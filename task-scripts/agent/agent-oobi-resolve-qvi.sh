#!/bin/bash
# agent-oobi-resolve-qvi.sh - Agent resolves QVI OOBI
# Usage: agent-oobi-resolve-qvi.sh <agentName>

set -e

AGENT_NAME=$1

if [ -z "$AGENT_NAME" ]; then
  echo "Usage: agent-oobi-resolve-qvi.sh <agentName>"
  echo "Example: agent-oobi-resolve-qvi.sh jupiterSellerAgent"
  exit 1
fi

echo "Agent ${AGENT_NAME} resolving QVI OOBI"

source ./task-scripts/workshop-env-vars.sh

QVI_OOBI=$(cat ./task-data/qvi-info.json | jq -r .oobi)

docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh agent/agent-oobi-resolve-qvi.ts \
    'docker' \
    "${AGENT_SALT:-AgentPass123}" \
    "${AGENT_NAME}" \
    "${QVI_OOBI}"

echo "✓ QVI OOBI resolved by agent ${AGENT_NAME}"
