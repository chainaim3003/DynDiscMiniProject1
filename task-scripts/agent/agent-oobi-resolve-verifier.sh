#!/bin/bash
# agent-oobi-resolve-verifier.sh - Agent resolves Sally verifier OOBI
# Usage: agent-oobi-resolve-verifier.sh <agentName>

set -e

AGENT_NAME=$1

if [ -z "$AGENT_NAME" ]; then
  echo "Usage: agent-oobi-resolve-verifier.sh <agentName>"
  echo "Example: agent-oobi-resolve-verifier.sh jupiterSellerAgent"
  exit 1
fi

echo "Agent ${AGENT_NAME} resolving Sally verifier OOBI"

source ./task-scripts/workshop-env-vars.sh

docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh agent/agent-oobi-resolve-verifier.ts \
    'docker' \
    "${AGENT_SALT:-AgentPass123}" \
    "${AGENT_NAME}"

echo "✓ Verifier OOBI resolved by agent ${AGENT_NAME}"
