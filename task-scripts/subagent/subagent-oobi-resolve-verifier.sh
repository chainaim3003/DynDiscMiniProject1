#!/bin/bash
################################################################################
# subagent-oobi-resolve-verifier.sh
#
# Purpose: Sub-agent resolves Sally verifier OOBI
#
# Usage: ./task-scripts/subagent/subagent-oobi-resolve-verifier.sh <sub_agent_alias>
################################################################################

set -e

SUB_AGENT_ALIAS=$1

if [ -z "$SUB_AGENT_ALIAS" ]; then
    echo "ERROR: Missing sub-agent alias"
    echo "Usage: $0 <sub_agent_alias>"
    exit 1
fi

echo "Sub-agent $SUB_AGENT_ALIAS resolving Sally verifier OOBI..."

# Verifier OOBI (Sally)
VERIFIER_OOBI="http://verifier:9723/oobi"

echo "Sub-agent $SUB_AGENT_ALIAS resolving Verifier (Sally) OOBI: $VERIFIER_OOBI"

# Run OOBI resolution
docker compose exec -T tsx-shell tsx \
    sig-wallet/src/tasks/common/oobi-resolve.ts \
    docker \
    "$SUB_AGENT_ALIAS" \
    "$VERIFIER_OOBI" \
    "/task-data"

echo "✓ Verifier OOBI resolved by sub-agent $SUB_AGENT_ALIAS"

exit 0