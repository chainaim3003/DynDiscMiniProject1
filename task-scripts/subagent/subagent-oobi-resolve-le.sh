#!/bin/bash
################################################################################
# subagent-oobi-resolve-le.sh
#
# Purpose: Sub-agent resolves LE OOBI
#
# Usage: ./task-scripts/subagent/subagent-oobi-resolve-le.sh <sub_agent_alias> <le_alias>
################################################################################

set -e

SUB_AGENT_ALIAS=$1
LE_ALIAS=$2

if [ -z "$SUB_AGENT_ALIAS" ] || [ -z "$LE_ALIAS" ]; then
    echo "ERROR: Missing arguments"
    echo "Usage: $0 <sub_agent_alias> <le_alias>"
    exit 1
fi

echo "Sub-agent $SUB_AGENT_ALIAS resolving LE OOBI for $LE_ALIAS..."

# Load LE info
LE_INFO_FILE="./task-data/le-info.json"
if [ ! -f "$LE_INFO_FILE" ]; then
    echo "ERROR: LE info file not found: $LE_INFO_FILE"
    exit 1
fi

LE_OOBI=$(jq -r '.oobi' "$LE_INFO_FILE")

echo "Sub-agent $SUB_AGENT_ALIAS resolving LE OOBI: $LE_OOBI"

# Run OOBI resolution
docker compose exec -T tsx-shell tsx \
    sig-wallet/src/tasks/common/oobi-resolve.ts \
    docker \
    "$SUB_AGENT_ALIAS" \
    "$LE_OOBI" \
    "/task-data"

echo "✓ LE OOBI resolved by sub-agent $SUB_AGENT_ALIAS"

exit 0