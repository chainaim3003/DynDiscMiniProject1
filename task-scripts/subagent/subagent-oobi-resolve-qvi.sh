#!/bin/bash
################################################################################
# subagent-oobi-resolve-qvi.sh
#
# Purpose: Sub-agent resolves QVI OOBI
#
# Usage: ./task-scripts/subagent/subagent-oobi-resolve-qvi.sh <sub_agent_alias>
################################################################################

set -e

SUB_AGENT_ALIAS=$1

if [ -z "$SUB_AGENT_ALIAS" ]; then
    echo "ERROR: Missing sub-agent alias"
    echo "Usage: $0 <sub_agent_alias>"
    exit 1
fi

echo "Sub-agent $SUB_AGENT_ALIAS resolving QVI OOBI..."

# Load QVI info
QVI_INFO_FILE="./task-data/qvi-info.json"
if [ ! -f "$QVI_INFO_FILE" ]; then
    echo "ERROR: QVI info file not found: $QVI_INFO_FILE"
    exit 1
fi

QVI_OOBI=$(jq -r '.oobi' "$QVI_INFO_FILE")

echo "Sub-agent $SUB_AGENT_ALIAS resolving QVI OOBI: $QVI_OOBI"

# Run OOBI resolution
docker compose exec -T tsx-shell tsx \
    sig-wallet/src/tasks/common/oobi-resolve.ts \
    docker \
    "$SUB_AGENT_ALIAS" \
    "$QVI_OOBI" \
    "/task-data"

echo "✓ QVI OOBI resolved by sub-agent $SUB_AGENT_ALIAS"

exit 0