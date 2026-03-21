#!/bin/bash
# subagent-delegate-approve.sh - Step 2: Parent agent approves delegation

set -e
PARENT_AGENT_ALIAS=$1
SUB_AGENT_ALIAS=$2

if [ -z "$PARENT_AGENT_ALIAS" ] || [ -z "$SUB_AGENT_ALIAS" ]; then
    echo "ERROR: Missing arguments"
    echo "Usage: $0 <parent_agent> <sub_agent>"
    exit 1
fi

echo "Parent agent $PARENT_AGENT_ALIAS approving delegation to $SUB_AGENT_ALIAS"
echo ""

docker compose exec -T tsx-shell tsx \
    sig-wallet/src/tasks/subagent/subagent-delegate-approve.ts \
    docker \
    "$PARENT_AGENT_ALIAS" \
    "$SUB_AGENT_ALIAS" \
    "/task-data"

exit $?