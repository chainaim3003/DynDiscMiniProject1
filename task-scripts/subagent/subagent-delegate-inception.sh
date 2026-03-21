#!/bin/bash
# subagent-delegate-inception.sh - Step 1: Create sub-agent AID

set -e
SUB_AGENT_ALIAS=$1
PARENT_AGENT_ALIAS=$2

if [ -z "$SUB_AGENT_ALIAS" ] || [ -z "$PARENT_AGENT_ALIAS" ]; then
    echo "ERROR: Missing arguments"
    echo "Usage: $0 <sub_agent> <parent_agent>"
    exit 1
fi

echo "Creating sub-agent AID: $SUB_AGENT_ALIAS"
echo "Parent agent: $PARENT_AGENT_ALIAS"
echo ""

docker compose exec -T tsx-shell tsx \
    sig-wallet/src/tasks/subagent/subagent-delegate-inception.ts \
    docker \
    "$SUB_AGENT_ALIAS" \
    "$PARENT_AGENT_ALIAS" \
    "/task-data"

exit $?