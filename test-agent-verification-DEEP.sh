#!/bin/bash
set -e

AGENT_NAME="${1:-jupiterSellerAgent}"
OOR_HOLDER_NAME="${2:-Jupiter_Chief_Sales_Officer}"
ENV="${3:-docker}"
JSON_OUTPUT="${4:-}"  # NEW: Optional --json flag

# Directory where agent info files are stored
TASK_DATA_DIR="./task-data"

# Try to read agent's BRAN from agent-brans.json (array structure)
AGENT_BRAN=""
if [ -f "${TASK_DATA_DIR}/agent-brans.json" ]; then
    # The structure is: { "agents": [ { "keriaAlias": "jupiterSellerAgent", "bran": "..." }, ... ] }
    AGENT_BRAN=$(jq -r ".agents[] | select(.keriaAlias == \"${AGENT_NAME}\") | .bran" "${TASK_DATA_DIR}/agent-brans.json" 2>/dev/null || echo "")
fi

# If BRAN not found in agent-brans.json, try the agent BRAN file directly
if [ -z "$AGENT_BRAN" ] || [ "$AGENT_BRAN" == "null" ]; then
    BRAN_FILE="${TASK_DATA_DIR}/${AGENT_NAME}-bran.txt"
    if [ -f "$BRAN_FILE" ]; then
        AGENT_BRAN=$(cat "$BRAN_FILE" 2>/dev/null || echo "")
    fi
fi

# If still not found, try reading from agent info file
if [ -z "$AGENT_BRAN" ] || [ "$AGENT_BRAN" == "null" ]; then
    AGENT_INFO_FILE="${TASK_DATA_DIR}/${AGENT_NAME}-info.json"
    if [ -f "$AGENT_INFO_FILE" ]; then
        AGENT_BRAN=$(jq -r '.bran // .passcode // empty' "$AGENT_INFO_FILE" 2>/dev/null || echo "")
    fi
fi

# Fallback to default if still not found
if [ -z "$AGENT_BRAN" ] || [ "$AGENT_BRAN" == "null" ]; then
    AGENT_BRAN="AgentPass123"
    if [ "$JSON_OUTPUT" != "--json" ]; then
        echo "⚠️  Warning: Could not find agent BRAN, using default passcode"
        echo "   Checked:"
        echo "   - ${TASK_DATA_DIR}/agent-brans.json (keriaAlias: ${AGENT_NAME})"
        echo "   - ${TASK_DATA_DIR}/${AGENT_NAME}-bran.txt"
        echo "   - ${TASK_DATA_DIR}/${AGENT_NAME}-info.json"
    fi
fi

# OOR Holder passcode - try to read from info file
OOR_PASSCODE="0ADckowyGuNwtJUPLeRqZvTp"
OOR_INFO_FILE="${TASK_DATA_DIR}/${OOR_HOLDER_NAME}-info.json"
if [ -f "$OOR_INFO_FILE" ]; then
    OOR_PASSCODE_FROM_FILE=$(jq -r '.bran // .passcode // empty' "$OOR_INFO_FILE" 2>/dev/null || echo "")
    if [ -n "$OOR_PASSCODE_FROM_FILE" ] && [ "$OOR_PASSCODE_FROM_FILE" != "null" ]; then
        OOR_PASSCODE="$OOR_PASSCODE_FROM_FILE"
    fi
fi

# Only show header if not JSON mode
if [ "$JSON_OUTPUT" != "--json" ]; then
    echo "=========================================="
    echo "DEEP AGENT DELEGATION VERIFICATION"
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "  Agent: ${AGENT_NAME}"
    echo "  OOR Holder: ${OOR_HOLDER_NAME}"
    echo "  ENV: ${ENV}"
    echo "  Agent Passcode: ${AGENT_BRAN:0:20}..."
    echo "  OOR Passcode: ${OOR_PASSCODE:0:20}..."
    echo ""
fi

docker compose exec -T tsx-shell tsx sig-wallet/src/tasks/agent/agent-verify-delegation-deep.ts \
  "${ENV}" \
  "${AGENT_BRAN}" \
  "${OOR_PASSCODE}" \
  "${AGENT_NAME}" \
  "${OOR_HOLDER_NAME}" \
  "${JSON_OUTPUT}"  # Pass JSON flag to TypeScript

if [ $? -eq 0 ]; then
    # Only show text if not JSON mode
    if [ "$JSON_OUTPUT" != "--json" ]; then
        echo ""
        echo "=========================================="
        echo "✅ DEEP VERIFICATION PASSED!"
        echo "=========================================="
    fi
else
    echo ""
    echo "=========================================="
    echo "❌ DEEP VERIFICATION FAILED"
    echo "=========================================="
    exit 1
fi
