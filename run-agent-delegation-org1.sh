#!/bin/bash
# run-agent-delegation-org1.sh
# Complete workflow for Organization 1 (Jupiter Knitting) agent delegation
# This assumes GEDA, QVI, LE, and OOR Holder are already set up

set -e

echo "=========================================="
echo "ORGANIZATION 1: JUPITER KNITTING"
echo "Agent Delegation Workflow"
echo "=========================================="

# Configuration
OOR_HOLDER_NAME="Jupiter_Chief_Sales_Officer"
AGENT_NAME="jupiterSellerAgent"
LE_NAME="Jupiter_Knitting"

echo ""
echo "Step 1: Agent initiates delegation request"
echo "------------------------------------------"
./task-scripts/person/person-delegate-agent-create.sh "${OOR_HOLDER_NAME}" "${AGENT_NAME}"

echo ""
echo "Step 2: OOR Holder approves delegation"
echo "------------------------------------------"
./task-scripts/person/person-approve-agent-delegation.sh "${OOR_HOLDER_NAME}" "${AGENT_NAME}"

echo ""
echo "Step 3: Agent completes delegation"
echo "------------------------------------------"
./task-scripts/agent/agent-aid-delegate-finish.sh "${AGENT_NAME}" "${OOR_HOLDER_NAME}"

echo ""
echo "Step 4: Agent resolves OOBIs"
echo "------------------------------------------"
echo "  4a. Resolving QVI OOBI..."
./task-scripts/agent/agent-oobi-resolve-qvi.sh "${AGENT_NAME}"

echo "  4b. Resolving LE OOBI..."
./task-scripts/agent/agent-oobi-resolve-le.sh "${AGENT_NAME}" "${LE_NAME}"

echo "  4c. Resolving Sally verifier OOBI..."
./task-scripts/agent/agent-oobi-resolve-verifier.sh "${AGENT_NAME}"

echo ""
echo "Step 5: Verify agent delegation"
echo "------------------------------------------"
./task-scripts/agent/agent-verify-delegation.sh "${AGENT_NAME}" "${OOR_HOLDER_NAME}"

echo ""
echo "=========================================="
echo "✓ ORGANIZATION 1 AGENT DELEGATION COMPLETE"
echo "=========================================="
echo ""
echo "Agent Summary:"
echo "  Name: ${AGENT_NAME}"
echo "  AID: $(cat ./task-data/${AGENT_NAME}-info.json | jq -r .aid)"
echo "  OOBI: $(cat ./task-data/${AGENT_NAME}-info.json | jq -r .oobi)"
echo "  Delegated from: ${OOR_HOLDER_NAME}"
echo "  Delegator AID: $(cat ./task-data/${OOR_HOLDER_NAME}-info.json | jq -r .aid)"
echo ""
