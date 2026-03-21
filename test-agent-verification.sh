#!/bin/bash
# test-agent-verification.sh
# Test script for agent delegation verification using standalone verification service

set -e

echo "=========================================="
echo "AGENT DELEGATION VERIFICATION TEST"
echo "=========================================="
echo ""

# Configuration
AGENT_NAME="${1:-jupiterSellerAgent}"
OOR_HOLDER_NAME="${2:-Jupiter_Chief_Sales_Officer}"

echo "Test Configuration:"
echo "  Agent Name: ${AGENT_NAME}"
echo "  OOR Holder Name: ${OOR_HOLDER_NAME}"
echo ""

# Check if info files exist
AGENT_INFO="./task-data/${AGENT_NAME}-info.json"
OOR_INFO="./task-data/${OOR_HOLDER_NAME}-info.json"

if [ ! -f "$AGENT_INFO" ]; then
    echo "❌ Error: Agent info file not found: $AGENT_INFO"
    echo "   Run the delegation workflow first: ./run-all-buyerseller-2-with-agents.sh"
    exit 1
fi

if [ ! -f "$OOR_INFO" ]; then
    echo "❌ Error: OOR Holder info file not found: $OOR_INFO"
    echo "   Ensure OOR holder exists and has been created"
    exit 1
fi

# Extract AIDs
AGENT_AID=$(cat "$AGENT_INFO" | jq -r '.aid')
OOR_AID=$(cat "$OOR_INFO" | jq -r '.aid')

echo "Extracted AIDs:"
echo "  Agent AID: ${AGENT_AID}"
echo "  OOR Holder AID: ${OOR_AID}"
echo ""

# ============================================
# TEST 1: TypeScript Task Method (Standalone Verification Service)
# ============================================
echo "=========================================="
echo "TEST 1: Verification via TypeScript Task"
echo "=========================================="
echo ""
echo "This calls agent-verify-delegation.ts which uses the"
echo "standalone verification service at vlei-verification:9723"
echo ""

./task-scripts/agent/agent-verify-delegation.sh "${AGENT_NAME}" "${OOR_HOLDER_NAME}"

RESULT=$?
if [ $RESULT -eq 0 ]; then
    echo ""
    echo "✅ TEST 1 PASSED: TypeScript verification succeeded"
else
    echo ""
    echo "❌ TEST 1 FAILED: TypeScript verification failed"
    exit 1
fi

echo ""
echo ""

# ============================================
# NOTE: Sally Direct Endpoint Tests Removed
# ============================================
# The standard Sally verifier (gleif/sally:1.0.2) does not include
# the /verify/agent-delegation endpoint. We use a standalone
# verification service (vlei-verification) instead.
#
# TEST 2, TEST 3, and TEST 4 have been removed as they tested
# Sally's non-existent endpoint and would always fail.
# ============================================

# ============================================
# SUMMARY
# ============================================
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""
echo "Agent: ${AGENT_NAME}"
echo "  AID: ${AGENT_AID}"
echo ""
echo "OOR Holder: ${OOR_HOLDER_NAME}"  
echo "  AID: ${OOR_AID}"
echo ""
echo "Results:"
echo "  ✅ Standalone Verification Service: PASSED"
echo ""
echo "Verification Service:"
echo "  Container: vlei-verification"
echo "  Internal URL: http://vlei-verification:9723"
echo "  External URL: http://localhost:9724"
echo ""
echo "=========================================="
echo "🎉 AGENT VERIFICATION TEST PASSED!"
echo "=========================================="
echo ""
echo "The agent delegation verification system is working correctly."
echo "Agent ${AGENT_NAME} is properly delegated from ${OOR_HOLDER_NAME}."
echo ""
