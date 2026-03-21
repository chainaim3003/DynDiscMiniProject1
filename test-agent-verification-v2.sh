#!/bin/bash
# test-agent-verification-v2.sh
# Enhanced test script that shows V2 detailed verification output

set -e

echo "=========================================="
echo "ENHANCED AGENT VERIFICATION TEST (V2)"
echo "=========================================="
echo ""

# Configuration
AGENT_NAME="${1:-jupiterSellerAgent}"
OOR_HOLDER_NAME="${2:-Jupiter_Chief_Sales_Officer}"
VERIFIER_URL="http://localhost:9724"

echo "Test Configuration:"
echo "  Agent Name: ${AGENT_NAME}"
echo "  OOR Holder Name: ${OOR_HOLDER_NAME}"
echo "  Verifier URL: ${VERIFIER_URL}"
echo ""

# Check if info files exist
AGENT_INFO="./task-data/${AGENT_NAME}-info.json"
OOR_INFO="./task-data/${OOR_HOLDER_NAME}-info.json"

if [ ! -f "$AGENT_INFO" ]; then
    echo "❌ Error: Agent info file not found: $AGENT_INFO"
    exit 1
fi

if [ ! -f "$OOR_INFO" ]; then
    echo "❌ Error: OOR Holder info file not found: $OOR_INFO"
    exit 1
fi

# Extract AIDs
AGENT_AID=$(cat "$AGENT_INFO" | jq -r '.aid')
OOR_AID=$(cat "$OOR_INFO" | jq -r '.aid')./

echo "Extracted AIDs:"
echo "  Agent AID: ${AGENT_AID:0:20}..."
echo "  OOR Holder AID: ${OOR_AID:0:20}..."
echo ""

# ============================================
# TEST 1: Check Verifier Version
# ============================================
echo "=========================================="
echo "TEST 1: Verifier Version Check"
echo "=========================================="
echo ""

VERSION_RESPONSE=$(curl -s ${VERIFIER_URL}/)
VERSION=$(echo "$VERSION_RESPONSE" | jq -r '.version // "unknown"')
COVERAGE=$(echo "$VERSION_RESPONSE" | jq -r '.verification_coverage // "unknown"')

echo "Verifier Information:"
echo "$VERSION_RESPONSE" | jq -r '.service'
echo "  Version: $VERSION"
echo "  Coverage: $COVERAGE"
echo ""

if [[ "$VERSION" == "2.0"* ]]; then
    echo "✅ V2 Detected - Enhanced verification available!"
    V2_ENABLED=true
else
    echo "ℹ️  V1 Detected - Basic verification"
    V2_ENABLED=false
fi

echo ""

# ============================================
# TEST 2: Direct Verification API Call
# ============================================
echo "=========================================="
echo "TEST 2: Direct Verification API Call"
echo "=========================================="
echo ""

VERIFY_PAYLOAD=$(cat <<EOF
{
  "aid": "${OOR_AID}",
  "agent_aid": "${AGENT_AID}",
  "verify_kel": true
}
EOF
)

echo "Sending verification request..."
echo ""

VERIFY_RESPONSE=$(curl -s -X POST ${VERIFIER_URL}/verify/agent-delegation \
  -H "Content-Type: application/json" \
  -d "$VERIFY_PAYLOAD")

VALID=$(echo "$VERIFY_RESPONSE" | jq -r '.valid')
VERIFIED=$(echo "$VERIFY_RESPONSE" | jq -r '.verified')

if [ "$VALID" = "true" ] && [ "$VERIFIED" = "true" ]; then
    echo "✅ TEST 2 PASSED: Verification succeeded"
else
    echo "❌ TEST 2 FAILED: Verification failed"
    echo "Response:"
    echo "$VERIFY_RESPONSE" | jq '.'
    exit 1
fi

echo ""

# ============================================
# TEST 3: Show Detailed Results (V2 only)
# ============================================
if [ "$V2_ENABLED" = true ]; then
    echo "=========================================="
    echo "TEST 3: V2 Enhanced Verification Details"
    echo "=========================================="
    echo ""
    
    echo "V2 Features Detected:"
    
    # Agent ICP Analysis
    if echo "$VERIFY_RESPONSE" | jq -e '.verification.agent_icp_analysis' > /dev/null 2>&1; then
        echo "  ✅ Agent ICP Analysis"
        ICP_HAS_DI=$(echo "$VERIFY_RESPONSE" | jq -r '.verification.agent_icp_analysis.has_delegator_field')
        ICP_MATCH=$(echo "$VERIFY_RESPONSE" | jq -r '.verification.agent_icp_analysis.delegator_matches')
        echo "     - Has 'di' field: $ICP_HAS_DI"
        echo "     - Delegator matches: $ICP_MATCH"
    fi
    
    # Delegation Seal Analysis
    if echo "$VERIFY_RESPONSE" | jq -e '.verification.delegation_seal_analysis' > /dev/null 2>&1; then
        echo "  ✅ Delegation Seal Analysis"
        SEAL_FOUND=$(echo "$VERIFY_RESPONSE" | jq -r '.verification.delegation_seal_analysis.seal_found_in_controller_kel')
        SEAL_SEQ=$(echo "$VERIFY_RESPONSE" | jq -r '.verification.delegation_seal_analysis.seal_sequence')
        echo "     - Seal found: $SEAL_FOUND"
        echo "     - Seal sequence: $SEAL_SEQ"
    fi
    
    # Consistency Checks
    if echo "$VERIFY_RESPONSE" | jq -e '.verification.consistency_checks' > /dev/null 2>&1; then
        echo "  ✅ Consistency Checks"
        ALL_PASSED=$(echo "$VERIFY_RESPONSE" | jq -r '.verification.consistency_checks.all_passed')
        CHECK_COUNT=$(echo "$VERIFY_RESPONSE" | jq '.verification.consistency_checks.checks | length')
        echo "     - All passed: $ALL_PASSED"
        echo "     - Checks performed: $CHECK_COUNT"
    fi
    
    echo ""
    echo "Full V2 Verification Response:"
    echo "$VERIFY_RESPONSE" | jq '.verification'
    echo ""
fi

# ============================================
# SUMMARY
# ============================================
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""
echo "Agent: ${AGENT_NAME}"
echo "  AID: ${AGENT_AID:0:20}..."
echo ""
echo "OOR Holder: ${OOR_HOLDER_NAME}"
echo "  AID: ${OOR_AID:0:20}..."
echo ""
echo "Verifier:"
echo "  Version: $VERSION"
echo "  Coverage: $COVERAGE"
echo ""
echo "=========================================="
echo "🎉 VERIFICATION TESTS COMPLETE!"
echo "=========================================="
echo ""

if [ "$V2_ENABLED" = true ]; then
    echo "✨ You're running V2 with enhanced KEL-based verification!"
else
    echo "ℹ️  You're running V1 - consider upgrading to V2"
fi
echo ""
