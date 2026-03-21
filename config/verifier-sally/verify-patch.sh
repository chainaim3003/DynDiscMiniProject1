#!/bin/bash
# verify-patch.sh - Verify the custom Sally patch was applied

echo "=== Verifying Sally Custom Patch ==="
echo ""

docker run --rm --entrypoint /bin/sh gleif/sally-custom:latest -c '
echo "1. Checking if AgentDelegationResource class exists:"
grep -n "class AgentDelegationResource" /sally/src/sally/core/serving.py && echo "✓ Found" || echo "✗ Not found"
echo ""

echo "2. Checking if custom endpoint is registered:"
grep -n "verify/agent-delegation" /sally/src/sally/core/serving.py && echo "✓ Found" || echo "✗ Not found"
echo ""

echo "3. Checking if CUSTOM AGENT DELEGATION VERIFICATION marker exists:"
grep -n "CUSTOM AGENT DELEGATION VERIFICATION" /sally/src/sally/core/serving.py && echo "✓ Found" || echo "✗ Not found"
echo ""

echo "4. Showing context around the custom code (first 20 lines):"
grep -A 20 "CUSTOM AGENT DELEGATION VERIFICATION" /sally/src/sally/core/serving.py | head -25
'

echo ""
echo "=== Verification complete ==="
