#!/bin/bash
################################################################################
# verify-invoice-implementation.sh
# Quick script to verify all invoice credential files are in place
################################################################################

echo "=========================================="
echo "Invoice Implementation Verification"
echo "=========================================="
echo ""

MISSING_FILES=0
PRESENT_FILES=0

check_file() {
    local file=$1
    local desc=$2
    
    if [ -f "$file" ]; then
        echo "✅ $desc"
        PRESENT_FILES=$((PRESENT_FILES + 1))
    else
        echo "❌ MISSING: $desc"
        echo "   File: $file"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
}

echo "Checking Schema & Configuration..."
check_file "schemas/invoice-credential-schema.json" "Invoice schema"
check_file "appconfig/invoiceConfig.json" "Invoice configuration"
echo ""

echo "Checking TypeScript Implementation..."
check_file "sig-wallet/src/tasks/invoice/invoice-registry-create.ts" "Registry creation"
check_file "sig-wallet/src/tasks/invoice/invoice-acdc-issue.ts" "Invoice issuance"
check_file "sig-wallet/src/tasks/invoice/invoice-acdc-admit.ts" "Invoice admission"
check_file "sig-wallet/src/tasks/invoice/invoice-acdc-present.ts" "Invoice presentation"
check_file "sig-wallet/src/tasks/invoice/invoice-verify-chain.ts" "Chain verification"
echo ""

echo "Checking Shell Scripts..."
check_file "task-scripts/invoice/invoice-registry-create.sh" "Registry shell script"
check_file "task-scripts/invoice/invoice-acdc-issue.sh" "Issuance shell script"
check_file "task-scripts/invoice/invoice-acdc-admit.sh" "Admission shell script"
check_file "task-scripts/invoice/invoice-acdc-present.sh" "Presentation shell script"
echo ""

echo "Checking Updated Scripts..."
check_file "run-all-buyerseller-3-with-agents.sh" "Main orchestration"
check_file "test-agent-verification-DEEP-credential.sh" "Verification test"
echo ""

echo "Checking Documentation..."
check_file "INVOICE-CREDENTIAL-DESIGN.md" "Design document"
check_file "INVOICE-IMPLEMENTATION-GUIDE.md" "Implementation guide"
check_file "INVOICE-IMPLEMENTATION-SUMMARY.md" "Implementation summary"
echo ""

echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo "Present: $PRESENT_FILES files"
echo "Missing: $MISSING_FILES files"
echo ""

if [ $MISSING_FILES -eq 0 ]; then
    echo "✅ All invoice credential files are in place!"
    echo ""
    echo "Next Steps:"
    echo "  1. Make scripts executable: chmod +x task-scripts/invoice/*.sh"
    echo "  2. Run workflow: ./run-all-buyerseller-3-with-agents.sh"
    echo "  3. Verify invoice: ./test-agent-verification-DEEP-credential.sh jupiterSellerAgent Jupiter_Chief_Sales_Officer true docker"
    echo ""
else
    echo "⚠️  Some files are missing. Please check the implementation."
    exit 1
fi
