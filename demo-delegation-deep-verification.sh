#!/bin/bash
#
# demo-delegation-deep-verification.sh - Deep Agent Verification
# Run this AFTER demo-delegation-issuance.sh completes
#

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}=========================================="
    echo -e "$1"
    echo -e "==========================================${NC}"
    echo ""
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

run_verification() {
    print_header "AGENT DELEGATION DEMONSTRATION - PART 2: VERIFICATION"
    echo "This script performs deep delegation verification"
    echo ""
    read -p "Press ENTER to continue or Ctrl+C to cancel..."

    print_header "DEEP DELEGATION VERIFICATION"

    echo "Verifying jupiterSellerAgent..."
    echo "-------------------------------------------------------------------"
    if ./test-agent-verification-DEEP.sh 2>&1 | tee /tmp/verification-jupiterSellerAgent.log; then
        echo "-------------------------------------------------------------------"
        print_success "jupiterSellerAgent verified"
    else
        print_error "jupiterSellerAgent verification failed"
        return 1
    fi

    echo ""
    echo "Verifying tommyBuyerAgent..."
    echo "-------------------------------------------------------------------"
    if ./test-agent-verification-DEEP.sh tommyBuyerAgent Tommy_Chief_Procurement_Officer 2>&1 | tee /tmp/verification-tommyBuyerAgent.log; then
        echo "-------------------------------------------------------------------"
        print_success "tommyBuyerAgent verified"
    else
        print_error "tommyBuyerAgent verification failed"
        return 1
    fi

    # SUCCESS
    print_header "🎉 VERIFICATION COMPLETE!"
    echo -e "${GREEN}"
    echo "Both agents verified successfully:"
    echo "  ✅ jupiterSellerAgent"
    echo "  ✅ tommyBuyerAgent"
    echo ""
    echo "Verification logs:"
    echo "  • /tmp/verification-jupiterSellerAgent.log"
    echo "  • /tmp/verification-tommyBuyerAgent.log"
    echo -e "${NC}"

    print_header "DEMONSTRATION SUCCESSFUL"
    echo "The complete agent delegation system is operational:"
    echo ""
    echo "  • Agents created and delegated from OOR holders"
    echo "  • KEL seals anchored in delegator KELs"
    echo "  • OOBIs resolved across trust network"
    echo "  • Sally verifier validated complete chain"
    echo ""
    echo "View trust tree:"
    echo "  cat task-data/trust-tree-buyerseller.txt"
    echo ""

    return 0
}

# Run verification
run_verification
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ Complete demonstration successful!${NC}"
else
    echo -e "${RED}❌ Verification encountered errors.${NC}"
fi

return $RESULT 2>/dev/null || true
