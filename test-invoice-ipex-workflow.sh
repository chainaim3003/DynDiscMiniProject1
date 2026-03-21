#!/bin/bash
# ============================================
# test-invoice-ipex-workflow.sh
# Test/Verify the Invoice IPEX workflow results
# ============================================
#
# This script verifies that the 4C workflow completed successfully
# by running the DEEP-EXT-credential.sh verifier.
#
# Prerequisites:
# - 4C workflow completed (run-all-buyerseller-4C-with-agents.sh)
#
# Usage:
#   ./test-invoice-ipex-workflow.sh
#
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
WHITE='\033[1;37m'
NC='\033[0m'

echo ""
echo -e "${WHITE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${WHITE}║          INVOICE CREDENTIAL VERIFICATION TEST                                ║${NC}"
echo -e "${WHITE}║          (Run after 4C workflow completes)                                   ║${NC}"
echo -e "${WHITE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Run the credential verifier
if [ -f "${SCRIPT_DIR}/DEEP-EXT-credential.sh" ]; then
    echo -e "${BLUE}Running DEEP-EXT-credential.sh verifier...${NC}"
    echo ""
    bash "${SCRIPT_DIR}/DEEP-EXT-credential.sh" tommyBuyerAgent jupiterSellerAgent Jupiter_Chief_Sales_Officer
else
    echo -e "${YELLOW}DEEP-EXT-credential.sh not found${NC}"
    exit 1
fi
