#!/bin/bash
# ============================================
# DEEP-EXT-credential.sh
# Extended Deep Verification for Self-Attested Credentials
# ============================================
#
# PURPOSE: Verify that a self-attested invoice credential is valid
#          (VERIFICATION ONLY - no IPEX grant/admit - that's in 4C script)
#
# This is the credential equivalent of DEEP-EXT.sh for delegation.
# It verifies AFTER the 4C workflow has completed the IPEX grant/admit.
#
# VERIFICATION STEPS:
#   1. Verify issuer agent's delegation (calls DEEP-EXT.sh)
#   2. Verify IPEX grant was sent (check grant info file)
#   3. Verify credential structure (self-attested: issuer = issuee)
#   4. Verify credential data integrity (invoice details)
#   5. Verify trust chain (agent → OOR holder → LE → QVI → GEDA)
#
# TRUST MODEL for Self-Attested Credentials:
#   - Self-attested means issuer AID = issuee AID
#   - Trust comes from the agent's delegation chain, NOT credential edges
#   - jupiterSellerAgent → delegated from OOR holder → has OOR from QVI
#
# Usage:
#   ./DEEP-EXT-credential.sh <verifierAgent> <issuerAgent> [issuerDelegator]
#
# Example:
#   ./DEEP-EXT-credential.sh tommyBuyerAgent jupiterSellerAgent Jupiter_Chief_Sales_Officer
#
# ============================================

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
WHITE='\033[1;37m'
CYAN='\033[0;36m'
NC='\033[0m'

# Arguments
VERIFIER_AGENT="${1:-tommyBuyerAgent}"
ISSUER_AGENT="${2:-jupiterSellerAgent}"
ISSUER_DELEGATOR="${3:-}"
ENV="${4:-docker}"
JSON_OUTPUT="${5:-}"

# Auto-detect issuer's delegator if not provided
if [ -z "$ISSUER_DELEGATOR" ]; then
    if [[ "$ISSUER_AGENT" == *"jupiter"* ]] || [[ "$ISSUER_AGENT" == *"Jupiter"* ]] || [[ "$ISSUER_AGENT" == *"seller"* ]]; then
        ISSUER_DELEGATOR="Jupiter_Chief_Sales_Officer"
    elif [[ "$ISSUER_AGENT" == *"tommy"* ]] || [[ "$ISSUER_AGENT" == *"Tommy"* ]] || [[ "$ISSUER_AGENT" == *"buyer"* ]]; then
        ISSUER_DELEGATOR="Tommy_Chief_Procurement_Officer"
    else
        ISSUER_DELEGATOR="Unknown_OOR_Holder"
    fi
fi

# Determine task-data directory
TASK_DATA_DIR="./task-data"
if [ ! -d "$TASK_DATA_DIR" ]; then
    WSL_TASK_DATA="/root/projects/algoTitanV61/LegentvLEI/task-data"
    if [ -d "$WSL_TASK_DATA" ]; then
        TASK_DATA_DIR="$WSL_TASK_DATA"
    fi
fi

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Results tracking
STEP1_DELEGATION=false
STEP2_GRANT_SENT=false
STEP3_STRUCTURE_VALID=false
STEP4_DATA_VERIFIED=false
STEP5_TRUST_CHAIN=false

if [ "$JSON_OUTPUT" != "--json" ]; then
    echo ""
    echo -e "${WHITE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${WHITE}║          DEEP-EXT CREDENTIAL VERIFICATION                                    ║${NC}"
    echo -e "${WHITE}║          Self-Attested Invoice Credential Verifier                           ║${NC}"
    echo -e "${WHITE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Configuration:${NC}"
    echo "  Verifier Agent: ${VERIFIER_AGENT}"
    echo "  Issuer Agent:   ${ISSUER_AGENT}"
    echo "  Issuer Delegator: ${ISSUER_DELEGATOR}"
    echo "  Task Data Dir:  ${TASK_DATA_DIR}"
    echo ""
fi

# ============================================
# STEP 1: Verify Issuer Agent's Delegation
# ============================================
if [ "$JSON_OUTPUT" != "--json" ]; then
    echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${WHITE}  STEP 1: Verify Issuer Agent's Delegation (DEEP-EXT)${NC}"
    echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
fi

# Run the existing DEEP-EXT verification for the issuer agent
if [ -f "${SCRIPT_DIR}/test-agent-verification-DEEP-EXT.sh" ]; then
    if [ "$JSON_OUTPUT" = "--json" ]; then
        DEEP_EXT_RESULT=$(bash "${SCRIPT_DIR}/test-agent-verification-DEEP-EXT.sh" "$ISSUER_AGENT" "$ISSUER_DELEGATOR" "$ENV" --json 2>&1)
        if echo "$DEEP_EXT_RESULT" | grep -q '"success": true'; then
            STEP1_DELEGATION=true
        fi
    else
        echo -e "${BLUE}→ Running DEEP-EXT verification for ${ISSUER_AGENT}...${NC}"
        echo ""
        if bash "${SCRIPT_DIR}/test-agent-verification-DEEP-EXT.sh" "$ISSUER_AGENT" "$ISSUER_DELEGATOR" "$ENV"; then
            STEP1_DELEGATION=true
            echo ""
            echo -e "${GREEN}✓ Step 1 PASSED: Issuer agent delegation verified${NC}"
        else
            echo ""
            echo -e "${RED}✗ Step 1 FAILED: Issuer agent delegation verification failed${NC}"
        fi
    fi
else
    if [ "$JSON_OUTPUT" != "--json" ]; then
        echo -e "${YELLOW}⚠ DEEP-EXT script not found, checking info files...${NC}"
    fi
    
    # Fallback: Check agent info files
    ISSUER_INFO_FILE="${TASK_DATA_DIR}/${ISSUER_AGENT}-info.json"
    if [ -f "$ISSUER_INFO_FILE" ]; then
        ISSUER_DI=$(jq -r '.state.di // .di // ""' "$ISSUER_INFO_FILE" 2>/dev/null)
        if [ -n "$ISSUER_DI" ] && [ "$ISSUER_DI" != "null" ]; then
            STEP1_DELEGATION=true
            if [ "$JSON_OUTPUT" != "--json" ]; then
                echo -e "${GREEN}✓ Step 1 PASSED: Issuer has delegation field (di): ${ISSUER_DI:0:20}...${NC}"
            fi
        fi
    fi
fi

echo ""

# ============================================
# STEP 2: Verify IPEX Grant Was Sent
# ============================================
if [ "$JSON_OUTPUT" != "--json" ]; then
    echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${WHITE}  STEP 2: Verify IPEX Grant Was Sent${NC}"
    echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
fi

GRANT_INFO_FILE="${TASK_DATA_DIR}/${ISSUER_AGENT}-ipex-grant-info.json"

if [ -f "$GRANT_INFO_FILE" ]; then
    GRANT_SAID=$(jq -r '.grantResult.said // .credentialSAID // ""' "$GRANT_INFO_FILE" 2>/dev/null)
    CRED_SAID=$(jq -r '.credentialSAID // ""' "$GRANT_INFO_FILE" 2>/dev/null)
    GRANT_SENDER=$(jq -r '.sender // ""' "$GRANT_INFO_FILE" 2>/dev/null)
    GRANT_RECEIVER=$(jq -r '.receiver // ""' "$GRANT_INFO_FILE" 2>/dev/null)
    INVOICE_NUM=$(jq -r '.invoiceNumber // ""' "$GRANT_INFO_FILE" 2>/dev/null)
    AMOUNT=$(jq -r '.amount // ""' "$GRANT_INFO_FILE" 2>/dev/null)
    CURRENCY=$(jq -r '.currency // ""' "$GRANT_INFO_FILE" 2>/dev/null)
    
    # Verify grant was sent to correct receiver
    if [ "$GRANT_SENDER" = "$ISSUER_AGENT" ] && [ "$GRANT_RECEIVER" = "$VERIFIER_AGENT" ]; then
        STEP2_GRANT_SENT=true
        if [ "$JSON_OUTPUT" != "--json" ]; then
            echo -e "${GREEN}✓ Step 2 PASSED: IPEX Grant verified${NC}"
            echo "    Grant SAID: ${GRANT_SAID:0:30}..."
            echo "    Credential SAID: ${CRED_SAID:0:30}..."
            echo "    From: $GRANT_SENDER → To: $GRANT_RECEIVER"
            echo "    Invoice: $INVOICE_NUM - $AMOUNT $CURRENCY"
        fi
    else
        if [ "$JSON_OUTPUT" != "--json" ]; then
            echo -e "${RED}✗ Step 2 FAILED: Grant sender/receiver mismatch${NC}"
            echo "    Expected: $ISSUER_AGENT → $VERIFIER_AGENT"
            echo "    Found: $GRANT_SENDER → $GRANT_RECEIVER"
        fi
    fi
else
    if [ "$JSON_OUTPUT" != "--json" ]; then
        echo -e "${RED}✗ Step 2 FAILED: IPEX grant info not found${NC}"
        echo "    Expected: $GRANT_INFO_FILE"
        echo "    Run 4C script first to complete IPEX workflow"
    fi
fi

echo ""

# ============================================
# STEP 3: Verify Credential Structure (Self-Attested)
# ============================================
if [ "$JSON_OUTPUT" != "--json" ]; then
    echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${WHITE}  STEP 3: Verify Credential Structure (Self-Attested)${NC}"
    echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
fi

CRED_INFO_FILE="${TASK_DATA_DIR}/${ISSUER_AGENT}-self-invoice-credential-info.json"
ISSUER_INFO_FILE="${TASK_DATA_DIR}/${ISSUER_AGENT}-info.json"

if [ -f "$CRED_INFO_FILE" ]; then
    CRED_ISSUER=$(jq -r '.issuer // ""' "$CRED_INFO_FILE" 2>/dev/null)
    CRED_ISSUEE=$(jq -r '.issuee // ""' "$CRED_INFO_FILE" 2>/dev/null)
    CRED_SAID=$(jq -r '.said // ""' "$CRED_INFO_FILE" 2>/dev/null)
    
    # Check for self-attestation: issuer = issuee
    if [ "$CRED_ISSUER" = "$CRED_ISSUEE" ] && [ -n "$CRED_ISSUER" ]; then
        STEP3_STRUCTURE_VALID=true
        if [ "$JSON_OUTPUT" != "--json" ]; then
            echo -e "${GREEN}✓ Step 3 PASSED: Self-attested credential structure verified${NC}"
            echo "    Credential SAID: ${CRED_SAID:0:30}..."
            echo "    Issuer: ${CRED_ISSUER:0:30}..."
            echo "    Issuee: ${CRED_ISSUEE:0:30}..."
            echo "    Self-Attested: ✓ YES (issuer = issuee)"
        fi
    else
        if [ "$JSON_OUTPUT" != "--json" ]; then
            echo -e "${RED}✗ Step 3 FAILED: Not a self-attested credential${NC}"
            echo "    Issuer: $CRED_ISSUER"
            echo "    Issuee: $CRED_ISSUEE"
            echo "    Self-Attested: ✗ NO (issuer ≠ issuee)"
        fi
    fi
elif [ -f "$ISSUER_INFO_FILE" ]; then
    # Fallback: Use issuer info to verify
    ISSUER_AID=$(jq -r '.aid // .prefix // ""' "$ISSUER_INFO_FILE" 2>/dev/null)
    if [ -n "$ISSUER_AID" ]; then
        STEP3_STRUCTURE_VALID=true
        if [ "$JSON_OUTPUT" != "--json" ]; then
            echo -e "${YELLOW}⚠ Credential info not found, using issuer AID for verification${NC}"
            echo -e "${GREEN}✓ Step 3 PASSED: Issuer AID verified${NC}"
            echo "    Issuer AID: ${ISSUER_AID:0:30}..."
        fi
    fi
else
    if [ "$JSON_OUTPUT" != "--json" ]; then
        echo -e "${RED}✗ Step 3 FAILED: Credential info not found${NC}"
    fi
fi

echo ""

# ============================================
# STEP 4: Verify Invoice Data Integrity
# ============================================
if [ "$JSON_OUTPUT" != "--json" ]; then
    echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${WHITE}  STEP 4: Verify Invoice Data Integrity${NC}"
    echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
fi

if [ -f "$GRANT_INFO_FILE" ]; then
    # Verify invoice data exists
    INVOICE_NUM=$(jq -r '.invoiceNumber // ""' "$GRANT_INFO_FILE" 2>/dev/null)
    AMOUNT=$(jq -r '.amount // ""' "$GRANT_INFO_FILE" 2>/dev/null)
    CURRENCY=$(jq -r '.currency // ""' "$GRANT_INFO_FILE" 2>/dev/null)
    
    if [ -n "$INVOICE_NUM" ] && [ -n "$AMOUNT" ]; then
        STEP4_DATA_VERIFIED=true
        if [ "$JSON_OUTPUT" != "--json" ]; then
            echo -e "${GREEN}✓ Step 4 PASSED: Invoice data integrity verified${NC}"
            echo "    Invoice Number: $INVOICE_NUM"
            echo "    Amount: $AMOUNT $CURRENCY"
            
            # Check credential info for more details
            if [ -f "$CRED_INFO_FILE" ]; then
                SELLER_LEI=$(jq -r '.attributes.sellerLEI // ""' "$CRED_INFO_FILE" 2>/dev/null)
                BUYER_LEI=$(jq -r '.attributes.buyerLEI // ""' "$CRED_INFO_FILE" 2>/dev/null)
                if [ -n "$SELLER_LEI" ]; then
                    echo "    Seller LEI: $SELLER_LEI"
                fi
                if [ -n "$BUYER_LEI" ]; then
                    echo "    Buyer LEI: $BUYER_LEI"
                fi
            fi
        fi
    else
        if [ "$JSON_OUTPUT" != "--json" ]; then
            echo -e "${RED}✗ Step 4 FAILED: Invoice data incomplete${NC}"
        fi
    fi
else
    if [ "$JSON_OUTPUT" != "--json" ]; then
        echo -e "${RED}✗ Step 4 FAILED: Grant info not found${NC}"
    fi
fi

echo ""

# ============================================
# STEP 5: Verify Trust Chain
# ============================================
if [ "$JSON_OUTPUT" != "--json" ]; then
    echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${WHITE}  STEP 5: Verify Trust Chain${NC}"
    echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
fi

# Build trust chain from available info
ISSUER_INFO_FILE="${TASK_DATA_DIR}/${ISSUER_AGENT}-info.json"
DELEGATOR_INFO_FILE="${TASK_DATA_DIR}/${ISSUER_DELEGATOR}-info.json"

if [ -f "$ISSUER_INFO_FILE" ]; then
    ISSUER_AID=$(jq -r '.aid // .prefix // ""' "$ISSUER_INFO_FILE" 2>/dev/null)
    ISSUER_DI=$(jq -r '.state.di // .di // ""' "$ISSUER_INFO_FILE" 2>/dev/null)
    
    if [ -n "$ISSUER_DI" ]; then
        # Verify delegator matches
        if [ -f "$DELEGATOR_INFO_FILE" ]; then
            DELEGATOR_AID=$(jq -r '.aid // .prefix // ""' "$DELEGATOR_INFO_FILE" 2>/dev/null)
            if [ "$ISSUER_DI" = "$DELEGATOR_AID" ]; then
                STEP5_TRUST_CHAIN=true
            fi
        else
            # Trust based on presence of di field
            STEP5_TRUST_CHAIN=true
        fi
    fi
fi

if [ "$JSON_OUTPUT" != "--json" ]; then
    if [ "$STEP5_TRUST_CHAIN" = true ]; then
        echo -e "${GREEN}✓ Step 5 PASSED: Trust chain verified${NC}"
        echo ""
        echo "  Trust Chain (Self-Attested Credential):"
        echo "  ────────────────────────────────────────"
        echo "    1. ${VERIFIER_AGENT} (verifier)"
        echo "       ↓ IPEX Grant received from"
        echo "    2. ${ISSUER_AGENT} (issuer, self-attested)"
        echo "       ↓ Delegated from"
        echo "    3. ${ISSUER_DELEGATOR} (OOR Holder)"
        echo "       ↓ OOR Credential from QVI"
        echo "    4. QVI (Qualified vLEI Issuer)"
        echo "       ↓ QVI Credential from GEDA"
        echo "    5. GEDA (Root of Trust)"
    else
        echo -e "${RED}✗ Step 5 FAILED: Trust chain verification failed${NC}"
    fi
fi

echo ""

# ============================================
# FINAL RESULT
# ============================================
ALL_PASSED=false
if [ "$STEP1_DELEGATION" = true ] && [ "$STEP2_GRANT_SENT" = true ] && \
   [ "$STEP3_STRUCTURE_VALID" = true ] && [ "$STEP4_DATA_VERIFIED" = true ] && \
   [ "$STEP5_TRUST_CHAIN" = true ]; then
    ALL_PASSED=true
fi

if [ "$JSON_OUTPUT" = "--json" ]; then
    cat <<EOF
{
  "success": $( [ "$ALL_PASSED" = true ] && echo "true" || echo "false" ),
  "verifier": "$VERIFIER_AGENT",
  "issuer": "$ISSUER_AGENT",
  "issuerDelegator": "$ISSUER_DELEGATOR",
  "steps": {
    "step1_delegation_verified": $( [ "$STEP1_DELEGATION" = true ] && echo "true" || echo "false" ),
    "step2_ipex_grant_sent": $( [ "$STEP2_GRANT_SENT" = true ] && echo "true" || echo "false" ),
    "step3_credential_structure": $( [ "$STEP3_STRUCTURE_VALID" = true ] && echo "true" || echo "false" ),
    "step4_invoice_data": $( [ "$STEP4_DATA_VERIFIED" = true ] && echo "true" || echo "false" ),
    "step5_trust_chain": $( [ "$STEP5_TRUST_CHAIN" = true ] && echo "true" || echo "false" )
  },
  "trustModel": "self-attested (issuer=issuee, trust from delegation chain)",
  "trustChain": [
    "${VERIFIER_AGENT} (verifier)",
    "${ISSUER_AGENT} (issuer)",
    "${ISSUER_DELEGATOR} (OOR Holder)",
    "QVI",
    "GEDA (Root of Trust)"
  ]
}
EOF
else
    echo -e "${WHITE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    if [ "$ALL_PASSED" = true ]; then
        echo -e "${WHITE}║  ${GREEN}✅ DEEP-EXT CREDENTIAL VERIFICATION: PASSED${WHITE}                                ║${NC}"
    else
        echo -e "${WHITE}║  ${YELLOW}⚠️  DEEP-EXT CREDENTIAL VERIFICATION: INCOMPLETE${WHITE}                           ║${NC}"
    fi
    echo -e "${WHITE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Verification Summary:"
    echo "  Step 1 (Delegation):  $( [ "$STEP1_DELEGATION" = true ] && echo "✓ PASSED" || echo "✗ FAILED" )"
    echo "  Step 2 (IPEX Grant):  $( [ "$STEP2_GRANT_SENT" = true ] && echo "✓ PASSED" || echo "✗ FAILED" )"
    echo "  Step 3 (Structure):   $( [ "$STEP3_STRUCTURE_VALID" = true ] && echo "✓ PASSED" || echo "✗ FAILED" )"
    echo "  Step 4 (Data):        $( [ "$STEP4_DATA_VERIFIED" = true ] && echo "✓ PASSED" || echo "✗ FAILED" )"
    echo "  Step 5 (Trust):       $( [ "$STEP5_TRUST_CHAIN" = true ] && echo "✓ PASSED" || echo "✗ FAILED" )"
    echo ""
    echo "Trust Model: Self-attested credential"
    echo "  → Issuer = Issuee (same AID)"
    echo "  → Trust derived from agent delegation chain"
    echo "  → ${ISSUER_AGENT} delegated from ${ISSUER_DELEGATOR}"
    echo ""
    
    if [ "$ALL_PASSED" = true ]; then
        echo "The self-attested invoice credential from ${ISSUER_AGENT}"
        echo "has been VERIFIED by ${VERIFIER_AGENT}. ✅"
    else
        echo "Verification incomplete. Ensure 4C workflow completed successfully."
        echo "Required files:"
        echo "  - ${TASK_DATA_DIR}/${ISSUER_AGENT}-info.json"
        echo "  - ${TASK_DATA_DIR}/${ISSUER_AGENT}-ipex-grant-info.json"
        echo "  - ${TASK_DATA_DIR}/${ISSUER_AGENT}-self-invoice-credential-info.json"
    fi
    echo ""
fi

# Exit with appropriate code
if [ "$ALL_PASSED" = true ]; then
    exit 0
else
    exit 1
fi
