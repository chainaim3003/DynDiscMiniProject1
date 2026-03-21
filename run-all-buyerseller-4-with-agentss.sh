#!/bin/bash
################################################################################
# run-all-buyerseller-4-with-agents.sh - Configuration-Driven vLEI with Self-Attested Invoices
#
# Purpose: Same as run-all-buyerseller-3-with-agents.sh BUT with SELF-ATTESTED invoice credentials
#
# KEY DIFFERENCE: Invoice credentials are self-attested by jupiterSalesAgent
#                 (issuer = issuee = jupiterSalesAgent), NOT chained to OOR
#
# Flow:
#   1-4. Same as run-all-buyerseller-3 (GEDA, QVI, LE, OOR, Agents)
#   5. ✨ NEW: Self-Attested Invoice Workflow:
#      - jupiterSalesAgent creates self-attested invoice
#      - jupiterSalesAgent stores in its own KERIA
#      - jupiterSalesAgent sends IPEX grant to tommyBuyerAgent
#      - tommyBuyerAgent admits the IPEX grant
#      - Both agents can query KERIA for the credential
#
# Date: November 14, 2025
# Updated: Self-attested invoice credential workflow
################################################################################

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration file location
CONFIG_FILE="./appconfig/configBuyerSellerAIAgent1.json"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  vLEI Configuration-Driven System${NC}"
echo -e "${BLUE}  ✨ WITH SELF-ATTESTED INVOICES${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

################################################################################
# SECTIONS 1-4: Identical to run-all-buyerseller-3-with-agents.sh
# These sections handle GEDA, QVI, LE, OOR, and Agent Delegation
################################################################################

echo -e "${YELLOW}[1/5] Validating Configuration...${NC}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Validate JSON syntax
if ! jq empty "$CONFIG_FILE" 2>/dev/null; then
    echo -e "${RED}ERROR: Invalid JSON in configuration file${NC}"
    exit 1
fi

# Extract configuration values
ROOT_ALIAS=$(jq -r '.root.alias' "$CONFIG_FILE")
QVI_ALIAS=$(jq -r '.qvi.alias' "$CONFIG_FILE")
QVI_LEI=$(jq -r '.qvi.lei' "$CONFIG_FILE")
ORG_COUNT=$(jq -r '.organizations | length' "$CONFIG_FILE")

echo -e "${GREEN}✓ Configuration validated${NC}"
echo "  Root: $ROOT_ALIAS"
echo "  QVI: $QVI_ALIAS (LEI: $QVI_LEI)"
echo "  Organizations: $ORG_COUNT"
echo ""

################################################################################
# SECTION 2: GEDA & QVI Setup
################################################################################

echo -e "${YELLOW}[2/5] GEDA & QVI Setup...${NC}"
echo "Creating root of trust and Qualified vLEI Issuer..."
echo ""

# GEDA AID Creation
echo -e "${BLUE}→ Creating GEDA AID...${NC}"
./task-scripts/geda/geda-aid-create.sh

# Recreate verifier with GEDA AID
echo -e "${BLUE}→ Recreating verifier with GEDA AID...${NC}"
./task-scripts/verifier/recreate-with-geda-aid.sh

# QVI AID Delegation
echo -e "${BLUE}→ Creating delegated QVI AID...${NC}"
./task-scripts/qvi/qvi-aid-delegate-create.sh
./task-scripts/geda/geda-delegate-approve.sh
./task-scripts/qvi/qvi-aid-delegate-finish.sh

# OOBI Resolution
echo -e "${BLUE}→ Resolving OOBI between GEDA and QVI...${NC}"
./task-scripts/geda/geda-oobi-resolve-qvi.sh

# Mutual challenge-response
echo -e "${BLUE}→ GEDA challenges QVI...${NC}"
./task-scripts/geda/geda-challenge-qvi.sh
./task-scripts/qvi/qvi-respond-geda-challenge.sh
./task-scripts/geda/geda-verify-qvi-response.sh

echo -e "${BLUE}→ QVI challenges GEDA...${NC}"
./task-scripts/qvi/qvi-challenge-geda.sh
./task-scripts/geda/geda-respond-qvi-challenge.sh
./task-scripts/qvi/qvi-verify-geda-response.sh

# QVI Credential Issuance
echo -e "${BLUE}→ Issuing QVI credential...${NC}"
./task-scripts/geda/geda-registry-create.sh
./task-scripts/geda/geda-acdc-issue-qvi.sh
./task-scripts/qvi/qvi-acdc-admit-qvi.sh

# QVI presents credential to verifier
echo -e "${BLUE}→ QVI presents credential to verifier...${NC}"
./task-scripts/qvi/qvi-oobi-resolve-verifier.sh
./task-scripts/qvi/qvi-acdc-present-qvi.sh

echo -e "${GREEN}✓ GEDA & QVI setup complete${NC}"
echo ""

################################################################################
# SECTION 3: Organization Loop
################################################################################

echo -e "${YELLOW}[3/5] Processing Organizations...${NC}"
echo ""

for ((org_idx=0; org_idx<$ORG_COUNT; org_idx++)); do
    
    ORG_ID=$(jq -r ".organizations[$org_idx].id" "$CONFIG_FILE")
    ORG_ALIAS=$(jq -r ".organizations[$org_idx].alias" "$CONFIG_FILE")
    ORG_NAME=$(jq -r ".organizations[$org_idx].name" "$CONFIG_FILE")
    ORG_LEI=$(jq -r ".organizations[$org_idx].lei" "$CONFIG_FILE")
    ORG_REGISTRY=$(jq -r ".organizations[$org_idx].registryName" "$CONFIG_FILE")
    PERSON_COUNT=$(jq -r ".organizations[$org_idx].persons | length" "$CONFIG_FILE")
    
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Organization: $ORG_NAME${NC}"
    echo -e "${BLUE}║  LEI: $ORG_LEI${NC}"
    echo -e "${BLUE}║  Alias: $ORG_ALIAS${NC}"
    echo -e "${BLUE}║  Persons: $PERSON_COUNT${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # LE AID Creation
    echo -e "${BLUE}  → Creating LE AID for $ORG_NAME...${NC}"
    ./task-scripts/le/le-aid-create.sh "$ORG_ALIAS"
    
    # OOBI Resolution
    echo -e "${BLUE}  → Resolving OOBI between LE and QVI...${NC}"
    ./task-scripts/le/le-oobi-resolve-qvi.sh
    ./task-scripts/qvi/qvi-oobi-resolve-le.sh
    
    # LE Credential Issuance
    echo -e "${BLUE}  → Creating QVI registry for LE credentials...${NC}"
    ./task-scripts/qvi/qvi-registry-create.sh
    
    echo -e "${BLUE}  → Issuing LE credential to $ORG_NAME...${NC}"
    echo -e "${GREEN}    ✓ Using LEI $ORG_LEI from configuration${NC}"
    ./task-scripts/qvi/qvi-acdc-issue-le.sh "$ORG_LEI"
    ./task-scripts/le/le-acdc-admit-le.sh "$ORG_ALIAS"
    
    # LE presents credential
    echo -e "${BLUE}  → LE presents credential to verifier...${NC}"
    ./task-scripts/le/le-oobi-resolve-verifier.sh
    ./task-scripts/le/le-acdc-present-le.sh "$ORG_ALIAS"
    
    echo -e "${GREEN}  ✓ LE credential issued and presented for $ORG_NAME${NC}"
    echo ""
    
    ##########################################################################
    # SECTION 4: Person Loop
    ##########################################################################
    
    echo -e "${YELLOW}  [4/5] Processing Persons for $ORG_NAME...${NC}"
    echo ""
    
    for ((person_idx=0; person_idx<$PERSON_COUNT; person_idx++)); do
        
        PERSON_ALIAS=$(jq -r ".organizations[$org_idx].persons[$person_idx].alias" "$CONFIG_FILE")
        PERSON_NAME=$(jq -r ".organizations[$org_idx].persons[$person_idx].legalName" "$CONFIG_FILE")
        PERSON_ROLE=$(jq -r ".organizations[$org_idx].persons[$person_idx].officialRole" "$CONFIG_FILE")
        
        echo -e "${BLUE}    ┌──────────────────────────────────────────────────────┐${NC}"
        echo -e "${BLUE}    │  Person: $PERSON_NAME${NC}"
        echo -e "${BLUE}    │  Role: $PERSON_ROLE${NC}"
        echo -e "${BLUE}    │  Alias: $PERSON_ALIAS${NC}"
        echo -e "${BLUE}    └──────────────────────────────────────────────────────┘${NC}"
        echo ""
        
        # Person AID Creation
        echo -e "${BLUE}      → Creating Person AID...${NC}"
        ./task-scripts/person/person-aid-create.sh "$PERSON_ALIAS"
        
        # OOBI Resolution
        echo -e "${BLUE}      → Resolving OOBIs for Person...${NC}"
        ./task-scripts/person/person-oobi-resolve-le.sh
        ./task-scripts/le/le-oobi-resolve-person.sh
        ./task-scripts/qvi/qvi-oobi-resolve-person.sh
        ./task-scripts/person/person-oobi-resolve-qvi.sh
        ./task-scripts/person/person-oobi-resolve-verifier.sh
        
        # OOR Credential Issuance
        echo -e "${BLUE}      → Creating LE registry for OOR credentials...${NC}"
        ./task-scripts/le/le-registry-create.sh "$ORG_ALIAS"
        
        # Step 1: LE issues OOR_AUTH to QVI
        echo -e "${BLUE}      → LE issues OOR_AUTH credential for $PERSON_NAME...${NC}"
        echo -e "${GREEN}        ✓ Using person: $PERSON_NAME, role: $PERSON_ROLE, LEI: $ORG_LEI from configuration${NC}"
        ./task-scripts/le/le-acdc-issue-oor-auth.sh "$PERSON_NAME" "$PERSON_ROLE" "$ORG_LEI" "$ORG_ALIAS"
        ./task-scripts/qvi/qvi-acdc-admit-oor-auth.sh
        
        # Step 2: QVI issues OOR to Person
        echo -e "${BLUE}      → QVI issues OOR credential to $PERSON_NAME...${NC}"
        echo -e "${GREEN}        ✓ Using person: $PERSON_NAME, role: $PERSON_ROLE, LEI: $ORG_LEI from configuration${NC}"
        ./task-scripts/qvi/qvi-acdc-issue-oor.sh "$PERSON_NAME" "$PERSON_ROLE" "$ORG_LEI"
        ./task-scripts/person/person-acdc-admit-oor.sh "$PERSON_ALIAS"
        
        # Person presents OOR credential
        echo -e "${BLUE}      → Person presents OOR credential to verifier...${NC}"
        ./task-scripts/person/person-acdc-present-oor.sh "$PERSON_ALIAS"
        
        echo -e "${GREEN}      ✓ OOR credential issued and presented for $PERSON_NAME${NC}"
        echo ""
        
        ##########################################################################
        # Agent Delegation Workflow
        ##########################################################################
        
        AGENT_COUNT=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents | length" "$CONFIG_FILE")
        if [ "$AGENT_COUNT" -gt 0 ]; then
            echo -e "${CYAN}      ╔═══════════════════════════════════════════════════════╗${NC}"
            echo -e "${CYAN}      ║  ✨ AGENT DELEGATION WORKFLOW                        ║${NC}"
            echo -e "${CYAN}      ╚═══════════════════════════════════════════════════════╝${NC}"
            echo -e "${BLUE}      → Processing $AGENT_COUNT delegated agent(s)...${NC}"
            echo ""
            
            for ((agent_idx=0; agent_idx<$AGENT_COUNT; agent_idx++)); do
                AGENT_ALIAS=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents[$agent_idx].alias" "$CONFIG_FILE")
                AGENT_TYPE=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents[$agent_idx].agentType" "$CONFIG_FILE")
                
                echo -e "${CYAN}        ┌─────────────────────────────────────────────────┐${NC}"
                echo -e "${CYAN}        │  Agent: $AGENT_ALIAS${NC}"
                echo -e "${CYAN}        │  Type: $AGENT_TYPE${NC}"
                echo -e "${CYAN}        │  Delegated from: $PERSON_ALIAS${NC}"
                echo -e "${CYAN}        └─────────────────────────────────────────────────┘${NC}"
                echo ""
                
                # Agent delegation steps
                echo -e "${BLUE}          [1/5] Creating agent delegation request...${NC}"
                ./task-scripts/person/person-delegate-agent-create.sh "$PERSON_ALIAS" "$AGENT_ALIAS"
                
                echo -e "${BLUE}          [2/5] OOR Holder approves delegation...${NC}"
                ./task-scripts/person/person-approve-agent-delegation.sh "$PERSON_ALIAS" "$AGENT_ALIAS"
                
                echo -e "${BLUE}          [3/5] Agent completes delegation...${NC}"
                ./task-scripts/agent/agent-aid-delegate-finish.sh "$AGENT_ALIAS" "$PERSON_ALIAS"
                
                # Agent resolves OOBIs
                echo -e "${BLUE}          [4/5] Agent resolves OOBIs...${NC}"
                echo -e "${BLUE}            → Resolving QVI OOBI...${NC}"
                ./task-scripts/agent/agent-oobi-resolve-qvi.sh "$AGENT_ALIAS"
                
                echo -e "${BLUE}            → Resolving LE OOBI...${NC}"
                ./task-scripts/agent/agent-oobi-resolve-le.sh "$AGENT_ALIAS" "$ORG_ALIAS"
                
                echo -e "${BLUE}            → Resolving Sally verifier OOBI...${NC}"
                ./task-scripts/agent/agent-oobi-resolve-verifier.sh "$AGENT_ALIAS"
                
                # Verify delegation
                echo -e "${BLUE}          [5/5] Verifying agent delegation via Sally...${NC}"
                ./task-scripts/agent/agent-verify-delegation.sh "$AGENT_ALIAS" "$PERSON_ALIAS"
                
                echo -e "${GREEN}          ✓ Agent $AGENT_ALIAS delegation complete and verified${NC}"
                
                if [ -f "./task-data/${AGENT_ALIAS}-info.json" ]; then
                    AGENT_AID=$(cat "./task-data/${AGENT_ALIAS}-info.json" | jq -r .aid)
                    echo -e "${GREEN}          Agent AID: $AGENT_AID${NC}"
                fi
                echo ""
                
            done  # End agent loop
            
            echo -e "${GREEN}      ✓ All agents processed for $PERSON_NAME${NC}"
            echo ""
        fi
        
    done  # End person loop
    
    echo -e "${GREEN}  ✓ All persons processed for $ORG_NAME${NC}"
    echo ""
    
done  # End organization loop

echo -e "${GREEN}✓ All organizations processed${NC}"
echo ""

################################################################################
# ✨ NEW SECTION 5: Self-Attested Invoice Credential Workflow
# 
# KEY CHANGES FROM run-all-buyerseller-3:
# - Invoice is SELF-ATTESTED by jupiterSalesAgent (issuer = issuee)
# - NO chaining to OOR credential
# - jupiterSalesAgent sends IPEX grant to tommyBuyerAgent
# - tommyBuyerAgent admits the IPEX grant
# - All proofs happen in KERIA agent
# - Both agents can query KERIA for the credential
################################################################################

echo -e "${MAGENTA}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║  🧾 SELF-ATTESTED INVOICE CREDENTIAL WORKFLOW          ║${NC}"
echo -e "${MAGENTA}║                                                          ║${NC}"
echo -e "${MAGENTA}║  Issuer: jupiterSalesAgent (SELF-ATTESTED)             ║${NC}"
echo -e "${MAGENTA}║  Issuee: jupiterSalesAgent (SAME AS ISSUER)            ║${NC}"
echo -e "${MAGENTA}║  Edge: NONE (no OOR chain)                             ║${NC}"
echo -e "${MAGENTA}║  Grant: jupiterSalesAgent → tommyBuyerAgent via IPEX   ║${NC}"
echo -e "${MAGENTA}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}Creating self-attested invoice from jupiterSalesAgent...${NC}"
echo ""

# Step 1: Create invoice registry for jupiterSalesAgent
AGENT_REGISTRY_FILE="./task-data/jupiterSalesAgent-invoice-registry-info.json"
if [ ! -f "$AGENT_REGISTRY_FILE" ]; then
    echo -e "${BLUE}  [1/5] Creating invoice credential registry in jupiterSalesAgent...${NC}"
    ./task-scripts/invoice/invoice-registry-create-agent.sh "jupiterSalesAgent" "JUPITER_AGENT_INVOICE_REGISTRY"
else
    echo -e "${GREEN}  [1/5] Invoice registry already exists ✓${NC}"
fi

# Step 2: Issue SELF-ATTESTED invoice credential
# NOTE: Issuer and issuee are BOTH jupiterSalesAgent
echo -e "${BLUE}  [2/5] Issuing SELF-ATTESTED invoice credential...${NC}"
echo -e "${CYAN}        Issuer: jupiterSalesAgent${NC}"
echo -e "${CYAN}        Issuee: jupiterSalesAgent (SELF-ATTESTED)${NC}"
echo -e "${CYAN}        Edge: NONE${NC}"
./task-scripts/invoice/invoice-acdc-issue-self-attested.sh \
    "jupiterSalesAgent" \
    "./appconfig/invoiceConfig.json"

# Step 3: jupiterSalesAgent admits its own credential
echo -e "${BLUE}  [3/5] jupiterSalesAgent admitting self-attested invoice...${NC}"
./task-scripts/invoice/invoice-acdc-admit-self.sh "jupiterSalesAgent"

# Step 4: jupiterSalesAgent sends IPEX grant to tommyBuyerAgent
echo -e "${BLUE}  [4/5] Sending IPEX grant to tommyBuyerAgent...${NC}"
./task-scripts/invoice/invoice-ipex-grant.sh \
    "jupiterSalesAgent" \
    "tommyBuyerAgent"

# Step 5: tommyBuyerAgent admits the IPEX grant
echo -e "${BLUE}  [5/5] tommyBuyerAgent admitting IPEX grant...${NC}"
./task-scripts/invoice/invoice-ipex-admit.sh \
    "tommyBuyerAgent" \
    "jupiterSalesAgent"

echo ""
echo -e "${GREEN}✓ Self-Attested Invoice Workflow Complete${NC}"
echo ""

# Verify credential is queryable from KERIA agents
echo -e "${BLUE}Verifying credential storage in KERIA agents...${NC}"
echo ""

# Query from jupiterSalesAgent
echo -e "${BLUE}  → Querying invoice from jupiterSalesAgent's KERIA...${NC}"
./task-scripts/invoice/invoice-query.sh "jupiterSalesAgent"

# Query from tommyBuyerAgent
echo -e "${BLUE}  → Querying invoice from tommyBuyerAgent's KERIA...${NC}"
./task-scripts/invoice/invoice-query.sh "tommyBuyerAgent"

echo -e "${GREEN}✓ Credential verified in both agents' KERIA${NC}"
echo ""

# Display invoice summary
INVOICE_INFO_FILE="./task-data/jupiterSalesAgent-self-invoice-credential-info.json"
if [ -f "$INVOICE_INFO_FILE" ]; then
    INVOICE_NUMBER=$(cat "$INVOICE_INFO_FILE" | jq -r '.invoiceNumber')
    INVOICE_AMOUNT=$(cat "$INVOICE_INFO_FILE" | jq -r '.totalAmount')
    INVOICE_CURRENCY=$(cat "$INVOICE_INFO_FILE" | jq -r '.currency')
    INVOICE_CHAIN=$(cat "$INVOICE_INFO_FILE" | jq -r '.paymentChainID')
    INVOICE_WALLET=$(cat "$INVOICE_INFO_FILE" | jq -r '.paymentWalletAddress')
    INVOICE_REF=$(cat "$INVOICE_INFO_FILE" | jq -r '.ref_uri')
    
    echo -e "${GREEN}📄 Invoice Summary (Self-Attested):${NC}"
    echo -e "${GREEN}  Number: $INVOICE_NUMBER${NC}"
    echo -e "${GREEN}  Amount: $INVOICE_AMOUNT $INVOICE_CURRENCY${NC}"
    echo -e "${GREEN}  Payment Chain: $INVOICE_CHAIN${NC}"
    echo -e "${GREEN}  Wallet Address: $INVOICE_WALLET${NC}"
    echo -e "${GREEN}  Reference URI: $INVOICE_REF${NC}"
    echo -e "${MAGENTA}  Issuer: jupiterSalesAgent (SELF-ATTESTED)${NC}"
    echo -e "${MAGENTA}  Issuee: jupiterSalesAgent${NC}"
    echo -e "${MAGENTA}  Granted to: tommyBuyerAgent via IPEX${NC}"
    echo ""
fi

################################################################################
# SECTION 6: Trust Tree Visualization
################################################################################

echo -e "${YELLOW}[5/5] Generating Trust Tree Visualization...${NC}"
echo ""

TRUST_TREE_FILE="./task-data/trust-tree-buyerseller-self-attested.txt"

cat > "$TRUST_TREE_FILE" << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║          vLEI Trust Chain - Buyer-Seller with Self-Attested Invoices        ║
║                Configuration-Driven System with Agent Delegation             ║
╚══════════════════════════════════════════════════════════════════════════════╝

ROOT (GLEIF External AID)
│
├─ QVI (Qualified vLEI Issuer)
│   │
│   ├─ QVI Credential (issued by GLEIF ROOT)
│   │   └─ Presented to Sally Verifier ✓
│   │
│   ├─── JUPITER KNITTING COMPANY (Seller)
│   │     LEI: 3358004DXAMRWRUIYJ05
│   │     │
│   │     ├─ LE Credential (issued by QVI)
│   │     │   └─ Presented to Sally Verifier ✓
│   │     │
│   │     └─ Chief Sales Officer
│   │         │
│   │         ├─ OOR Credential (issued by QVI to Person)
│   │         │   ├─ Chained to: LE Credential
│   │         │   └─ Presented to Sally Verifier ✓
│   │         │
│   │         └─ ✨ Delegated Agent: jupiterSalesAgent (AI Agent)
│   │             ├─ Agent AID (Delegated from OOR Holder)
│   │             ├─ KEL Seal (Anchored in OOR Holder's KEL)
│   │             ├─ OOBI Resolved (QVI, LE, Sally)
│   │             ├─ ✓ Verified by Sally Verifier
│   │             │
│   │             └─ 🧾 SELF-ATTESTED INVOICE CREDENTIAL
│   │                 ├─ Issuer: jupiterSalesAgent ⚠️ (SELF-ATTESTED)
│   │                 ├─ Issuee: jupiterSalesAgent (SAME AS ISSUER)
│   │                 ├─ Edge: NONE (no OOR chain)
│   │                 ├─ Trust: Agent delegation chain only
│   │                 ├─ Stored in: jupiterSalesAgent's KERIA
│   │                 ├─ ✓ Queryable from KERIA
│   │                 │
│   │                 └─ IPEX Grant ➜ tommyBuyerAgent
│   │                     ├─ Grant sent via IPEX protocol
│   │                     ├─ ✓ Admitted by tommyBuyerAgent
│   │                     └─ ✓ Queryable from tommyBuyerAgent's KERIA
│   │
│   └─── TOMMY HILFIGER EUROPE B.V. (Buyer)
│         LEI: 54930012QJWZMYHNJW95
│         │
│         ├─ LE Credential (issued by QVI)
│         │   └─ Presented to Sally Verifier ✓
│         │
│         └─ Chief Procurement Officer
│             │
│             ├─ OOR Credential (issued by QVI to Person)
│             │   ├─ Chained to: LE Credential
│             │   └─ Presented to Sally Verifier ✓
│             │
│             └─ ✨ Delegated Agent: tommyBuyerAgent (AI Agent)
│                 ├─ Agent AID (Delegated from OOR Holder)
│                 ├─ KEL Seal (Anchored in OOR Holder's KEL)
│                 ├─ OOBI Resolved (QVI, LE, Sally)
│                 ├─ ✓ Verified by Sally Verifier
│                 │
│                 └─ 🧾 RECEIVED INVOICE (via IPEX)
│                     ├─ Source: jupiterSalesAgent
│                     ├─ ✓ IPEX grant admitted
│                     └─ ✓ Queryable from KERIA

╔══════════════════════════════════════════════════════════════════════════════╗
║                     Self-Attested Invoice Workflow Summary                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. jupiterSalesAgent creates SELF-ATTESTED invoice
   ├─ Issuer: jupiterSalesAgent
   ├─ Issuee: jupiterSalesAgent (SELF-ATTESTED)
   ├─ Edge: NONE (no OOR reference)
   └─ Trust Chain: Agent delegation only (agent → OOR → LE → QVI → Root)

2. jupiterSalesAgent stores credential in its KERIA
   └─ ✓ Queryable via KERIA API

3. jupiterSalesAgent sends IPEX grant to tommyBuyerAgent
   └─ IPEX protocol for credential sharing

4. tommyBuyerAgent admits IPEX grant
   ├─ ✓ Grant accepted
   └─ ✓ Credential stored in tommyBuyerAgent's KERIA

5. Both agents can query credential
   ├─ jupiterSalesAgent: Original self-attested credential
   └─ tommyBuyerAgent: Received credential via IPEX

╔══════════════════════════════════════════════════════════════════════════════╗
║                    Key Difference from run-all-buyerseller-3                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

run-all-buyerseller-3:
  • Invoice issued by: Jupiter_Chief_Sales_Officer (OOR Holder)
  • Invoice held by: tommyBuyerAgent
  • Edge: References OOR credential (I2I chain)
  • Trust Chain: Invoice → OOR → LE → QVI → Root

run-all-buyerseller-4 (THIS SCRIPT):
  • Invoice issued by: jupiterSalesAgent (SELF-ATTESTED)
  • Invoice held by: jupiterSalesAgent (initially), then granted to tommyBuyerAgent
  • Edge: NONE (no OOR reference)
  • Trust Chain: Agent delegation only (agent → OOR → LE → QVI → Root)

╔══════════════════════════════════════════════════════════════════════════════╗
║                            Verification Points                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Sally Verifier validates:
  ✓ Digital signatures (KERI)
  ✓ Credential chain integrity (for chained credentials)
  ✓ Root of trust (GEDA)
  ✓ Delegation relationships
  ✓ Schema compliance
  ✓ Edge validity (where applicable)
  ✓ Revocation status
  ✓ OOBI resolution
  ✓ Agent delegation chain
  ✓ OOR Holder KEL seal
  ⚠️ Self-attested invoices: No OOR chain validation

KERIA Query Verification:
  ✓ jupiterSalesAgent can query its self-attested credential
  ✓ tommyBuyerAgent can query received credential
  ✓ All proofs stored in respective KERIA agents

EOF

echo -e "${GREEN}✓ Trust tree visualization created: $TRUST_TREE_FILE${NC}"
echo ""
cat "$TRUST_TREE_FILE"
echo ""

################################################################################
# SECTION 7: Summary
################################################################################

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║               Execution Complete (Self-Attested)             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}✅ Setup Complete!${NC}"
echo ""
echo -e "${MAGENTA}🧾 Self-Attested Invoice Workflow:${NC}"
echo "  • jupiterSalesAgent issued invoice to ITSELF (self-attested)"
echo "  • Invoice credential stored in jupiterSalesAgent's KERIA"
echo "  • Invoice granted via IPEX to tommyBuyerAgent"
echo "  • tommyBuyerAgent admitted and now holds the invoice"
echo "  • Both agents can query KERIA for the credential"
echo ""

echo -e "${BLUE}📋 Verification:${NC}"
echo "  • Agent delegation: ✓ Verified by Sally"
echo "  • Invoice self-attestation: ✓ Issuer = Issuee"
echo "  • IPEX grant/admit: ✓ Complete"
echo "  • KERIA queryable: ✓ Available in both agents"
echo ""

echo -e "${CYAN}🔑 Key Differences from run-all-buyerseller-3:${NC}"
echo "  • Invoice issuer: jupiterSalesAgent (NOT OOR holder)"
echo "  • Invoice issuee: jupiterSalesAgent (SELF-ATTESTED)"
echo "  • Edge: NONE (no OOR chain reference)"
echo "  • Trust: Through agent delegation chain only"
echo "  • IPEX: Full grant/admit workflow implemented"
echo "  • KERIA: Queryable from both agents"
echo ""

echo -e "${GREEN}✨ vLEI system with self-attested invoices completed successfully!${NC}"
echo ""

exit 0
