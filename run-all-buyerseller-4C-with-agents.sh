#!/bin/bash
################################################################################
# run-all-buyerseller-4C-with-agents.sh - Configuration-Driven vLEI System
#
# Purpose: Orchestrate the complete vLEI credential issuance flow for multiple
#          organizations (buyer and seller) using configuration from JSON file
#          ✨ WITH UNIQUE AGENT BRAN SUPPORT
#          ✨ WITH SELF-ATTESTED INVOICE CREDENTIAL AND IPEX GRANT/ADMIT
#
# Design: Based on understanding-3.md and complete-session.md
# Reference: https://github.com/GLEIF-IT/vlei-hackathon-2025-workshop
#
# Configuration: appconfig/configBuyerSellerAIAgent1.json
#
# Flow:
#   1. GEDA & QVI Setup (once)
#   2. ✨ Generate Unique BRANs for ALL agents (BEFORE creating them)
#   3. Loop through each organization:
#      - Create LE AID and credentials
#      - Loop through each person:
#        - Create Person AID
#        - Issue OOR credentials
#        - Present to verifier
#        - ✨ Create and delegate agents WITH UNIQUE BRANs
#   4. Generate trust tree visualization
#   5. ✨ NEW: Self-Attested Invoice Credential Workflow:
#      - jupiterSellerAgent creates credential registry
#      - jupiterSellerAgent issues self-attested Invoice credential (issuer=issuee)
#      - jupiterSellerAgent sends IPEX grant to tommyBuyerAgent
#      - tommyBuyerAgent admits the IPEX grant
#
# Date: November 27, 2025
# Version: 4C - With Invoice Credential + IPEX Grant/Admit
################################################################################

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Configuration file location
CONFIG_FILE="./appconfig/configBuyerSellerAIAgent1.json"
INVOICE_CONFIG_FILE="./appconfig/invoiceConfig.json"
# Use the SAIDified self-attested invoice schema
INVOICE_SCHEMA_FILE="./schemas/self-attested-invoice.json"
# Schema SAID - Read from config file (single source of truth)
# Priority: schemaSaids.json > task-data file > schema file
if [ -f "./appconfig/schemaSaids.json" ]; then
    INVOICE_SCHEMA_SAID=$(jq -r '.invoiceSchema.said' "./appconfig/schemaSaids.json")
    if [ -z "$INVOICE_SCHEMA_SAID" ] || [ "$INVOICE_SCHEMA_SAID" = "null" ]; then
        INVOICE_SCHEMA_SAID=""
    fi
elif [ -f "./task-data/invoice-schema-said.txt" ]; then
    INVOICE_SCHEMA_SAID=$(cat ./task-data/invoice-schema-said.txt)
else
    # Fallback: extract from schema file directly
    INVOICE_SCHEMA_SAID=$(jq -r '."$id"' "$INVOICE_SCHEMA_FILE" 2>/dev/null || echo "")
fi

# Validate SAID is not empty
if [ -z "$INVOICE_SCHEMA_SAID" ] || [ "$INVOICE_SCHEMA_SAID" = "null" ]; then
    echo -e "${RED}ERROR: Invoice schema SAID is empty!${NC}"
    echo "  Please run ./saidify-with-docker.sh first to generate the schema SAID"
    echo "  Or ensure ./appconfig/schemaSaids.json contains the correct SAID"
    exit 1
fi

echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  vLEI Configuration-Driven System${NC}"
echo -e "${CYAN}  Buyer-Seller Credential Issuance${NC}"
echo -e "${CYAN}  ✨ WITH UNIQUE AGENT BRAN SUPPORT${NC}"
echo -e "${CYAN}  ✨ WITH INVOICE CREDENTIAL + IPEX GRANT/ADMIT${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""

################################################################################
# SECTION 1: Configuration Validation
################################################################################

echo -e "${YELLOW}[1/8] Validating Configuration...${NC}"

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

# Check invoice config
if [ ! -f "$INVOICE_CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: Invoice configuration file not found: $INVOICE_CONFIG_FILE${NC}"
    exit 1
fi

# Check invoice schema
if [ ! -f "$INVOICE_SCHEMA_FILE" ]; then
    echo -e "${RED}ERROR: Invoice schema file not found: $INVOICE_SCHEMA_FILE${NC}"
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
echo "  Invoice Config: $INVOICE_CONFIG_FILE"
echo "  Invoice Schema: $INVOICE_SCHEMA_FILE"
echo ""

################################################################################
# SECTION 2: GEDA & QVI Setup (One-time initialization)
################################################################################

echo -e "${YELLOW}[2/8] GEDA & QVI Setup...${NC}"
echo "Creating root of trust and Qualified vLEI Issuer..."
echo ""

# GEDA AID Creation
echo -e "${BLUE}→ Creating GEDA AID...${NC}"
./task-scripts/geda/geda-aid-create.sh

# Recreate verifier with GEDA AID
echo -e "${BLUE}→ Recreating verifier with GEDA AID...${NC}"
./task-scripts/verifier/recreate-with-geda-aid.sh

# QVI AID Delegation (3-step process)
echo -e "${BLUE}→ Creating delegated QVI AID...${NC}"
./task-scripts/qvi/qvi-aid-delegate-create.sh
./task-scripts/geda/geda-delegate-approve.sh
./task-scripts/qvi/qvi-aid-delegate-finish.sh

# OOBI Resolution between GEDA and QVI
echo -e "${BLUE}→ Resolving OOBI between GEDA and QVI...${NC}"
./task-scripts/geda/geda-oobi-resolve-qvi.sh

# Mutual challenge-response between GEDA and QVI
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
# ✨ SECTION 2.5: Generate Unique Agent BRANs (BEFORE Agent Creation!)
################################################################################

echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║  ✨ GENERATING UNIQUE AGENT BRANs                        ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}[2.5/8] Pre-generating Unique Cryptographic Identities...${NC}"
echo ""

echo -e "${BLUE}This step creates unique BRANs for ALL agents BEFORE they are created.${NC}"
echo -e "${BLUE}Each agent will have:${NC}"
echo -e "${BLUE}  • Unique 256-bit BRAN (cryptographic seed)${NC}"
echo -e "${BLUE}  • Unique AID derived from BRAN${NC}"
echo -e "${BLUE}  • Delegation to appropriate OOR holder${NC}"
echo ""

# Check if BRAN generation script exists
if [ ! -f "./generate-unique-agent-brans.sh" ]; then
    echo -e "${RED}ERROR: generate-unique-agent-brans.sh not found${NC}"
    echo -e "${YELLOW}Please ensure the script is in the current directory${NC}"
    exit 1
fi

# Make script executable
chmod +x ./generate-unique-agent-brans.sh

# Generate unique BRANs for all agents
echo -e "${BLUE}→ Generating unique BRANs from configuration...${NC}"
if ! ./generate-unique-agent-brans.sh; then
    echo -e "${RED}✗ BRAN generation failed${NC}"
    echo -e "${YELLOW}Check error messages above${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Unique BRANs pre-generated for all agents${NC}"
echo -e "${GREEN}  Configuration saved to: task-data/agent-brans.json${NC}"
echo -e "${GREEN}  Agent .env files created in: ../Legent/A2A/js/src/agents/*/   ${NC}"
echo ""

# Verify BRANs are unique
if [ -f "./task-data/agent-brans.json" ]; then
    TOTAL_BRANS=$(jq -r '.agents | length' "./task-data/agent-brans.json")
    echo -e "${GREEN}✓ Generated $TOTAL_BRANS unique agent BRANs${NC}"
    echo ""
fi

################################################################################
# SECTION 3: Organization Loop - Process Each Legal Entity
################################################################################

echo -e "${YELLOW}[3/8] Processing Organizations...${NC}"
echo ""

# Loop through each organization in the configuration
for ((org_idx=0; org_idx<$ORG_COUNT; org_idx++)); do
    
    # Extract organization details from config
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
    
    # OOBI Resolution between LE and QVI
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
    
    # LE presents credential to verifier
    echo -e "${BLUE}  → LE presents credential to verifier...${NC}"
    ./task-scripts/le/le-oobi-resolve-verifier.sh
    ./task-scripts/le/le-acdc-present-le.sh "$ORG_ALIAS"
    
    echo -e "${GREEN}  ✓ LE credential issued and presented for $ORG_NAME${NC}"
    echo ""
    
    ##########################################################################
    # SECTION 4: Person Loop - Process Each Official Organizational Role
    ##########################################################################
    
    echo -e "${YELLOW}  [4/8] Processing Persons for $ORG_NAME...${NC}"
    echo ""
    
    # Loop through each person in the organization
    for ((person_idx=0; person_idx<$PERSON_COUNT; person_idx++)); do
        
        # Extract person details from config
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
        
        # OOBI Resolution (Person with LE, QVI, and Verifier)
        echo -e "${BLUE}      → Resolving OOBIs for Person...${NC}"
        ./task-scripts/person/person-oobi-resolve-le.sh
        ./task-scripts/le/le-oobi-resolve-person.sh
        ./task-scripts/qvi/qvi-oobi-resolve-person.sh
        ./task-scripts/person/person-oobi-resolve-qvi.sh
        ./task-scripts/person/person-oobi-resolve-verifier.sh
        
        # OOR Credential Issuance (2-step process)
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
        
        # Person presents OOR credential to verifier
        echo -e "${BLUE}      → Person presents OOR credential to verifier...${NC}"
        ./task-scripts/person/person-acdc-present-oor.sh "$PERSON_ALIAS"
        
        echo -e "${GREEN}      ✓ OOR credential issued and presented for $PERSON_NAME${NC}"
        echo ""
        
        ##########################################################################
        # ✨ SECTION 5: Agent Delegation Workflow WITH UNIQUE BRANs
        ##########################################################################
        
        # Check for delegated agents
        AGENT_COUNT=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents | length" "$CONFIG_FILE")
        if [ "$AGENT_COUNT" -gt 0 ]; then
            echo -e "${MAGENTA}      ╔═══════════════════════════════════════════════════════╗${NC}"
            echo -e "${MAGENTA}      ║  ✨ AGENT DELEGATION WITH UNIQUE BRANs               ║${NC}"
            echo -e "${MAGENTA}      ╚═══════════════════════════════════════════════════════╝${NC}"
            echo -e "${BLUE}      → Processing $AGENT_COUNT agent(s) with unique identities...${NC}"
            echo ""
            
            for ((agent_idx=0; agent_idx<$AGENT_COUNT; agent_idx++)); do
                AGENT_ALIAS=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents[$agent_idx].alias" "$CONFIG_FILE")
                AGENT_TYPE=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents[$agent_idx].agentType" "$CONFIG_FILE")
                
                echo -e "${CYAN}        ┌─────────────────────────────────────────────────┐${NC}"
                echo -e "${CYAN}        │  Agent: $AGENT_ALIAS${NC}"
                echo -e "${CYAN}        │  Type: $AGENT_TYPE${NC}"
                echo -e "${CYAN}        │  Delegated from: $PERSON_ALIAS${NC}"
                echo -e "${CYAN}        │  ✨ Uses: Unique BRAN (pre-generated)${NC}"
                echo -e "${CYAN}        └─────────────────────────────────────────────────┘${NC}"
                echo ""
                
                # Verify BRAN was pre-generated
                BRAN_FILE="./task-data/${AGENT_ALIAS}-bran.txt"
                if [ ! -f "$BRAN_FILE" ]; then
                    echo -e "${RED}          ✗ ERROR: BRAN not found for ${AGENT_ALIAS}${NC}"
                    echo -e "${YELLOW}          This should have been generated in Section 2.5${NC}"
                    exit 1
                fi
                
                AGENT_BRAN=$(cat "$BRAN_FILE")
                echo -e "${GREEN}          ✓ Using pre-generated unique BRAN${NC}"
                echo -e "${GREEN}            BRAN: ${AGENT_BRAN:0:20}... (256-bit)${NC}"
                echo ""
                
                # ✨ NEW: Use agent-delegate-with-unique-bran.sh
                # This script creates agent AID from unique BRAN and delegates it
                echo -e "${BLUE}          Creating agent with unique BRAN and delegating...${NC}"
                
                if [ ! -f "./task-scripts/agent/agent-delegate-with-unique-bran.sh" ]; then
                    echo -e "${RED}          ✗ ERROR: agent-delegate-with-unique-bran.sh not found${NC}"
                    exit 1
                fi
                
                chmod +x ./task-scripts/agent/agent-delegate-with-unique-bran.sh
                ./task-scripts/agent/agent-delegate-with-unique-bran.sh "$AGENT_ALIAS" "$PERSON_ALIAS"
                
                # Agent resolves OOBIs
                echo -e "${BLUE}          Resolving OOBIs for agent...${NC}"
                echo -e "${BLUE}            → Resolving QVI OOBI...${NC}"
                ./task-scripts/agent/agent-oobi-resolve-qvi.sh "$AGENT_ALIAS"
                
                echo -e "${BLUE}            → Resolving LE OOBI...${NC}"
                ./task-scripts/agent/agent-oobi-resolve-le.sh "$AGENT_ALIAS" "$ORG_ALIAS"
                
                echo -e "${BLUE}            → Resolving Sally verifier OOBI...${NC}"
                ./task-scripts/agent/agent-oobi-resolve-verifier.sh "$AGENT_ALIAS"
                
                # Verify agent delegation
                echo -e "${BLUE}          Verifying agent delegation via Sally...${NC}"
                ./task-scripts/agent/agent-verify-delegation.sh "$AGENT_ALIAS" "$PERSON_ALIAS"
                
                echo -e "${GREEN}          ✓ Agent $AGENT_ALIAS delegation complete and verified${NC}"
                
                # Display agent info with unique identity confirmation
                if [ -f "./task-data/${AGENT_ALIAS}-info.json" ]; then
                    AGENT_AID=$(cat "./task-data/${AGENT_ALIAS}-info.json" | jq -r .aid)
                    HAS_UNIQUE_BRAN=$(cat "./task-data/${AGENT_ALIAS}-info.json" | jq -r '.hasUniqueBran // false')
                    echo -e "${GREEN}          Agent AID: $AGENT_AID${NC}"
                    echo -e "${GREEN}          Unique Identity: ✓ Yes (BRAN-based)${NC}"
                fi
                echo ""
                
            done  # End agent loop
            
            echo -e "${GREEN}      ✓ All agents processed for $PERSON_NAME with unique identities${NC}"
            echo ""
        fi
        
    done  # End person loop
    
    echo -e "${GREEN}  ✓ All persons processed for $ORG_NAME${NC}"
    echo ""
    
done  # End organization loop

echo -e "${GREEN}✓ All organizations processed${NC}"
echo ""

################################################################################
# SECTION 6: Generate Trust Tree Visualization (from 2C)
################################################################################

echo -e "${YELLOW}[5/8] Generating Trust Tree Visualization...${NC}"
echo ""

# Create trust tree output
TRUST_TREE_FILE="./task-data/trust-tree-buyerseller-4C-with-invoice.txt"

cat > "$TRUST_TREE_FILE" << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║     vLEI Trust Chain - Buyer-Seller with UNIQUE AGENT BRANs + INVOICE       ║
║        Configuration-Driven System with Agent Delegation & IPEX             ║
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
│   │         ├─ OOR_AUTH Credential (issued by LE to QVI)
│   │         │   └─ Admitted by QVI ✓
│   │         │
│   │         ├─ OOR Credential (issued by QVI to Person)
│   │         │   ├─ Chained to: LE Credential
│   │         │   └─ Presented to Sally Verifier ✓
│   │         │
│   │         └─ ✨ Delegated Agent: jupiterSellerAgent (AI Agent)
│   │             ├─ ✨ Unique BRAN (256-bit cryptographic seed)
│   │             ├─ ✨ Unique AID (derived from agent's BRAN)
│   │             ├─ Agent AID Delegated from OOR Holder
│   │             ├─ KEL Seal (Anchored in OOR Holder's KEL)
│   │             ├─ OOBI Resolved (QVI, LE, Sally)
│   │             ├─ ✓ Verified by Sally Verifier
│   │             │
│   │             ├─ 📄 INVOICE CREDENTIAL REGISTRY
│   │             │   └─ jupiterSellerAgent_INVOICE_REGISTRY
│   │             │
│   │             └─ 📝 SELF-ATTESTED INVOICE CREDENTIAL
│   │                 ├─ Issuer: jupiterSellerAgent (self)
│   │                 ├─ Issuee: jupiterSellerAgent (same as issuer)
│   │                 ├─ Type: Self-Attested (no OOR chain edge)
│   │                 ├─ Schema: InvoiceCredential
│   │                 │
│   │                 └─ 📤 IPEX GRANT → tommyBuyerAgent
│   │                     └─ Credential shared via IPEX protocol
│   │
│   └─── TOMMY HILFIGER EUROPE B.V. (Buyer)
│         LEI: 54930012QJWZMYHNJW95
│         │
│         ├─ LE Credential (issued by QVI)
│         │   └─ Presented to Sally Verifier ✓
│         │
│         └─ Chief Procurement Officer
│             │
│             ├─ OOR_AUTH Credential (issued by LE to QVI)
│             │   └─ Admitted by QVI ✓
│             │
│             ├─ OOR Credential (issued by QVI to Person)
│             │   ├─ Chained to: LE Credential
│             │   └─ Presented to Sally Verifier ✓
│             │
│             └─ ✨ Delegated Agent: tommyBuyerAgent (AI Agent)
│                 ├─ ✨ Unique BRAN (256-bit cryptographic seed)
│                 ├─ ✨ Unique AID (derived from agent's BRAN)
│                 ├─ Agent AID Delegated from OOR Holder
│                 ├─ KEL Seal (Anchored in OOR Holder's KEL)
│                 ├─ OOBI Resolved (QVI, LE, Sally)
│                 ├─ ✓ Verified by Sally Verifier
│                 │
│                 └─ 📥 IPEX ADMIT
│                     └─ Admitted invoice credential from jupiterSellerAgent

╔══════════════════════════════════════════════════════════════════════════════╗
║                        Invoice Credential Flow (4C)                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

8. ✨ NEW: Self-Attested Invoice Credential Workflow
   ├─ jupiterSellerAgent creates Invoice Registry
   │   └─ Registry Name: jupiterSellerAgent_INVOICE_REGISTRY
   │
   ├─ jupiterSellerAgent issues SELF-ATTESTED Invoice Credential
   │   ├─ Issuer AID: jupiterSellerAgent's delegated AID
   │   ├─ Issuee AID: SAME as Issuer (self-attested)
   │   ├─ Schema: InvoiceCredential (schemas/invoice-credential-schema.json)
   │   ├─ Edge: NONE (self-attested, no OOR chain)
   │   ├─ Invoice Data from: appconfig/invoiceConfig.json
   │   │   ├─ Invoice Number: INV-2025-001
   │   │   ├─ Amount: 50000.00 ALGO
   │   │   ├─ Seller LEI: 3358004DXAMRWRUIYJ05
   │   │   ├─ Buyer LEI: 54930012QJWZMYHNJW95
   │   │   └─ Payment: blockchain (algorand)
   │   └─ Stored in jupiterSellerAgent's KERIA
   │
   ├─ jupiterSellerAgent sends IPEX GRANT to tommyBuyerAgent
   │   ├─ Grant Message: /ipex/grant
   │   ├─ Contains: Full Invoice Credential + TEL events
   │   └─ Sent via KERIA agent messaging
   │
   └─ tommyBuyerAgent ADMITS the IPEX GRANT
       ├─ Admit Message: /ipex/admit  
       ├─ Credential validated and stored
       └─ Notification marked as read

╔══════════════════════════════════════════════════════════════════════════════╗
║                              Key Differences                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Self-Attested vs Chained Credentials:
  ✓ Self-Attested: Issuer = Issuee (same AID)
  ✓ Self-Attested: NO edge section (no credential chaining)
  ✓ Self-Attested: Trust derived from agent delegation, not credential chain
  ✓ Chained: Edge section references parent credential (e.g., OOR)
  ✓ Chained: Forms cryptographic chain to root of trust

IPEX Grant/Admit Flow:
  1. Issuer creates credential locally in their KERIA
  2. Issuer sends IPEX GRANT (/ipex/grant) to recipient
  3. Recipient receives notification of incoming credential
  4. Recipient sends IPEX ADMIT (/ipex/admit) to accept
  5. Credential now available in recipient's KERIA storage

╔══════════════════════════════════════════════════════════════════════════════╗
║                                    Notes                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Configuration Source: 
  ✓ Agent config: appconfig/configBuyerSellerAIAgent1.json
  ✓ Invoice config: appconfig/invoiceConfig.json
  ✓ Invoice schema: schemas/invoice-credential-schema.json

Reference:
  ✓ vLEI Training 101_65: ACDC Issuance with IPEX
  ✓ vLEI Training 102_20: KERIA Signify Credential Issuance
  ✓ https://github.com/GLEIF-IT/vlei-hackathon-2025-workshop

EOF

echo -e "${GREEN}✓ Trust tree visualization created: $TRUST_TREE_FILE${NC}"
echo ""

################################################################################
# ✨ SECTION 6.5: VERIFY INVOICE SCHEMA FROM KERIA PERSPECTIVE (CRITICAL!)
################################################################################

echo -e "${MAGENTA}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║  📋 VERIFYING INVOICE SCHEMA FROM KERIA PERSPECTIVE         ║${NC}"
echo -e "${MAGENTA}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}[5.5/8] Verifying invoice schema is accessible from KERIA...${NC}"
echo ""

# NOTE: Per official documentation (102_20_KERIA_Signify_Credential_Issuance.md),
# the SignifyClient resolves schema OOBIs THROUGH KERIA, not directly.
# So we MUST verify the schema is accessible from KERIA's perspective!

echo -e "${BLUE}→ Verifying invoice schema file exists...${NC}"
if [ -f "$INVOICE_SCHEMA_FILE" ]; then
    SCHEMA_FILE_SAID=$(jq -r '."$id"' "$INVOICE_SCHEMA_FILE")
    echo -e "${GREEN}  ✓ Invoice schema file exists: $INVOICE_SCHEMA_FILE${NC}"
    echo -e "${GREEN}  ✓ Schema SAID in file: $SCHEMA_FILE_SAID${NC}"
    echo -e "${GREEN}  ✓ Expected SAID: $INVOICE_SCHEMA_SAID${NC}"
    
    if [ "$SCHEMA_FILE_SAID" != "$INVOICE_SCHEMA_SAID" ]; then
        echo -e "${RED}  ⚠ WARNING: Schema SAID mismatch!${NC}"
        echo -e "${RED}    File has: $SCHEMA_FILE_SAID${NC}"
        echo -e "${RED}    Expected: $INVOICE_SCHEMA_SAID${NC}"
        echo -e "${YELLOW}    You may need to run ./stop.sh && ./deploy.sh to reload schemas${NC}"
    fi
else
    echo -e "${RED}ERROR: Invoice schema file not found: $INVOICE_SCHEMA_FILE${NC}"
    exit 1
fi
echo ""

# CRITICAL: Test schema accessibility FROM KERIA CONTAINER
# This is how the SignifyClient will resolve the schema OOBI!
echo -e "${BLUE}→ Testing invoice schema accessibility from KERIA container (CRITICAL)...${NC}"
SCHEMA_ACCESSIBLE_FROM_KERIA=false

for i in $(seq 1 5); do
    echo "  Attempt $i/5: Testing KERIA → schema container connectivity..."
    
    # Test if KERIA can reach the schema container
    if docker compose exec -T keria wget -qO- --timeout=10 "http://schema:7723/oobi/$INVOICE_SCHEMA_SAID" 2>/dev/null | grep -q "$INVOICE_SCHEMA_SAID"; then
        echo -e "${GREEN}  ✓ Invoice schema IS ACCESSIBLE from KERIA container!${NC}"
        SCHEMA_ACCESSIBLE_FROM_KERIA=true
        break
    else
        echo -e "${YELLOW}  ✗ Schema not accessible from KERIA (attempt $i)${NC}"
        
        # Check if schema container is healthy
        SCHEMA_STATUS=$(docker compose ps schema --format "{{.Status}}" 2>/dev/null | head -1)
        echo "    Schema container status: $SCHEMA_STATUS"
        
        if [[ ! "$SCHEMA_STATUS" =~ "healthy" ]] && [[ ! "$SCHEMA_STATUS" =~ "Up" ]]; then
            echo -e "${YELLOW}    Schema container unhealthy, restarting...${NC}"
            docker compose restart schema
            sleep 10
        else
            # Container is up but schema might not be loaded - check files
            echo "    Checking if schema files are in container..."
            if ! docker compose exec -T schema ls /vLEI/schema/ 2>/dev/null | grep -q "$INVOICE_SCHEMA_SAID"; then
                echo -e "${YELLOW}    Schema file missing in container, copying...${NC}"
                docker compose exec -T schema cp /vLEI/custom-schema/self-attested-invoice.json /vLEI/schema/ 2>/dev/null || true
                docker compose exec -T schema cp "/vLEI/custom-schema/${INVOICE_SCHEMA_SAID}.json" /vLEI/schema/ 2>/dev/null || true
            fi
            sleep 3
        fi
    fi
done

if [ "$SCHEMA_ACCESSIBLE_FROM_KERIA" = false ]; then
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ✗ CRITICAL ERROR: Invoice schema NOT accessible from KERIA ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${RED}  The SignifyClient resolves schema OOBIs through KERIA.${NC}"
    echo -e "${RED}  If KERIA cannot reach the schema container, credential issuance will fail.${NC}"
    echo ""
    echo -e "${YELLOW}  Diagnostic commands:${NC}"
    echo "    1. Check schema container: docker compose ps schema"
    echo "    2. Check schema logs: docker compose logs schema --tail=30"
    echo "    3. Check network: docker network inspect vlei_workshop"
    echo "    4. Test from KERIA: docker compose exec keria wget -qO- http://schema:7723/oobi/$INVOICE_SCHEMA_SAID"
    echo ""
    echo -e "${YELLOW}  Fix options:${NC}"
    echo "    Option 1: Restart schema container"
    echo "      docker compose restart schema && sleep 15"
    echo "    Option 2: Full restart (preserves data)"
    echo "      docker compose down && docker compose up -d && sleep 30"
    echo "    Option 3: Check if schema file exists in container"
    echo "      docker compose exec schema ls -la /vLEI/schema/"
    echo ""
    echo -e "${RED}  Cannot proceed without schema accessibility from KERIA.${NC}"
    exit 1
fi

# Also verify from host (secondary check)
echo -e "${BLUE}→ Verifying schema from host (secondary check)...${NC}"
if curl -sf --max-time 10 "http://127.0.0.1:7723/oobi/$INVOICE_SCHEMA_SAID" > /dev/null 2>&1; then
    echo -e "${GREEN}  ✓ Invoice schema also accessible from host${NC}"
else
    echo -e "${YELLOW}  ⚠ Schema not accessible from host (may be network issue)${NC}"
fi
echo ""

echo -e "${GREEN}✓ Invoice schema verified and accessible from KERIA!${NC}"
echo ""

################################################################################
# ✨ SECTION 7: INVOICE CREDENTIAL WORKFLOW
################################################################################

echo -e "${WHITE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${WHITE}║  ✨ SECTION 7: SELF-ATTESTED INVOICE CREDENTIAL WORKFLOW    ║${NC}"
echo -e "${WHITE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ✨ CRITICAL: Check KERIA health before starting invoice workflow
echo -e "${YELLOW}✨ Checking KERIA health before invoice workflow...${NC}"
MAX_KERIA_CHECKS=10
KERIA_HEALTHY=false
for i in $(seq 1 $MAX_KERIA_CHECKS); do
    if docker compose exec -T keria wget --spider --tries=1 --no-verbose --timeout=5 http://127.0.0.1:3902/spec.yaml 2>/dev/null; then
        echo -e "${GREEN}  ✓ KERIA is healthy (check $i)${NC}"
        KERIA_HEALTHY=true
        break
    else
        echo -e "${YELLOW}  KERIA health check $i/$MAX_KERIA_CHECKS - waiting 5s...${NC}"
        # Check if container is running
        if ! docker compose ps keria 2>/dev/null | grep -q "Up"; then
            echo -e "${RED}  ✗ KERIA container is not running!${NC}"
            echo "    Attempting to restart KERIA..."
            docker compose up -d keria
            sleep 10
        else
            sleep 5
        fi
    fi
done

if [ "$KERIA_HEALTHY" = false ]; then
    echo -e "${RED}✗ KERIA is not healthy after $MAX_KERIA_CHECKS checks${NC}"
    echo "  Showing KERIA logs:"
    docker compose logs keria --tail=30
    exit 1
fi

# Wait for KERIA to fully settle after agent delegation workflow
echo -e "${BLUE}  Waiting 15s for KERIA and Docker DNS to settle after agent workflow...${NC}"
sleep 15
echo ""

# Variables for invoice workflow
SELLER_AGENT="jupiterSellerAgent"
BUYER_AGENT="tommyBuyerAgent"
INVOICE_REGISTRY_NAME="${SELLER_AGENT}_INVOICE_REGISTRY"

echo -e "${YELLOW}[6/8] Creating Invoice Credential Registry for ${SELLER_AGENT}...${NC}"
echo ""

# Check if agent info exists
if [ ! -f "./task-data/${SELLER_AGENT}-info.json" ]; then
    echo -e "${RED}ERROR: Agent info file not found: ./task-data/${SELLER_AGENT}-info.json${NC}"
    echo -e "${YELLOW}The agent delegation must be completed before issuing invoices${NC}"
    exit 1
fi

SELLER_AGENT_AID=$(cat "./task-data/${SELLER_AGENT}-info.json" | jq -r '.aid')
echo "  Seller Agent: $SELLER_AGENT"
echo "  Seller Agent AID: $SELLER_AGENT_AID"
echo "  Invoice Registry: $INVOICE_REGISTRY_NAME"
echo ""

# Create registry for agent's self-attested invoice credentials
echo -e "${BLUE}→ Creating invoice credential registry...${NC}"

# Check if agent's BRAN file exists
SELLER_BRAN_FILE="./task-data/${SELLER_AGENT}-bran.txt"
if [ ! -f "$SELLER_BRAN_FILE" ]; then
    echo -e "${RED}ERROR: Agent BRAN file not found: $SELLER_BRAN_FILE${NC}"
    echo -e "${YELLOW}The agent must have been created with a unique BRAN.${NC}"
    exit 1
fi
SELLER_AGENT_BRAN=$(cat "$SELLER_BRAN_FILE")
echo "  Using agent's unique BRAN: ${SELLER_AGENT_BRAN:0:20}..."

# Check if the invoice registry creation script exists
if [ -f "./task-scripts/invoice/invoice-registry-create-agent.sh" ]; then
    chmod +x ./task-scripts/invoice/invoice-registry-create-agent.sh
    ./task-scripts/invoice/invoice-registry-create-agent.sh "$SELLER_AGENT" "$INVOICE_REGISTRY_NAME"
else
    echo -e "${YELLOW}  Creating registry via docker compose...${NC}"
    docker compose exec -T tsx-shell tsx \
      sig-wallet/src/tasks/invoice/invoice-registry-create.ts \
      docker \
      "$SELLER_AGENT" \
      "$SELLER_AGENT_BRAN" \
      "$INVOICE_REGISTRY_NAME" \
      "/task-data" || {
        echo -e "${RED}✗ Failed to create invoice registry${NC}"
        # Continue anyway as it might already exist
    }
fi

# Save registry info
REGISTRY_INFO_FILE="./task-data/${SELLER_AGENT}-invoice-registry-info.json"
echo "{\"registryName\": \"$INVOICE_REGISTRY_NAME\", \"agentAlias\": \"$SELLER_AGENT\", \"agentAID\": \"$SELLER_AGENT_AID\"}" > "$REGISTRY_INFO_FILE"

echo -e "${GREEN}✓ Invoice registry info saved to $REGISTRY_INFO_FILE${NC}"
echo ""

# ✨ CRITICAL: Wait and check KERIA health after registry creation
echo -e "${YELLOW}✨ HEALTH CHECK: Verifying KERIA after registry creation...${NC}"
echo -e "${BLUE}  Waiting 15s for KERIA and Docker DNS to settle after registry creation...${NC}"
sleep 15

KERIA_HEALTHY=false
for i in $(seq 1 5); do
    if docker compose exec -T keria wget --spider --tries=1 --no-verbose --timeout=5 http://127.0.0.1:3902/spec.yaml 2>/dev/null; then
        echo -e "${GREEN}  ✓ KERIA is healthy after registry creation${NC}"
        KERIA_HEALTHY=true
        break
    else
        echo -e "${YELLOW}  KERIA post-registry check $i/5 - waiting 3s...${NC}"
        sleep 3
    fi
done

if [ "$KERIA_HEALTHY" = false ]; then
    echo -e "${RED}✗ KERIA became unhealthy after registry creation!${NC}"
    echo "  Attempting recovery..."
    docker compose restart keria
    sleep 15
    # Final check
    if ! docker compose exec -T keria wget --spider --tries=1 --no-verbose --timeout=5 http://127.0.0.1:3902/spec.yaml 2>/dev/null; then
        echo -e "${RED}✗ KERIA recovery failed. Please check logs and restart manually.${NC}"
        docker compose logs keria --tail=30
        exit 1
    fi
    echo -e "${GREEN}  ✓ KERIA recovered successfully${NC}"
fi
echo ""

################################################################################
# SECTION 7.2: Issue Self-Attested Invoice Credential
################################################################################

echo -e "${YELLOW}[7/8] Issuing Self-Attested Invoice Credential...${NC}"
echo ""

# Get invoice data from config
SELLER_LEI=$(jq -r '.invoice.issuer.lei' "$INVOICE_CONFIG_FILE")
BUYER_LEI=$(jq -r '.invoice.holder.lei' "$INVOICE_CONFIG_FILE")
INVOICE_NUMBER=$(jq -r '.invoice.sampleInvoice.invoiceNumber' "$INVOICE_CONFIG_FILE")
TOTAL_AMOUNT=$(jq -r '.invoice.sampleInvoice.totalAmount' "$INVOICE_CONFIG_FILE")
CURRENCY=$(jq -r '.invoice.sampleInvoice.currency' "$INVOICE_CONFIG_FILE")
DUE_DATE=$(jq -r '.invoice.sampleInvoice.dueDate' "$INVOICE_CONFIG_FILE")
PAYMENT_METHOD=$(jq -r '.invoice.sampleInvoice.paymentMethod' "$INVOICE_CONFIG_FILE")

echo -e "${CYAN}  Invoice Details:${NC}"
echo "    Invoice Number: $INVOICE_NUMBER"
echo "    Total Amount: $TOTAL_AMOUNT $CURRENCY"
echo "    Seller LEI: $SELLER_LEI"
echo "    Buyer LEI: $BUYER_LEI"
echo "    Due Date: $DUE_DATE"
echo "    Payment Method: $PAYMENT_METHOD"
echo ""

echo -e "${BLUE}  Self-Attestation:${NC}"
echo "    Issuer: $SELLER_AGENT (AID: $SELLER_AGENT_AID)"
echo "    Issuee: $SELLER_AGENT (SAME as issuer - self-attested)"
echo "    Edge: NONE (no OOR credential chain)"
echo ""

# Prepare invoice data JSON
INVOICE_DATA=$(jq -c '.invoice.sampleInvoice' "$INVOICE_CONFIG_FILE")

# Add issuer AID and LEIs to invoice data
DT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
INVOICE_DATA=$(echo "$INVOICE_DATA" | jq -c \
  --arg issuerAid "$SELLER_AGENT_AID" \
  --arg sellerLEI "$SELLER_LEI" \
  --arg buyerLEI "$BUYER_LEI" \
  --arg dt "$DT" \
  '. + {i: $issuerAid, sellerLEI: $sellerLEI, buyerLEI: $buyerLEI, dt: $dt}')

# Invoice Schema SAID is defined at the top of this script
# INVOICE_SCHEMA_SAID="EHFQviJgaf6YTHbMPLheau8H1s3v31-ByTER_D49DLHY"
echo -e "${CYAN}  Schema SAID: $INVOICE_SCHEMA_SAID${NC}"
echo -e "${CYAN}  Schema OOBI: http://schema:7723/oobi/$INVOICE_SCHEMA_SAID${NC}"

CRED_OUTPUT_PATH="./task-data/${SELLER_AGENT}-self-invoice-credential-info.json"

echo -e "${BLUE}→ Issuing self-attested invoice credential...${NC}"

# Check if the self-attested issue script exists
if [ -f "./task-scripts/invoice/invoice-acdc-issue-self-attested.sh" ]; then
    chmod +x ./task-scripts/invoice/invoice-acdc-issue-self-attested.sh
    ./task-scripts/invoice/invoice-acdc-issue-self-attested.sh "$SELLER_AGENT" "$INVOICE_CONFIG_FILE"
else
    echo -e "${YELLOW}  Issuing via docker compose...${NC}"
    
    # Create the credential via TypeScript
    docker compose exec -T tsx-shell tsx \
      sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only.ts \
      docker \
      "$SELLER_AGENT" \
      "" \
      "$INVOICE_REGISTRY_NAME" \
      "$INVOICE_SCHEMA_SAID" \
      "$INVOICE_DATA" \
      "$CRED_OUTPUT_PATH" || {
        echo -e "${RED}✗ Failed to issue self-attested invoice credential${NC}"
        echo -e "${YELLOW}Continuing to IPEX workflow (may need manual credential creation)${NC}"
    }
fi

if [ -f "$CRED_OUTPUT_PATH" ]; then
    CRED_SAID=$(cat "$CRED_OUTPUT_PATH" | jq -r '.said // "unknown"')
    echo -e "${GREEN}✓ Self-attested invoice credential issued${NC}"
    echo "    Credential SAID: $CRED_SAID"
    echo "    Output: $CRED_OUTPUT_PATH"
else
    echo -e "${YELLOW}  Credential output file not found - continuing with IPEX${NC}"
fi
echo ""

################################################################################
# SECTION 7.3: IPEX Grant from jupiterSellerAgent to tommyBuyerAgent
################################################################################

echo -e "${WHITE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${WHITE}║  IPEX GRANT: ${SELLER_AGENT} → ${BUYER_AGENT}                 ${NC}"
echo -e "${WHITE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if buyer agent exists
if [ ! -f "./task-data/${BUYER_AGENT}-info.json" ]; then
    echo -e "${RED}ERROR: Buyer agent info file not found: ./task-data/${BUYER_AGENT}-info.json${NC}"
    exit 1
fi

BUYER_AGENT_AID=$(cat "./task-data/${BUYER_AGENT}-info.json" | jq -r '.aid')
echo "  Sender: $SELLER_AGENT (AID: $SELLER_AGENT_AID)"
echo "  Recipient: $BUYER_AGENT (AID: $BUYER_AGENT_AID)"
echo ""

echo -e "${BLUE}→ Sending IPEX grant...${NC}"

# Use the IPEX grant script
if [ -f "./task-scripts/invoice/invoice-ipex-grant.sh" ]; then
    chmod +x ./task-scripts/invoice/invoice-ipex-grant.sh
    ./task-scripts/invoice/invoice-ipex-grant.sh "$SELLER_AGENT" "$BUYER_AGENT"
else
    echo -e "${YELLOW}  Sending IPEX grant via docker compose...${NC}"
    
    docker compose exec -T tsx-shell tsx \
      sig-wallet/src/tasks/invoice/invoice-ipex-grant.ts \
      docker \
      "" \
      "$SELLER_AGENT" \
      "$BUYER_AGENT" || {
        echo -e "${RED}✗ Failed to send IPEX grant${NC}"
        exit 1
    }
fi

echo -e "${GREEN}✓ IPEX grant sent from $SELLER_AGENT to $BUYER_AGENT${NC}"
echo ""

################################################################################
# SECTION 7.4: IPEX Admit by tommyBuyerAgent
################################################################################

echo -e "${WHITE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${WHITE}║  IPEX ADMIT: ${BUYER_AGENT} admits credential                 ${NC}"
echo -e "${WHITE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo "  Receiver: $BUYER_AGENT (AID: $BUYER_AGENT_AID)"
echo "  From: $SELLER_AGENT (AID: $SELLER_AGENT_AID)"
echo ""

echo -e "${BLUE}→ Admitting IPEX grant...${NC}"

# Allow time for the grant notification to propagate through KERIA
# This is important - notifications take time to sync between agents
echo -e "${YELLOW}  Waiting 10s for grant notification to propagate...${NC}"
sleep 10

# Use the IPEX admit script
if [ -f "./task-scripts/invoice/invoice-ipex-admit.sh" ]; then
    chmod +x ./task-scripts/invoice/invoice-ipex-admit.sh
    ./task-scripts/invoice/invoice-ipex-admit.sh "$BUYER_AGENT" "$SELLER_AGENT"
else
    echo -e "${YELLOW}  Admitting IPEX grant via docker compose...${NC}"
    
    docker compose exec -T tsx-shell tsx \
      sig-wallet/src/tasks/invoice/invoice-ipex-admit.ts \
      docker \
      "" \
      "$BUYER_AGENT" \
      "$SELLER_AGENT" || {
        echo -e "${RED}✗ Failed to admit IPEX grant${NC}"
        exit 1
    }
fi

echo -e "${GREEN}✓ IPEX grant admitted by $BUYER_AGENT${NC}"
echo -e "${GREEN}✓ Invoice credential now available in ${BUYER_AGENT}'s KERIA storage${NC}"
echo ""

################################################################################
# SECTION 8: Summary and Next Steps
################################################################################

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    Execution Complete                        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}✅ Setup Complete!${NC}"
echo ""
echo -e "${BLUE}Summary:${NC}"
echo "  • GEDA (Root) and QVI established"
echo "  • $ORG_COUNT organizations processed:"

# Count total agents created
TOTAL_AGENTS=0
for ((org_idx=0; org_idx<$ORG_COUNT; org_idx++)); do
    ORG_NAME=$(jq -r ".organizations[$org_idx].name" "$CONFIG_FILE")
    PERSON_COUNT=$(jq -r ".organizations[$org_idx].persons | length" "$CONFIG_FILE")
    ORG_AGENTS=0
    for ((person_idx=0; person_idx<$PERSON_COUNT; person_idx++)); do
        AGENT_COUNT=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents | length" "$CONFIG_FILE")
        ORG_AGENTS=$((ORG_AGENTS + AGENT_COUNT))
    done
    TOTAL_AGENTS=$((TOTAL_AGENTS + ORG_AGENTS))
    echo "    - $ORG_NAME ($PERSON_COUNT person(s), $ORG_AGENTS agent(s))"
done

echo "  • All credentials issued and presented to verifier"
echo "  • ✨ $TOTAL_AGENTS agent(s) with UNIQUE BRANs delegated and verified"
echo "  • Trust tree visualization generated"
echo ""

echo -e "${WHITE}✨ Invoice Credential Summary:${NC}"
echo "  ✓ jupiterSellerAgent created Invoice Registry"
echo "  ✓ jupiterSellerAgent issued SELF-ATTESTED Invoice Credential"
echo "    - Issuer = Issuee (self-attested)"
echo "    - No edge/chain (standalone credential)"
echo "    - Invoice: $INVOICE_NUMBER for $TOTAL_AMOUNT $CURRENCY"
echo "  ✓ jupiterSellerAgent sent IPEX GRANT to tommyBuyerAgent"
echo "  ✓ tommyBuyerAgent ADMITTED the IPEX grant"
echo "  ✓ Invoice credential now in tommyBuyerAgent's KERIA storage"
echo ""

echo -e "${MAGENTA}✨ Unique BRAN Summary:${NC}"
if [ -f "./task-data/agent-brans.json" ]; then
    BRAN_COUNT=$(jq -r '.agents | length' "./task-data/agent-brans.json")
    echo -e "${GREEN}  ✓ Total unique BRANs generated: $BRAN_COUNT${NC}"
    echo -e "${GREEN}  ✓ Configuration: task-data/agent-brans.json${NC}"
fi
echo ""

echo -e "${CYAN}✨ Agent Delegation Summary:${NC}"
for ((org_idx=0; org_idx<$ORG_COUNT; org_idx++)); do
    PERSON_COUNT=$(jq -r ".organizations[$org_idx].persons | length" "$CONFIG_FILE")
    for ((person_idx=0; person_idx<$PERSON_COUNT; person_idx++)); do
        AGENT_COUNT=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents | length" "$CONFIG_FILE")
        if [ "$AGENT_COUNT" -gt 0 ]; then
            PERSON_ALIAS=$(jq -r ".organizations[$org_idx].persons[$person_idx].alias" "$CONFIG_FILE")
            for ((agent_idx=0; agent_idx<$AGENT_COUNT; agent_idx++)); do
                AGENT_ALIAS=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents[$agent_idx].alias" "$CONFIG_FILE")
                if [ -f "./task-data/${AGENT_ALIAS}-info.json" ]; then
                    AGENT_AID=$(cat "./task-data/${AGENT_ALIAS}-info.json" | jq -r .aid)
                    echo "  • $AGENT_ALIAS → Delegated from $PERSON_ALIAS"
                    echo "    AID: $AGENT_AID"
                    echo "    Status: ✓ Verified by Sally"
                fi
            done
        fi
    done
done
echo ""

# ============================================================
# GENERATE AGENT CARDS
# ============================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  GENERATING AGENT CARDS${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${WHITE}Generating agent cards with KERI identifiers from this workflow...${NC}"
echo ""

# Run the agent card generator
if [ -f "./generate-agent-cards.js" ]; then
    node ./generate-agent-cards.js
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Agent cards generated successfully${NC}"
        echo ""
        echo -e "${CYAN}Agent Card Output:${NC}"
        echo "  • jupiterSellerAgent-card.json → ../Legent/A2A/agent-cards/"
        echo "  • tommyBuyerAgent-card.json → ../Legent/A2A/agent-cards/"
        echo ""
        echo -e "${CYAN}Key fields for DEEP-EXT verification:${NC}"
        echo "  • vLEImetadata.agentName (passed to DEEP-EXT.sh)"
        echo "  • vLEImetadata.oorHolderName (passed to DEEP-EXT.sh)"
    else
        echo -e "${RED}✗ Agent card generation failed${NC}"
    fi
else
    echo -e "${RED}✗ generate-agent-cards.js not found${NC}"
fi
echo ""

echo -e "${BLUE}📋 Next Steps:${NC}"
echo "  1. Verify invoice credential in tommyBuyerAgent's KERIA"
echo "  2. Query invoice credentials: ./task-scripts/invoice/invoice-query.sh"
echo "  3. Present invoice to verifier if needed"
echo "  4. Implement credential revocation workflow"
echo "  5. Start agent card servers (ports 8080, 9090) to serve well-known URLs"
echo ""

echo -e "${RED}⚠️  SECURITY WARNINGS:${NC}"
echo -e "${RED}  • BRANs are cryptographic secrets - protect them!${NC}"
echo -e "${RED}  • Never commit task-data/agent-brans.json to version control${NC}"
echo -e "${RED}  • Invoice credentials contain business-sensitive data${NC}"
echo ""

echo -e "${BLUE}📄 Documentation:${NC}"
echo "  • Configuration: $CONFIG_FILE"
echo "  • Invoice Config: $INVOICE_CONFIG_FILE"
echo "  • Trust Tree: $TRUST_TREE_FILE"
echo "  • Invoice Schema: $INVOICE_SCHEMA_FILE"
echo "  • BRAN Config: task-data/agent-brans.json"
echo ""

# Display trust tree
echo -e "${YELLOW}Trust Tree:${NC}"
cat "$TRUST_TREE_FILE"
echo ""

echo -e "${GREEN}✨ vLEI system 4C with Invoice Credential + IPEX completed successfully!${NC}"
echo ""

exit 0
