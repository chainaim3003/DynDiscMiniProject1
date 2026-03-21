#!/bin/bash
################################################################################
# run-all-buyerseller-2.sh - Configuration-Driven vLEI Credential System
#
# Purpose: Orchestrate the complete vLEI credential issuance flow for multiple
#          organizations (buyer and seller) using configuration from JSON file
#
# Design: Based on understanding-3.md and complete-session.md
# Reference: https://github.com/GLEIF-IT/vlei-hackathon-2025-workshop
#
# Configuration: appconfig/configBuyerSellerAIAgent1.json
#
# Flow:
#   1. GEDA & QVI Setup (once)
#   2. Loop through each organization:
#      - Create LE AID and credentials
#      - Loop through each person:
#        - Create Person AID
#        - Issue OOR credentials
#        - Present to verifier
#        - ✨ NEW: Create and delegate agents
#   3. Generate trust tree visualization
#
# Date: November 11, 2025
# Updated: Added agent delegation workflow
################################################################################

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration file location
CONFIG_FILE="./appconfig/configBuyerSellerAIAgent1.json"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  vLEI Configuration-Driven System${NC}"
echo -e "${BLUE}  Buyer-Seller Credential Issuance${NC}"
echo -e "${BLUE}  ✨ WITH AGENT DELEGATION${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

################################################################################
# SECTION 1: Configuration Validation
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
# SECTION 2: GEDA & QVI Setup (One-time initialization)
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
# SECTION 3: Organization Loop - Process Each Legal Entity
################################################################################

echo -e "${YELLOW}[3/5] Processing Organizations...${NC}"
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
    
    echo -e "${YELLOW}  [4/5] Processing Persons for $ORG_NAME...${NC}"
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
        # ✨ NEW SECTION: Agent Delegation Workflow
        ##########################################################################
        
        # Check for delegated agents
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
                
                # Step 1: Agent initiates delegation request
                echo -e "${BLUE}          [1/5] Creating agent delegation request...${NC}"
                ./task-scripts/person/person-delegate-agent-create.sh "$PERSON_ALIAS" "$AGENT_ALIAS"
                
                # Step 2: OOR Holder approves delegation
                echo -e "${BLUE}          [2/5] OOR Holder approves delegation...${NC}"
                ./task-scripts/person/person-approve-agent-delegation.sh "$PERSON_ALIAS" "$AGENT_ALIAS"
                
                # Step 3: Agent completes delegation
                echo -e "${BLUE}          [3/5] Agent completes delegation...${NC}"
                ./task-scripts/agent/agent-aid-delegate-finish.sh "$AGENT_ALIAS" "$PERSON_ALIAS"
                


                # Step 4: Agent resolves OOBIs
                echo -e "${BLUE}          [4/5] Agent resolves OOBIs...${NC}"
                echo -e "${BLUE}            → Resolving QVI OOBI...${NC}"
                ./task-scripts/agent/agent-oobi-resolve-qvi.sh "$AGENT_ALIAS"
                
                echo -e "${BLUE}            → Resolving LE OOBI...${NC}"
                ./task-scripts/agent/agent-oobi-resolve-le.sh "$AGENT_ALIAS" "$ORG_ALIAS"
                
                echo -e "${BLUE}            → Resolving Sally verifier OOBI...${NC}"
                ./task-scripts/agent/agent-oobi-resolve-verifier.sh "$AGENT_ALIAS"
                
                # Step 5: Verify agent delegation
                echo -e "${BLUE}          [5/5] Verifying agent delegation via Sally...${NC}"
                ./task-scripts/agent/agent-verify-delegation.sh "$AGENT_ALIAS" "$PERSON_ALIAS"
                
                echo -e "${GREEN}          ✓ Agent $AGENT_ALIAS delegation complete and verified${NC}"
                
                # Display agent info
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
# SECTION 5: Generate Trust Tree Visualization
################################################################################

echo -e "${YELLOW}[5/5] Generating Trust Tree Visualization...${NC}"
echo ""

# Create trust tree output
TRUST_TREE_FILE="./task-data/trust-tree-buyerseller.txt"

cat > "$TRUST_TREE_FILE" << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                    vLEI Trust Chain - Buyer-Seller Scenario                 ║
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
│   │         ├─ OOR_AUTH Credential (issued by LE to QVI)
│   │         │   └─ Admitted by QVI ✓
│   │         │
│   │         ├─ OOR Credential (issued by QVI to Person)
│   │         │   ├─ Chained to: LE Credential
│   │         │   └─ Presented to Sally Verifier ✓
│   │         │
│   │         └─ ✨ Delegated Agent: jupiterSellerAgent (AI Agent)
│   │             ├─ Agent AID (Delegated from OOR Holder)
│   │             ├─ KEL Seal (Anchored in OOR Holder's KEL)
│   │             ├─ OOBI Resolved (QVI, LE, Sally)
│   │             └─ ✓ Verified by Sally Verifier
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
│                 ├─ Agent AID (Delegated from OOR Holder)
│                 ├─ KEL Seal (Anchored in OOR Holder's KEL)
│                 ├─ OOBI Resolved (QVI, LE, Sally)
│                 └─ ✓ Verified by Sally Verifier

╔══════════════════════════════════════════════════════════════════════════════╗
║                              Credential Flow Summary                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. GEDA (ROOT) → QVI
   └─ Issues QVI Credential
   
2. QVI → Legal Entities (Jupiter & Tommy)
   └─ Issues LE Credentials (with LEI)
   
3. Legal Entities → QVI (OOR_AUTH)
   └─ Authorizes QVI to issue OOR credential to specific person
   
4. QVI → Persons (Chief Sales Officer, Chief Procurement Officer)
   └─ Issues OOR Credentials (chained to LE Credential)
   
5. Persons → Sally Verifier
   └─ Presents OOR Credentials for verification

6. ✨ NEW: OOR Holders → Agents (Delegation)
   ├─ Agent creates delegation request
   ├─ OOR Holder approves with KEL seal
   ├─ Agent completes delegation
   ├─ Agent resolves OOBIs (QVI, LE, Sally)
   └─ Sally verifies complete delegation chain

╔══════════════════════════════════════════════════════════════════════════════╗
║                            Verification Points                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Sally Verifier validates:
  ✓ Digital signatures (KERI)
  ✓ Credential chain integrity
  ✓ Root of trust (GEDA)
  ✓ Delegation relationships
  ✓ Schema compliance
  ✓ Edge validity
  ✓ Revocation status
  ✓ OOBI resolution
  ✨ Agent delegation chain
  ✨ OOR Holder KEL seal
  ✨ Complete credential chain for agents

╔══════════════════════════════════════════════════════════════════════════════╗
║                                    Notes                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Configuration Source: appconfig/configBuyerSellerAIAgent1.json

✅ Features Implemented:
  ✓ Configuration-driven flow
  ✓ LEI values from configuration
  ✓ Person names and roles from configuration
  ✓ ✨ Agent delegation workflow
  ✓ ✨ Sally verification for agents

Recommended Enhancements:
  → Add credential revocation and rotation features
  → Add multi-agent support per person
  → Add agent capability scopes

Standards Compliance:
  ✓ GLEIF vLEI Ecosystem Governance Framework
  ✓ KERI (Key Event Receipt Infrastructure)
  ✓ ACDC (Authentic Chained Data Containers)
  ✓ CESR (Composable Event Streaming Representation)
  ✓ OOBI (Out-Of-Band Introduction)
  ✓ ✨ Agent Delegation via KEL

Reference: https://github.com/GLEIF-IT/vlei-hackathon-2025-workshop

EOF

echo -e "${GREEN}✓ Trust tree visualization created: $TRUST_TREE_FILE${NC}"
echo ""

# Display the trust tree
cat "$TRUST_TREE_FILE"
echo ""

################################################################################
# SECTION 6: Summary and Next Steps
################################################################################

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Execution Complete                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
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
echo "  • ✨ $TOTAL_AGENTS agent(s) delegated and verified"
echo "  • Trust tree visualization generated"
echo ""

echo -e "${GREEN}✓ Configuration Integration:${NC}"
echo "  1. LEI values sourced from configuration file"
echo "  2. Person names and roles sourced from configuration file"
echo "  3. Agent data sourced from configuration file"
echo "  4. ✨ Complete agent delegation workflow implemented"
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

echo -e "${BLUE}📋 Next Steps:${NC}"
echo "  1. Agents can now act on behalf of OOR holders"
echo "  2. Test agent delegation verification independently"
echo "  3. Add agent capability scopes if needed"
echo "  4. Implement credential revocation and rotation"
echo "  5. Enhance error handling and validation"
echo ""

echo -e "${BLUE}📄 Documentation:${NC}"
echo "  • Configuration: $CONFIG_FILE"
echo "  • Trust Tree: $TRUST_TREE_FILE"
echo "  • Agent Guide: AGENT-DELEGATION-QUICK-START.md"
echo "  • Complete Docs: AGENT-DELEGATION-IMPLEMENTATION-COMPLETE.md"
echo ""

echo -e "${GREEN}✨ vLEI credential system with agent delegation completed successfully!${NC}"
echo ""

exit 0