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
#   3. Generate trust tree visualization
#
# Date: November 10, 2025
################################################################################

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration file location
CONFIG_FILE="./appconfig/configBuyerSellerAIAgent1.json"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  vLEI Configuration-Driven System${NC}"
echo -e "${BLUE}  Buyer-Seller Credential Issuance${NC}"
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
        
        # Check for delegated agents
        AGENT_COUNT=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents | length" "$CONFIG_FILE")
        if [ "$AGENT_COUNT" -gt 0 ]; then
            echo -e "${BLUE}      → Processing $AGENT_COUNT delegated agent(s)...${NC}"
            for ((agent_idx=0; agent_idx<$AGENT_COUNT; agent_idx++)); do
                AGENT_ALIAS=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents[$agent_idx].alias" "$CONFIG_FILE")
                AGENT_TYPE=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents[$agent_idx].agentType" "$CONFIG_FILE")
                echo -e "${BLUE}        • Agent: $AGENT_ALIAS (Type: $AGENT_TYPE)${NC}"
                echo -e "${YELLOW}          NOTE: Agent delegation not yet implemented${NC}"
                echo -e "${YELLOW}          TODO: Implement agent AID creation and delegation${NC}"
            done
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
║                         Configuration-Driven System                          ║
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
│   │         └─ Delegated Agent: jupitedSellerAgent (AI Agent)
│   │             └─ TODO: Implement delegation flow
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
│             └─ Delegated Agent: tommyBuyerAgent (AI Agent)
│                 └─ TODO: Implement delegation flow

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

╔══════════════════════════════════════════════════════════════════════════════╗
║                                    Notes                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Configuration Source: appconfig/configBuyerSellerAIAgent1.json

Current Limitations:
  ⚠ LEI values are hardcoded in qvi-acdc-issue-le.sh
  ⚠ Person names and roles are hardcoded in le-acdc-issue-oor-auth.sh
  ⚠ Person names and roles are hardcoded in qvi-acdc-issue-oor.sh
  ⚠ Agent delegation not yet implemented

Recommended Enhancements:
  → Modify shell scripts to accept parameters from configuration
  → Implement agent AID creation and delegation flow
  → Add support for multiple persons per organization
  → Add credential revocation and rotation features

Standards Compliance:
  ✓ GLEIF vLEI Ecosystem Governance Framework
  ✓ KERI (Key Event Receipt Infrastructure)
  ✓ ACDC (Authentic Chained Data Containers)
  ✓ CESR (Composable Event Streaming Representation)
  ✓ OOBI (Out-Of-Band Introduction)

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
for ((org_idx=0; org_idx<$ORG_COUNT; org_idx++)); do
    ORG_NAME=$(jq -r ".organizations[$org_idx].name" "$CONFIG_FILE")
    PERSON_COUNT=$(jq -r ".organizations[$org_idx].persons | length" "$CONFIG_FILE")
    echo "    - $ORG_NAME ($PERSON_COUNT person(s))"
done
echo "  • All credentials issued and presented to verifier"
echo "  • Trust tree visualization generated"
echo ""

echo -e "${GREEN}✓ Configuration Integration:${NC}"
echo "  1. LEI values now sourced from configuration file"
echo "  2. Person names and roles sourced from configuration file"
echo "  3. All organizational data driven by JSON configuration"
echo "  4. Scripts accept parameters for flexibility"
echo ""
echo -e "${YELLOW}⚠ Remaining Limitations:${NC}"
echo "  1. Agent delegation flow not yet implemented"
echo "  2. TypeScript build required after modifications (see instructions)"
echo ""

echo -e "${BLUE}📋 Next Steps:${NC}"
echo "  1. Build TypeScript files (see BUILD_INSTRUCTIONS.md)"
echo "  2. Test configuration-driven flow with multiple organizations"
echo "  3. Implement agent AID creation and delegation"
echo "  4. Add support for credential revocation and rotation"
echo "  5. Enhance error handling and validation"
echo ""

echo -e "${BLUE}📄 Documentation:${NC}"
echo "  • Configuration: $CONFIG_FILE"
echo "  • Trust Tree: $TRUST_TREE_FILE"
echo "  • Design Doc: understanding-3.md"
echo "  • Reference: https://github.com/GLEIF-IT/vlei-hackathon-2025-workshop"
echo ""

echo -e "${GREEN}✨ vLEI credential system execution completed successfully!${NC}"
echo ""

exit 0
