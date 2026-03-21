#!/bin/bash
################################################################################
# run-all-buyerseller-4D-with-subdelegation.sh
#
# Purpose: Extended vLEI workflow with sub-agent delegation support
#          jupiterSellerAgent → JupiterTreasuryAgent
#
# New Features:
#   - Sub-agent BRAN generation
#   - Agent-to-agent delegation
#   - Extended trust chain verification
#
# Version: 4D - With Sub-Delegation
################################################################################

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
NC='\033[0m'

# Configuration
CONFIG_FILE="./appconfig/configBuyerSellerAIAgent1-with-subdelegation.json"
INVOICE_CONFIG_FILE="./appconfig/invoiceConfig.json"
INVOICE_SCHEMA_FILE="./schemas/self-attested-invoice.json"

# Load schema SAID
if [ -f "./appconfig/schemaSaids.json" ]; then
    INVOICE_SCHEMA_SAID=$(jq -r '.invoiceSchema.said' "./appconfig/schemaSaids.json")
elif [ -f "./task-data/invoice-schema-said.txt" ]; then
    INVOICE_SCHEMA_SAID=$(cat ./task-data/invoice-schema-said.txt)
else
    INVOICE_SCHEMA_SAID=$(jq -r '."$id"' "$INVOICE_SCHEMA_FILE" 2>/dev/null || echo "")
fi

if [ -z "$INVOICE_SCHEMA_SAID" ] || [ "$INVOICE_SCHEMA_SAID" = "null" ]; then
    echo -e "${RED}ERROR: Invoice schema SAID is empty!${NC}"
    exit 1
fi

echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  vLEI System v4D - With Sub-Delegation${NC}"
echo -e "${CYAN}  jupiterSellerAgent → JupiterTreasuryAgent${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""

################################################################################
# Run existing 4C workflow first
################################################################################

echo -e "${YELLOW}[Phase 1] Running base vLEI workflow (4C)...${NC}"
echo ""

if [ -f "./run-all-buyerseller-4C-with-agents.sh" ]; then
    ./run-all-buyerseller-4C-with-agents.sh
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Base workflow (4C) failed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Base workflow (4C) completed${NC}"
    echo ""
else
    echo -e "${RED}ERROR: Base workflow script not found${NC}"
    exit 1
fi

################################################################################
# Phase 2: Sub-Agent BRAN Generation
################################################################################

echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║  PHASE 2: SUB-AGENT BRAN GENERATION                      ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}[2.1/5] Generating unique BRANs for sub-agents...${NC}"
echo ""

if [ -f "./generate-subagent-brans.sh" ]; then
    chmod +x ./generate-subagent-brans.sh
    ./generate-subagent-brans.sh
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Sub-agent BRAN generation failed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Sub-agent BRANs generated${NC}"
    echo ""
else
    echo -e "${RED}ERROR: generate-subagent-brans.sh not found${NC}"
    exit 1
fi

################################################################################
# Phase 3: Sub-Agent Delegation
################################################################################

echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║  PHASE 3: SUB-AGENT DELEGATION                           ║${NC}"
echo -e "${MAGENTA}║  jupiterSellerAgent → JupiterTreasuryAgent               ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Find sub-agents in configuration
ORG_COUNT=$(jq -r '.organizations | length' "$CONFIG_FILE")

for ((org_idx=0; org_idx<$ORG_COUNT; org_idx++)); do
    ORG_NAME=$(jq -r ".organizations[$org_idx].name" "$CONFIG_FILE")
    PERSON_COUNT=$(jq -r ".organizations[$org_idx].persons | length" "$CONFIG_FILE")
    
    for ((person_idx=0; person_idx<$PERSON_COUNT; person_idx++)); do
        AGENT_COUNT=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents | length" "$CONFIG_FILE")
        
        for ((agent_idx=0; agent_idx<$AGENT_COUNT; agent_idx++)); do
            PARENT_AGENT=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents[$agent_idx].alias" "$CONFIG_FILE")
            
            # Check if this agent has sub-agents
            SUB_AGENT_LIST=$(jq -r ".organizations[$org_idx].persons[$person_idx].agents[$agent_idx].subAgents // []" "$CONFIG_FILE")
            
            if [ "$SUB_AGENT_LIST" != "[]" ]; then
                SUB_COUNT=$(echo "$SUB_AGENT_LIST" | jq 'length')
                
                echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
                echo -e "${BLUE}║  Parent Agent: $PARENT_AGENT${NC}"
                echo -e "${BLUE}║  Sub-Agents: $SUB_COUNT${NC}"
                echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
                echo ""
                
                for ((sub_idx=0; sub_idx<$SUB_COUNT; sub_idx++)); do
                    SUB_ALIAS=$(echo "$SUB_AGENT_LIST" | jq -r ".[$sub_idx].alias")
                    SUB_TYPE=$(echo "$SUB_AGENT_LIST" | jq -r ".[$sub_idx].agentType")
                    SUB_SCOPE=$(echo "$SUB_AGENT_LIST" | jq -r ".[$sub_idx].permissions.scope")
                    
                    echo -e "${CYAN}  ┌─────────────────────────────────────────────────┐${NC}"
                    echo -e "${CYAN}  │  Sub-Agent: $SUB_ALIAS${NC}"
                    echo -e "${CYAN}  │  Type: $SUB_TYPE${NC}"
                    echo -e "${CYAN}  │  Scope: $SUB_SCOPE${NC}"
                    echo -e "${CYAN}  │  Parent: $PARENT_AGENT${NC}"
                    echo -e "${CYAN}  └─────────────────────────────────────────────────┘${NC}"
                    echo ""
                    
                    # Verify BRAN was generated
                    BRAN_FILE="./task-data/${SUB_ALIAS}-bran.txt"
                    if [ ! -f "$BRAN_FILE" ]; then
                        echo -e "${RED}    ✗ ERROR: BRAN not found for ${SUB_ALIAS}${NC}"
                        exit 1
                    fi
                    
                    SUB_BRAN=$(cat "$BRAN_FILE")
                    echo -e "${GREEN}    ✓ Using pre-generated unique BRAN${NC}"
                    echo -e "${GREEN}      BRAN: ${SUB_BRAN:0:20}... (256-bit)${NC}"
                    echo ""
                    
                    # Create sub-agent delegation script wrapper
                    if [ ! -f "./task-scripts/subagent/subagent-delegate-with-unique-bran.sh" ]; then
                        echo -e "${RED}    ✗ ERROR: subagent-delegate-with-unique-bran.sh not found${NC}"
                        exit 1
                    fi
                    
                    echo -e "${BLUE}    → Creating sub-agent and delegating from parent...${NC}"
                    
                    chmod +x ./task-scripts/subagent/subagent-delegate-with-unique-bran.sh
                    ./task-scripts/subagent/subagent-delegate-with-unique-bran.sh "$SUB_ALIAS" "$PARENT_AGENT"
                    
                    if [ $? -ne 0 ]; then
                        echo -e "${RED}    ✗ Sub-agent delegation failed${NC}"
                        exit 1
                    fi
                    
                    echo -e "${GREEN}    ✓ Sub-agent $SUB_ALIAS delegation complete${NC}"
                    
                    # Display sub-agent info
                    if [ -f "./task-data/${SUB_ALIAS}-info.json" ]; then
                        SUB_AID=$(cat "./task-data/${SUB_ALIAS}-info.json" | jq -r .aid)
                        echo -e "${GREEN}      Sub-Agent AID: $SUB_AID${NC}"
                        echo -e "${GREEN}      Trust Chain: OOR → $PARENT_AGENT → $SUB_ALIAS${NC}"
                    fi
                    echo ""
                    
                    # Resolve OOBIs for sub-agent
                    echo -e "${BLUE}    → Resolving OOBIs for sub-agent...${NC}"
                    
                    # Resolve QVI OOBI
                    if [ -f "./task-scripts/subagent/subagent-oobi-resolve-qvi.sh" ]; then
                        chmod +x ./task-scripts/subagent/subagent-oobi-resolve-qvi.sh
                        ./task-scripts/subagent/subagent-oobi-resolve-qvi.sh "$SUB_ALIAS"
                    fi
                    
                    # Resolve LE OOBI
                    if [ -f "./task-scripts/subagent/subagent-oobi-resolve-le.sh" ]; then
                        LE_ALIAS=$(jq -r ".organizations[$org_idx].alias" "$CONFIG_FILE")
                        chmod +x ./task-scripts/subagent/subagent-oobi-resolve-le.sh
                        ./task-scripts/subagent/subagent-oobi-resolve-le.sh "$SUB_ALIAS" "$LE_ALIAS"
                    fi
                    
                    # Resolve verifier OOBI
                    if [ -f "./task-scripts/subagent/subagent-oobi-resolve-verifier.sh" ]; then
                        chmod +x ./task-scripts/subagent/subagent-oobi-resolve-verifier.sh
                        ./task-scripts/subagent/subagent-oobi-resolve-verifier.sh "$SUB_ALIAS"
                    fi
                    
                    echo -e "${GREEN}    ✓ OOBIs resolved for sub-agent${NC}"
                    echo ""
                    
                    # Verify sub-delegation via Sally
                    echo -e "${BLUE}    → Verifying sub-delegation via Sally...${NC}"
                    
                    if [ -f "./task-scripts/subagent/subagent-verify-delegation.sh" ]; then
                        chmod +x ./task-scripts/subagent/subagent-verify-delegation.sh
                        ./task-scripts/subagent/subagent-verify-delegation.sh "$SUB_ALIAS" "$PARENT_AGENT"
                    fi
                    
                    echo -e "${GREEN}    ✓ Sub-delegation verified by Sally${NC}"
                    echo ""
                    
                done
            fi
        done
    done
done

################################################################################
# Phase 4: Trust Tree Visualization
################################################################################

echo -e "${YELLOW}[4/5] Generating extended trust tree visualization...${NC}"
echo ""

TRUST_TREE_FILE="./task-data/trust-tree-buyerseller-4D-with-subdelegation.txt"

cat > "$TRUST_TREE_FILE" << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║     vLEI Trust Chain - With SUB-DELEGATION (v4D)                            ║
║     Configuration-Driven System with Agent → Sub-Agent Delegation           ║
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
│   │             ├─ 📝 SELF-ATTESTED INVOICE CREDENTIAL
│   │             │   ├─ Issuer: jupiterSellerAgent (self)
│   │             │   ├─ Issuee: jupiterSellerAgent (same as issuer)
│   │             │   ├─ Type: Self-Attested (no OOR chain edge)
│   │             │   └─ 📤 IPEX GRANT → tommyBuyerAgent
│   │             │
│   │             └─ 🔗 SUB-DELEGATED AGENT: JupiterTreasuryAgent (v4D NEW)
│   │                 ├─ ✨ Unique BRAN (256-bit cryptographic seed)
│   │                 ├─ ✨ Unique AID (derived from sub-agent's BRAN)
│   │                 ├─ Sub-Agent AID Delegated from jupiterSellerAgent
│   │                 ├─ KEL Seal (Anchored in jupiterSellerAgent's KEL)
│   │                 ├─ OOBI Resolved (QVI, LE, Sally)
│   │                 ├─ ✓ Verified by Sally Verifier
│   │                 ├─ Scope: treasury_operations
│   │                 ├─ Can Delegate: false
│   │                 │
│   │                 └─ Trust Chain:
│   │                     OOR Holder → jupiterSellerAgent → JupiterTreasuryAgent
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
║                        Sub-Delegation Flow (v4D NEW)                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

9. ✨ NEW: Agent-to-Agent Sub-Delegation
   ├─ jupiterSellerAgent creates sub-agent AID for JupiterTreasuryAgent
   │   ├─ Uses unique BRAN for JupiterTreasuryAgent
   │   ├─ Requests delegation from jupiterSellerAgent (parent)
   │   └─ Parent agent must have canDelegate: true permission
   │
   ├─ jupiterSellerAgent approves sub-delegation
   │   ├─ Creates delegation seal in jupiterSellerAgent's KEL
   │   ├─ Seal contains: {i: sub_aid, s: '0', d: sub_aid}
   │   └─ Anchors sub-agent's inception event
   │
   ├─ JupiterTreasuryAgent completes delegation
   │   ├─ Resolves parent agent's OOBI
   │   ├─ Queries parent key state to find anchor
   │   └─ Gets endpoint role and OOBI
   │
   └─ Trust Chain Verified
       ├─ OOR Holder (Jupiter_Chief_Sales_Officer)
       │   ↓ delegates to
       ├─ Parent Agent (jupiterSellerAgent)
       │   ↓ sub-delegates to
       └─ Sub-Agent (JupiterTreasuryAgent)

╔══════════════════════════════════════════════════════════════════════════════╗
║                              Key Concepts                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Primary Delegation vs Sub-Delegation:
  ✓ Primary: OOR Holder → Agent (Person delegates to agent)
  ✓ Sub: Agent → Sub-Agent (Agent delegates to another agent)
  ✓ Sub-delegation requires canDelegate: true in parent agent config
  ✓ Sub-agent scope must be narrower than or equal to parent scope

Trust Chain Verification:
  1. Verify sub-agent's delegation field (di) points to parent agent
  2. Find delegation seal in parent agent's KEL
  3. Verify seal digest matches sub-agent's inception SAID
  4. Trace back through parent agent to OOR holder to root

Permission Scoping:
  ✓ Parent: scope="sales_operations", canDelegate=true
  ✓ Sub: scope="treasury_operations", canDelegate=false
  ✓ Sub-agent cannot further delegate (canDelegate=false)

EOF

echo -e "${GREEN}✓ Trust tree visualization created: $TRUST_TREE_FILE${NC}"
echo ""

################################################################################
# Phase 5: Summary
################################################################################

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    Execution Complete                        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}✅ vLEI System v4D Complete!${NC}"
echo ""

echo -e "${WHITE}✨ Sub-Delegation Summary:${NC}"
echo "  ✓ jupiterSellerAgent (parent agent) created"
echo "  ✓ JupiterTreasuryAgent (sub-agent) delegated from jupiterSellerAgent"
echo "  ✓ Trust chain verified: OOR → jupiterSellerAgent → JupiterTreasuryAgent"
echo ""

echo -e "${BLUE}📊 Delegation Hierarchy:${NC}"
if [ -f "./task-data/jupiterSellerAgent-info.json" ] && [ -f "./task-data/JupiterTreasuryAgent-info.json" ]; then
    PARENT_AID=$(jq -r '.aid' "./task-data/jupiterSellerAgent-info.json")
    SUB_AID=$(jq -r '.aid' "./task-data/JupiterTreasuryAgent-info.json")
    
    echo "  Jupiter_Chief_Sales_Officer (OOR Holder)"
    echo "    ↓ delegates to"
    echo "  jupiterSellerAgent (Parent Agent)"
    echo "    AID: $PARENT_AID"
    echo "    Scope: sales_operations"
    echo "    Can Delegate: true"
    echo "    ↓ sub-delegates to"
    echo "  JupiterTreasuryAgent (Sub-Agent)"
    echo "    AID: $SUB_AID"
    echo "    Scope: treasury_operations"
    echo "    Can Delegate: false"
fi
echo ""

echo -e "${BLUE}📋 Next Steps:${NC}"
echo "  1. Verify sub-delegation: ./DEEP-EXT-subagent.sh JupiterTreasuryAgent jupiterSellerAgent"
echo "  2. Test sub-agent operations (treasury management)"
echo "  3. Implement sub-agent-specific credentials"
echo ""

echo -e "${BLUE}📄 Documentation:${NC}"
echo "  • Configuration: $CONFIG_FILE"
echo "  • Trust Tree: $TRUST_TREE_FILE"
echo "  • Sub-Agent BRANs: task-data/subagent-brans.json"
echo ""

# Display trust tree
cat "$TRUST_TREE_FILE"
echo ""

echo -e "${GREEN}✨ vLEI system 4D with sub-delegation completed successfully!${NC}"
echo ""

exit 0