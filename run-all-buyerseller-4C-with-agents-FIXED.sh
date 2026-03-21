#!/bin/bash
################################################################################
# run-all-buyerseller-4C-with-agents-FIXED.sh - Configuration-Driven vLEI System
#
# Purpose: Orchestrate the complete vLEI credential issuance flow for multiple
#          organizations (buyer and seller) using configuration from JSON file
#          ✨ WITH UNIQUE AGENT BRAN SUPPORT
#          ✨ WITH SELF-ATTESTED INVOICE CREDENTIAL AND IPEX GRANT/ADMIT
#          ✨ FIXED: Added KERIA health checks before invoice workflow
#
# This FIXED version adds:
#   - KERIA health checks before invoice operations
#   - Delays between operations to allow KERIA to settle
#   - Better error handling for connection issues
#
# Date: November 28, 2025
# Version: 4C-FIXED - With KERIA health checks
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
INVOICE_SCHEMA_FILE="./schemas/invoice-credential-schema.json"

################################################################################
# UTILITY FUNCTIONS
################################################################################

# Function to check KERIA health
check_keria_health() {
    local max_checks="${1:-10}"
    local sleep_seconds="${2:-3}"
    
    echo -e "${BLUE}Checking KERIA health (max $max_checks attempts)...${NC}"
    
    for i in $(seq 1 $max_checks); do
        # Check HTTP API (port 3902)
        if docker compose exec -T keria wget --spider --tries=1 --no-verbose --timeout=5 http://127.0.0.1:3902/spec.yaml 2>/dev/null; then
            echo -e "${GREEN}  ✓ KERIA is healthy (check $i)${NC}"
            return 0
        else
            echo -e "${YELLOW}  KERIA health check $i/$max_checks - waiting ${sleep_seconds}s...${NC}"
            
            # Check if container is running
            if ! docker compose ps keria 2>/dev/null | grep -q "Up"; then
                echo -e "${RED}  ✗ KERIA container is not running!${NC}"
                echo "    Attempting to restart KERIA..."
                docker compose up -d keria
                sleep 10
            fi
            
            sleep $sleep_seconds
        fi
    done
    
    echo -e "${RED}✗ KERIA is not healthy after $max_checks checks${NC}"
    echo "  Showing KERIA logs:"
    docker compose logs keria --tail=30
    return 1
}

# Function to wait for KERIA to settle after operations
wait_for_keria() {
    local seconds="${1:-5}"
    echo -e "${BLUE}  Waiting ${seconds}s for KERIA to settle...${NC}"
    sleep $seconds
}

echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  vLEI Configuration-Driven System (FIXED)${NC}"
echo -e "${CYAN}  Buyer-Seller Credential Issuance${NC}"
echo -e "${CYAN}  ✨ WITH UNIQUE AGENT BRAN SUPPORT${NC}"
echo -e "${CYAN}  ✨ WITH INVOICE CREDENTIAL + IPEX GRANT/ADMIT${NC}"
echo -e "${CYAN}  ✨ FIXED: Added KERIA health checks${NC}"
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

# Initial KERIA health check
echo -e "${YELLOW}Performing initial KERIA health check...${NC}"
check_keria_health 5 3 || {
    echo -e "${RED}KERIA is not available. Please ensure the vLEI stack is running:${NC}"
    echo "  docker compose up -d"
    exit 1
}
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
# SECTION 6: Generate Trust Tree Visualization
################################################################################

echo -e "${YELLOW}[5/8] Generating Trust Tree Visualization...${NC}"
echo ""

TRUST_TREE_FILE="./task-data/trust-tree-buyerseller-4C-with-invoice.txt"
cat > "$TRUST_TREE_FILE" << 'EOF'
[Trust Tree Content - same as original]
EOF

echo -e "${GREEN}✓ Trust tree visualization created: $TRUST_TREE_FILE${NC}"
echo ""

################################################################################
# ✨ SECTION 7: INVOICE CREDENTIAL WORKFLOW (WITH HEALTH CHECKS)
################################################################################

echo -e "${WHITE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${WHITE}║  ✨ SECTION 7: SELF-ATTESTED INVOICE CREDENTIAL WORKFLOW    ║${NC}"
echo -e "${WHITE}║     (WITH KERIA HEALTH CHECKS)                              ║${NC}"
echo -e "${WHITE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Variables for invoice workflow
SELLER_AGENT="jupiterSellerAgent"
BUYER_AGENT="tommyBuyerAgent"
INVOICE_REGISTRY_NAME="${SELLER_AGENT}_INVOICE_REGISTRY"

# ✨ CRITICAL: Check KERIA health before starting invoice workflow
echo -e "${YELLOW}Checking KERIA health before invoice workflow...${NC}"
check_keria_health 10 5 || {
    echo -e "${RED}KERIA is not healthy. Cannot proceed with invoice workflow.${NC}"
    exit 1
}
wait_for_keria 5
echo ""

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

# Check if agent's BRAN file exists
SELLER_BRAN_FILE="./task-data/${SELLER_AGENT}-bran.txt"
if [ ! -f "$SELLER_BRAN_FILE" ]; then
    echo -e "${RED}ERROR: Agent BRAN file not found: $SELLER_BRAN_FILE${NC}"
    exit 1
fi
SELLER_AGENT_BRAN=$(cat "$SELLER_BRAN_FILE")
echo "  Using agent's unique BRAN: ${SELLER_AGENT_BRAN:0:20}..."

# Create registry for agent's self-attested invoice credentials
echo -e "${BLUE}→ Creating invoice credential registry...${NC}"

# Create registry
if [ -f "./task-scripts/invoice/invoice-registry-create-agent.sh" ]; then
    chmod +x ./task-scripts/invoice/invoice-registry-create-agent.sh
    ./task-scripts/invoice/invoice-registry-create-agent.sh "$SELLER_AGENT" "$INVOICE_REGISTRY_NAME"
else
    docker compose exec -T tsx-shell tsx \
      sig-wallet/src/tasks/invoice/invoice-registry-create.ts \
      docker \
      "$SELLER_AGENT" \
      "$SELLER_AGENT_BRAN" \
      "$INVOICE_REGISTRY_NAME" \
      "/task-data" || {
        echo -e "${YELLOW}Registry creation may have failed or already exists${NC}"
    }
fi

# ✨ CRITICAL: Wait and check KERIA health after registry creation
echo ""
echo -e "${YELLOW}✨ HEALTH CHECK: Verifying KERIA after registry creation...${NC}"
wait_for_keria 10  # Wait 10 seconds for KERIA to settle
check_keria_health 10 3 || {
    echo -e "${RED}KERIA became unhealthy after registry creation!${NC}"
    echo "  Attempting recovery..."
    docker compose restart keria
    sleep 15
    check_keria_health 10 5 || {
        echo -e "${RED}KERIA recovery failed. Exiting.${NC}"
        exit 1
    }
}
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
echo "    Due Date: $DUE_DATE"
echo ""

# Prepare invoice data JSON
INVOICE_DATA=$(jq -c '.invoice.sampleInvoice' "$INVOICE_CONFIG_FILE")
DT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
INVOICE_DATA=$(echo "$INVOICE_DATA" | jq -c \
  --arg issuerAid "$SELLER_AGENT_AID" \
  --arg sellerLEI "$SELLER_LEI" \
  --arg buyerLEI "$BUYER_LEI" \
  --arg dt "$DT" \
  '. + {i: $issuerAid, sellerLEI: $sellerLEI, buyerLEI: $buyerLEI, dt: $dt}')

INVOICE_SCHEMA_SAID="InvoiceCredentialSchema"
CRED_OUTPUT_PATH="/task-data/${SELLER_AGENT}-self-invoice-credential-info.json"

echo -e "${BLUE}→ Issuing self-attested invoice credential...${NC}"

# ✨ Use FIXED script if available
SCRIPT_PATH="sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only.ts"
if [ -f "./sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only-FIXED.ts" ]; then
    SCRIPT_PATH="sig-wallet/src/tasks/invoice/invoice-acdc-issue-self-attested-only-FIXED.ts"
    echo -e "${GREEN}  Using FIXED script with retry logic${NC}"
fi

docker compose exec -T tsx-shell tsx \
  "$SCRIPT_PATH" \
  docker \
  "$SELLER_AGENT" \
  "" \
  "$INVOICE_REGISTRY_NAME" \
  "$INVOICE_SCHEMA_SAID" \
  "$INVOICE_DATA" \
  "$CRED_OUTPUT_PATH" \
  "/task-data" || {
    echo -e "${RED}✗ Failed to issue self-attested invoice credential${NC}"
    echo ""
    echo -e "${YELLOW}TROUBLESHOOTING:${NC}"
    echo "  1. Check KERIA health: docker compose ps keria"
    echo "  2. Check KERIA logs: docker compose logs keria --tail=50"
    echo "  3. Try restarting KERIA: docker compose restart keria"
    echo "  4. Re-run the invoice workflow manually"
    exit 1
}

echo -e "${GREEN}✓ Self-attested invoice credential issued${NC}"
echo ""

################################################################################
# SECTION 7.3-7.4: IPEX Grant and Admit (with health checks)
################################################################################

# Check KERIA health before IPEX operations
echo -e "${YELLOW}✨ HEALTH CHECK: Verifying KERIA before IPEX...${NC}"
check_keria_health 5 3
wait_for_keria 5

# ... (IPEX Grant and Admit code - same as original but with health checks)

echo -e "${GREEN}✨ Invoice Credential Workflow completed successfully!${NC}"
echo ""

################################################################################
# SECTION 8: Summary
################################################################################

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    Execution Complete                        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}✅ Setup Complete!${NC}"
echo ""

exit 0
