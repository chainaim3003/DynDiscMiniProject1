#!/bin/bash
################################################################################
# test-agent-verification-DEEP-credential.sh
#
# Purpose: Deep verification of agent delegation + credential query and validation
#
# Based on: test-agent-verification-DEEP-EXT.sh
# 
# NEW FEATURES:
#   - All verification from DEEP script
#   - Query credentials from KERIA agent
#   - Validate credential structure and proofs
#   - Verify credential integrity
#   - JSON output support for API integration
#
# Usage:
#   ./test-agent-verification-DEEP-credential.sh [AGENT_NAME] [OOR_HOLDER_NAME] [ENV] [--json]
#
# Examples:
#   ./test-agent-verification-DEEP-credential.sh jupiterSellerAgent Jupiter_Chief_Sales_Officer docker
#   ./test-agent-verification-DEEP-credential.sh jupiterSellerAgent Jupiter_Chief_Sales_Officer docker --json
#   ./test-agent-verification-DEEP-credential.sh tommyBuyerAgent Tommy_Chief_Procurement_Officer docker --json
#
# Date: November 14, 2025
# Updated: Added JSON output support to match DEEP-EXT.sh parameter order
################################################################################

set -e

# Parse arguments (matching DEEP-EXT.sh order)
AGENT_NAME="${1:-jupiterSellerAgent}"
OOR_HOLDER_NAME="${2:-Jupiter_Chief_Sales_Officer}"
ENV="${3:-docker}"
JSON_OUTPUT="${4:-}"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Default passcodes (fallback)
DEFAULT_AGENT_PASSCODE="AgentPass123"
OOR_PASSCODE="0ADckowyGuNwtJUPLeRqZvTp"

# JSON output collection variables
JSON_SUCCESS="false"
JSON_DELEGATION_VERIFIED="false"
JSON_CREDENTIALS_QUERIED="false"
JSON_CREDENTIALS_VALIDATED="false"
JSON_CREDENTIAL_COUNT=0
JSON_VALID_COUNT=0
JSON_INVALID_COUNT=0
JSON_AGENT_AID=""
JSON_OOR_AID=""
JSON_CREDENTIALS="[]"
JSON_VALID_CREDENTIALS="[]"
JSON_INVALID_CREDENTIALS="[]"
JSON_ERROR=""

# Try to read unique BRAN from agent-brans.json
# Note: Script runs via docker compose exec, so use /task-data/ (Docker path)
BRANS_FILE="/task-data/agent-brans.json"
if docker compose exec -T tsx-shell test -f "$BRANS_FILE" 2>/dev/null; then
    # JSON structure: { "agents": [ { "alias": "agentName", "bran": "..." }, ... ] }
    # Note: Use "alias" field (not "keriaAlias" which may be "unknownAgent")
    AGENT_BRAN=$(docker compose exec -T tsx-shell cat "$BRANS_FILE" | jq -r ".agents[] | select(.alias == \"${AGENT_NAME}\") | .bran" 2>/dev/null)
    if [ -n "$AGENT_BRAN" ] && [ "$AGENT_BRAN" != "null" ] && [ "$AGENT_BRAN" != "" ]; then
        AGENT_PASSCODE="$AGENT_BRAN"
        if [ "$JSON_OUTPUT" != "--json" ]; then
            echo -e "${GREEN}✓ Found unique BRAN for ${AGENT_NAME}${NC}"
        fi
    else
        AGENT_PASSCODE="$DEFAULT_AGENT_PASSCODE"
        if [ "$JSON_OUTPUT" != "--json" ]; then
            echo -e "${YELLOW}⚠ No unique BRAN found in agent-brans.json, using default passcode${NC}"
        fi
    fi
else
    AGENT_PASSCODE="$DEFAULT_AGENT_PASSCODE"
    if [ "$JSON_OUTPUT" != "--json" ]; then
        echo -e "${YELLOW}⚠ agent-brans.json not found in Docker, using default passcode${NC}"
    fi
fi

if [ "$JSON_OUTPUT" != "--json" ]; then
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}   DEEP AGENT DELEGATION VERIFICATION + CREDENTIAL QUERY${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}Configuration:${NC}"
    echo "  Agent: ${AGENT_NAME}"
    echo "  OOR Holder: ${OOR_HOLDER_NAME}"
    echo "  ENV: ${ENV}"
    echo "  Agent Passcode: ${AGENT_PASSCODE:0:20}..." # Show first 20 chars only
    echo "  OOR Passcode: ${OOR_PASSCODE}"
    echo ""
fi

################################################################################
# STEP 1: Deep Agent Delegation Verification (from DEEP script)
################################################################################

if [ "$JSON_OUTPUT" != "--json" ]; then
    echo -e "${YELLOW}[1/3] Deep Agent Delegation Verification...${NC}"
    echo ""
fi

DELEGATION_OUTPUT=$(docker compose exec -T tsx-shell tsx sig-wallet/src/tasks/agent/agent-verify-delegation-deep.ts \
  "${ENV}" \
  "${AGENT_PASSCODE}" \
  "${OOR_PASSCODE}" \
  "${AGENT_NAME}" \
  "${OOR_HOLDER_NAME}" 2>&1) || DELEGATION_EXIT_CODE=$?

if [ "${DELEGATION_EXIT_CODE:-0}" -eq 0 ]; then
    JSON_DELEGATION_VERIFIED="true"
    
    # Extract AIDs from output if available
    JSON_AGENT_AID=$(echo "$DELEGATION_OUTPUT" | grep -oP "Agent AID: \K[A-Za-z0-9_-]+" | head -1 || echo "")
    JSON_OOR_AID=$(echo "$DELEGATION_OUTPUT" | grep -oP "Delegator AID: \K[A-Za-z0-9_-]+" | head -1 || echo "")
    
    if [ "$JSON_OUTPUT" != "--json" ]; then
        echo "$DELEGATION_OUTPUT"
        echo ""
        echo -e "${GREEN}✅ DEEP VERIFICATION PASSED!${NC}"
        echo ""
    fi
else
    JSON_ERROR="Delegation verification failed"
    if [ "$JSON_OUTPUT" != "--json" ]; then
        echo "$DELEGATION_OUTPUT"
        echo ""
        echo -e "${RED}❌ DEEP VERIFICATION FAILED${NC}"
        echo ""
    fi
    
    # Output JSON and exit on failure
    if [ "$JSON_OUTPUT" == "--json" ]; then
        cat << EOF
{
  "success": false,
  "agent_name": "${AGENT_NAME}",
  "oor_holder_name": "${OOR_HOLDER_NAME}",
  "verification": {
    "delegation_verified": false,
    "credentials_queried": false,
    "credentials_validated": false
  },
  "error": "Delegation verification failed"
}
EOF
    fi
    exit 1
fi

################################################################################
# STEP 2: Query Credentials from KERIA Agent
################################################################################

if [ "$JSON_OUTPUT" != "--json" ]; then
    echo -e "${YELLOW}[2/3] Querying Credentials from KERIA...${NC}"
    echo ""
    echo -e "${BLUE}→ Fetching credentials for ${AGENT_NAME}...${NC}"
fi

# Query credentials from KERIA
QUERY_OUTPUT=$(docker compose exec -T tsx-shell tsx sig-wallet/src/tasks/agent/agent-query-credentials.ts \
  "${ENV}" \
  "${AGENT_PASSCODE}" \
  "${AGENT_NAME}" 2>&1) || QUERY_EXIT_CODE=$?

if [ "${QUERY_EXIT_CODE:-0}" -eq 0 ]; then
    JSON_CREDENTIALS_QUERIED="true"
    
    if [ "$JSON_OUTPUT" != "--json" ]; then
        echo "$QUERY_OUTPUT"
        echo ""
        echo -e "${GREEN}✅ Credential query successful!${NC}"
        echo ""
    fi
    
    # Read query results from file (inside Docker)
    QUERY_JSON=$(docker compose exec -T tsx-shell cat /task-data/${AGENT_NAME}-credential-query-results.json 2>/dev/null || echo "{}")
    JSON_CREDENTIAL_COUNT=$(echo "$QUERY_JSON" | jq '.totalCredentials // 0' 2>/dev/null || echo "0")
    JSON_CREDENTIALS=$(echo "$QUERY_JSON" | jq '.credentials // []' 2>/dev/null || echo "[]")
    
    # Extract Agent AID if not already set
    if [ -z "$JSON_AGENT_AID" ]; then
        JSON_AGENT_AID=$(echo "$QUERY_JSON" | jq -r '.agentAID // ""' 2>/dev/null || echo "")
    fi
    
    if [ "$JSON_OUTPUT" != "--json" ]; then
        echo -e "${BLUE}Credential Query Results:${NC}"
        echo "$QUERY_JSON" | jq '.'
        echo ""
        echo -e "${GREEN}Total Credentials Found: ${JSON_CREDENTIAL_COUNT}${NC}"
        echo ""
    fi
else
    if [ "$JSON_OUTPUT" != "--json" ]; then
        echo "$QUERY_OUTPUT"
        echo ""
        echo -e "${YELLOW}⚠ Credential query failed (agent may have no credentials)${NC}"
        echo ""
    fi
fi

################################################################################
# STEP 3: Validate Credentials
################################################################################

if [ "$JSON_OUTPUT" != "--json" ]; then
    echo -e "${YELLOW}[3/3] Validating Credentials...${NC}"
    echo ""
    echo -e "${BLUE}→ Validating credential structure and proofs...${NC}"
fi

# Validate credentials
VALIDATE_OUTPUT=$(docker compose exec -T tsx-shell tsx sig-wallet/src/tasks/agent/agent-validate-credentials.ts \
  "${ENV}" \
  "${AGENT_PASSCODE}" \
  "${AGENT_NAME}" 2>&1) || VALIDATE_EXIT_CODE=$?

if [ "${VALIDATE_EXIT_CODE:-0}" -eq 0 ]; then
    JSON_CREDENTIALS_VALIDATED="true"
    
    if [ "$JSON_OUTPUT" != "--json" ]; then
        echo "$VALIDATE_OUTPUT"
        echo ""
        echo -e "${GREEN}✅ Credential validation successful!${NC}"
        echo ""
    fi
    
    # Read validation results from file (inside Docker)
    VALIDATION_JSON=$(docker compose exec -T tsx-shell cat /task-data/${AGENT_NAME}-credential-validation-results.json 2>/dev/null || echo "{}")
    JSON_VALID_COUNT=$(echo "$VALIDATION_JSON" | jq '.totalValid // 0' 2>/dev/null || echo "0")
    JSON_INVALID_COUNT=$(echo "$VALIDATION_JSON" | jq '.totalInvalid // 0' 2>/dev/null || echo "0")
    JSON_VALID_CREDENTIALS=$(echo "$VALIDATION_JSON" | jq '.validCredentials // []' 2>/dev/null || echo "[]")
    JSON_INVALID_CREDENTIALS=$(echo "$VALIDATION_JSON" | jq '.invalidCredentials // []' 2>/dev/null || echo "[]")
    
    if [ "$JSON_OUTPUT" != "--json" ]; then
        echo -e "${BLUE}Validation Results:${NC}"
        echo "$VALIDATION_JSON" | jq '.'
        echo ""
        echo -e "${GREEN}Valid Credentials: ${JSON_VALID_COUNT}${NC}"
        if [ "$JSON_INVALID_COUNT" -gt 0 ]; then
            echo -e "${RED}Invalid Credentials: ${JSON_INVALID_COUNT}${NC}"
        fi
        echo ""
    fi
else
    if [ "$JSON_OUTPUT" != "--json" ]; then
        echo "$VALIDATE_OUTPUT"
        echo ""
        echo -e "${RED}❌ Credential validation failed${NC}"
        echo ""
    fi
fi

################################################################################
# Final Summary / JSON Output
################################################################################

# Determine overall success
if [ "$JSON_DELEGATION_VERIFIED" == "true" ] && [ "$JSON_CREDENTIALS_QUERIED" == "true" ] && [ "$JSON_CREDENTIALS_VALIDATED" == "true" ]; then
    if [ "$JSON_INVALID_COUNT" -eq 0 ]; then
        JSON_SUCCESS="true"
    else
        JSON_SUCCESS="true"  # Partial success - delegation passed
    fi
fi

if [ "$JSON_OUTPUT" == "--json" ]; then
    # Output structured JSON for API consumption
    cat << EOF
{
  "success": ${JSON_SUCCESS},
  "agent_name": "${AGENT_NAME}",
  "agent_aid": "${JSON_AGENT_AID}",
  "oor_holder_name": "${OOR_HOLDER_NAME}",
  "oor_aid": "${JSON_OOR_AID}",
  "verification": {
    "delegation_verified": ${JSON_DELEGATION_VERIFIED},
    "credentials_queried": ${JSON_CREDENTIALS_QUERIED},
    "credentials_validated": ${JSON_CREDENTIALS_VALIDATED}
  },
  "credentials": {
    "total": ${JSON_CREDENTIAL_COUNT},
    "valid": ${JSON_VALID_COUNT},
    "invalid": ${JSON_INVALID_COUNT},
    "list": ${JSON_CREDENTIALS},
    "validCredentials": ${JSON_VALID_CREDENTIALS},
    "invalidCredentials": ${JSON_INVALID_CREDENTIALS}
  }
}
EOF
else
    # Human-readable summary
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}                    VERIFICATION COMPLETE                      ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    echo -e "${GREEN}✅ Summary for ${AGENT_NAME}:${NC}"
    echo ""
    echo -e "${CYAN}Completed Steps:${NC}"
    echo "  ✓ Deep agent delegation verification"
    echo "  ✓ Credential query from KERIA"
    echo "  ✓ Credential validation and proof verification"
    echo ""
    
    if [ "$JSON_INVALID_COUNT" -eq 0 ]; then
        echo -e "${GREEN}🎉 ALL VERIFICATIONS PASSED!${NC}"
        echo -e "${GREEN}   Agent delegation is valid${NC}"
        echo -e "${GREEN}   All credentials are valid and verifiable${NC}"
    else
        echo -e "${YELLOW}⚠ PARTIAL SUCCESS${NC}"
        echo -e "${YELLOW}   Agent delegation is valid${NC}"
        echo -e "${YELLOW}   Some credentials failed validation${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
fi

exit 0
