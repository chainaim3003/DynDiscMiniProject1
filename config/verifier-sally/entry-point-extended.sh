#!/bin/bash
set -e

echo "=================================================="
echo "Sally Verifier - Extended Entry Point"
echo "Custom Agent Delegation Verification Enabled"
echo "=================================================="

# Environment variables from docker-compose
SALLY_KS_NAME="${SALLY_KS_NAME:-verifier}"
SALLY_SALT="${SALLY_SALT}"
SALLY_PASSCODE="${SALLY_PASSCODE}"
SALLY_PORT="${SALLY_PORT:-9723}"
GEDA_PRE="${GEDA_PRE}"
WEBHOOK_URL="${WEBHOOK_URL:-http://resource:9923}"

# Paths
KERI_DIR="/usr/local/var/keri"
KS_DIR="${KERI_DIR}/ks/${SALLY_KS_NAME}"

echo "Environment:"
echo "  SALLY_KS_NAME: ${SALLY_KS_NAME}"
echo "  SALLY_PORT: ${SALLY_PORT}"
echo "  GEDA_PRE: ${GEDA_PRE}"
echo "  KERI_DIR: ${KERI_DIR}"
echo "  WEBHOOK_URL: ${WEBHOOK_URL}"

# Check if Sally AID already exists
echo ""
echo "Checking for existing Sally AID..."
if [[ -d "${KS_DIR}" ]]; then
    echo "  ✓ Sally keystore found at: ${KS_DIR}"
    echo "  Skipping initialization (AID already exists)"
    SKIP_INIT=true
else
    echo "  Creating new Sally AID..."
    SKIP_INIT=false
fi

# Initialize KERI (if needed)
if [ "${SKIP_INIT}" = false ]; then
    echo ""
    echo "Initializing KERI..."
    
    # Validate required environment variables
    if [ -z "${SALLY_SALT}" ] || [ -z "${SALLY_PASSCODE}" ]; then
        echo "ERROR: SALLY_SALT and SALLY_PASSCODE must be set"
        exit 1
    fi
    
    echo "  Running kli init..."
    kli init \
        --name "${SALLY_KS_NAME}" \
        --salt "${SALLY_SALT}" \
        --passcode "${SALLY_PASSCODE}" \
        --config-dir /sally/conf \
        --config-file verifier
    
    echo "  ✓ KERI initialized"
    
    # Create Sally AID
    echo ""
    echo "Creating Sally AID..."
    kli incept \
        --name "${SALLY_KS_NAME}" \
        --alias "${SALLY_KS_NAME}" \
        --passcode "${SALLY_PASSCODE}" \
        --file /sally/conf/incept-no-wits.json
    
    echo "  ✓ Sally AID created"
else
    echo ""
    echo "Skipped initialization (AID exists)"
fi

# Start Sally with all required arguments
echo ""
echo "Starting Sally server with custom agent verification..."
echo "=================================================="
echo ""
echo "Custom endpoints available:"
echo "  POST /verify/agent-delegation - Agent delegation verification"
echo ""

# Start Sally with ALL required arguments
exec sally server start \
    --name "${SALLY_KS_NAME}" \
    --passcode "${SALLY_PASSCODE}" \
    --alias "${SALLY_KS_NAME}" \
    --config-dir /sally/conf \
    --config-file verifier \
    --web-hook "${WEBHOOK_URL}" \
    --auth "${GEDA_PRE}" \
    --direct \
    --port "${SALLY_PORT}"
