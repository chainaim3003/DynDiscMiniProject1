#!/bin/bash
# entry-point-with-agent-verification.sh
# Sally entry point with Agent Delegation Verification endpoint

set -e

EXPECTED_AID=EMrjKv0T43sslqFfhlEHC9v3t9UoxHWrGznQ1EveRXUO
GEDA_PRE="${GEDA_PRE:-ED1e8pD24aqd0dCZTQHaGpfcluPFD2ajGIY3ARgE5Yvr}"
SALLY_KS_NAME="${SALLY_KS_NAME:-sally}"
SALLY_SALT="${SALLY_SALT:-0ABVqAtad0CBkhDhCEPd514T}"
SALLY_PASSCODE="${SALLY_PASSCODE:-4TBjjhmKu9oeDp49J7Xdy}"
SALLY_PORT="${SALLY_PORT:-9723}"
WEBHOOK_URL="${WEBHOOK_URL:-http://resource:9923}"

if [ -z "${GEDA_PRE}" ]; then
  echo "GEDA_PRE auth AID is not set. Exiting."
  exit 1
fi

echo "=================================================="
echo "  Sally Verifier with Agent Verification"
echo "  Extended with POST /verify/agent-delegation"
echo "=================================================="
echo ""
echo "Configuration:"
echo "   GEDA_PRE: ${GEDA_PRE}"
echo "   SALLY_PORT: ${SALLY_PORT}"
echo "   WEBHOOK_URL: ${WEBHOOK_URL}"
echo ""

export DEBUG_KLI=true
export PYTHONPATH=/sally:${PYTHONPATH}

function start_sally_with_agent_verification() {
  echo "Starting Sally with agent verification extension..."
  
  # Execute the startup script that patches Sally
  python3 -c "
import sys
sys.path.insert(0, '/sally')

# Import and execute the patched startup
import sally_startup_with_agent_verification
" &
  
  # Small delay to let the patch load
  sleep 1
  
  # Start Sally normally - it will use the patched version
  sally server start \
    --direct \
    --name "${SALLY_KS_NAME}" \
    --alias "${SALLY_KS_NAME}" \
    --passcode "${SALLY_PASSCODE}" \
    --http "${SALLY_PORT}" \
    --config-dir /sally/conf \
    --config-file verifier.json \
    --web-hook "${WEBHOOK_URL}" \
    --auth "${GEDA_PRE}" \
    --loglevel INFO
}

function init_sally_aid() {
  echo "Initializing Sally AID..."
  
  kli init \
    --name "${SALLY_KS_NAME}" \
    --salt "${SALLY_SALT}" \
    --passcode "${SALLY_PASSCODE}" \
    --config-dir /sally/conf \
    --config-file "${SALLY_KS_NAME}.json"

  kli incept \
      --name "${SALLY_KS_NAME}" \
      --alias "${SALLY_KS_NAME}" \
      --passcode "${SALLY_PASSCODE}" \
      --config /sally/conf \
      --file "/sally/conf/incept-no-wits.json"
}

# Ensure directory exists
mkdir -p /usr/local/var/keri/ks

echo "Checking for existing Sally AID..."
if [[ -d "/usr/local/var/keri/ks/${SALLY_KS_NAME}" ]]; then
  echo "Sally keystore directory exists."
  
  EXISTING_AID=$(kli aid --name "${SALLY_KS_NAME}" --alias "${SALLY_KS_NAME}" --passcode "${SALLY_PASSCODE}")
  echo "Existing Sally AID: ${EXISTING_AID}"
  
  if [[ "${EXISTING_AID}" == "${EXPECTED_AID}" ]]; then
    echo "✓ Sally AID matches expected value"
    start_sally_with_agent_verification
  else
    echo "✗ Sally AID mismatch!"
    echo "   Expected: ${EXPECTED_AID}"
    echo "   Actual:   ${EXISTING_AID}"
    exit 1
  fi
else
  echo "Sally keystore does not exist. Initializing..."
  init_sally_aid
  start_sally_with_agent_verification
fi
