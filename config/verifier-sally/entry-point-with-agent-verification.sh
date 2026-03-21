#!/bin/bash
# entry-point-with-agent-verification.sh
# Sally with Agent Delegation Verification Endpoint

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
else
  echo "GEDA_PRE auth AID is set to: ${GEDA_PRE}"
fi

echo "Starting Sally verifier with Agent Delegation Verification..."

export DEBUG_KLI=true

function start_sally() {
  # Start Sally with custom agent verification module
  python3 -c "
import sys
sys.path.insert(0, '/sally')
from sally.app.cli.commands import server
from sally_agent_verification import add_agent_delegation_route

# Monkey-patch to add our endpoint
original_setup = server.setup

def setup_with_agent_verification(*args, **kwargs):
    app, hab = original_setup(*args, **kwargs)
    add_agent_delegation_route(app, hab)
    return app, hab

server.setup = setup_with_agent_verification
" &

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

mkdir -p /usr/local/var/keri/ks

echo "Checking for existing Sally AID..."
if [[ -d "/usr/local/var/keri/ks/${SALLY_KS_NAME}" ]]; then
  echo "Sally keystore directory exists."
  EXISTING_AID=$(kli aid --name "${SALLY_KS_NAME}" --alias "${SALLY_KS_NAME}" --passcode "${SALLY_PASSCODE}")
  echo "Existing Sally AID: ${EXISTING_AID}"
  
  if [[ "${EXISTING_AID}" == "${EXPECTED_AID}" ]]; then
    echo "Sally AID prefix matches expected value: ${EXPECTED_AID}"
    start_sally
  else
    echo "Sally AID prefix mismatch!"
    echo "   Expected: ${EXPECTED_AID}"
    echo "   Actual:   ${EXISTING_AID}"
    exit 1
  fi
else
  echo "Sally keystore does not exist. Initializing..."
  init_sally_aid
  start_sally
fi
