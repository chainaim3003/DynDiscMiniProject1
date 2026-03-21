#!/bin/bash
# entry-point.sh - Entry point script for Sally verifier
# This script initializes Sally with the correct configuration
# The below configuration should cause inception of an AID with the prefix
#   EMrjKv0T43sslqFfhlEHC9v3t9UoxHWrGznQ1EveRXUO

set -e

EXPECTED_AID=EMrjKv0T43sslqFfhlEHC9v3t9UoxHWrGznQ1EveRXUO
GEDA_PRE="${GEDA_PRE:-ED1e8pD24aqd0dCZTQHaGpfcluPFD2ajGIY3ARgE5Yvr}"
SALLY_KS_NAME="${SALLY_KS_NAME:-sally}"
SALLY_SALT="${SALLY_SALT:-0ABVqAtad0CBkhDhCEPd514T}"
SALLY_PASSCODE="${SALLY_PASSCODE:-4TBjjhmKu9oeDp49J7Xdy}"
SALLY_PORT="${SALLY_PORT:-9723}"
WEBHOOK_URL="${WEBHOOK_URL:-http://resource:9923}"

# Function to wait for schema service OOBIs to be available
wait_for_schema() {
  echo "Waiting for schema service OOBIs to be available..."
  local MAX_RETRIES=30
  local RETRY_COUNT=0
  local SCHEMA_URL="http://schema:7723/oobi/EBfdlu8R27Fbx-ehrqwImnK-8Cm79sqbAQ4MmvEAYqao"
  
  while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if wget --spider --quiet "$SCHEMA_URL" 2>/dev/null; then
      echo "Schema service is ready!"
      return 0
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for schema service... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
  done
  
  echo "ERROR: Schema service not available after $MAX_RETRIES attempts"
  return 1
}

# Wait for schema service before proceeding
wait_for_schema || exit 1

if [ -z "${GEDA_PRE}" ]; then
  echo "GEDA_PRE auth AID is not set. Exiting."
  exit 1
else
  echo "GEDA_PRE auth AID is set to: ${GEDA_PRE}"
fi

echo "Starting Sally verifier..."

# Set environment variables
echo "Configuration:"
echo "   EXPECTED AID: ${EXPECTED_AID}"
echo "   SALLY KEYSTORE NAME: ${SALLY_KS_NAME}"
echo "   GEDA_PRE: ${GEDA_PRE}"
echo "   SALLY_PORT: ${SALLY_PORT}"
echo "   WEBHOOK_URL: ${WEBHOOK_URL}"

export DEBUG_KLI=true

function start_sally() {
  # Start Sally in direct mode
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
  # Create Habery / keystore
  kli init \
    --name "${SALLY_KS_NAME}" \
    --salt "${SALLY_SALT}" \
    --passcode "${SALLY_PASSCODE}" \
    --config-dir /sally/conf \
    --config-file "${SALLY_KS_NAME}.json"

  # Create sally identifier
  kli incept \
      --name "${SALLY_KS_NAME}" \
      --alias "${SALLY_KS_NAME}" \
      --passcode "${SALLY_PASSCODE}" \
      --config /sally/conf \
      --file "/sally/conf/incept-no-wits.json"
}

# make sure parent directory exists so dir check works
mkdir -p /usr/local/var/keri/ks

echo "Checking for existing Sally AID..."
# check if the /usr/local/var/keri/ks/sally directory exists (has already been initialized
if [[ -d "/usr/local/var/keri/ks/${SALLY_KS_NAME}" ]]; then
  echo "Sally keystore directory exists."

  EXISTING_AID=$(kli aid --name "${SALLY_KS_NAME}" --alias "${SALLY_KS_NAME}" --passcode "${SALLY_PASSCODE}")
  echo "Existing Sally AID: ${EXISTING_AID}"
  echo "Sally AID already exists: ${EXISTING_AID}"
  if [[ "${EXISTING_AID}" == "${EXPECTED_AID}" ]]; then
    echo "Sally AID prefix matches expected value: ${EXPECTED_AID}"
    echo "Existing Sally AID matches expected value. No need to recreate."
    start_sally
  else
    echo "Sally AID prefix mismatch!"
    echo "   Expected: ${EXPECTED_AID}"
    echo "   Actual:   ${ACTUAL_AID}"
    echo "Error: Existing Sally AID does not match expected value. Exiting"
    exit 1
  fi
else
  echo "Sally keystore does not exist. Initializing..."
  # Set up keystore and AID for Sally verifier
  # This manually calls init and incept in order to have a deterministic AID.
  init_sally_aid
  start_sally
fi

