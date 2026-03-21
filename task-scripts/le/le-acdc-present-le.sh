#!/bin/bash
# le-acdc-present-le.sh - Present the LE ACDC to the verifier (Sally)
# Usage: le-acdc-present-le.sh [alias]
#   alias: Optional unique alias for the LE AID (defaults to 'le')

set -e

# Accept optional alias parameter
LE_ALIAS=${1:-"le"}

CRED_SAID=$(jq -r .said ./task-data/le-credential-info.json)
echo "Presenting LE ACDC to verifier Credential SAID: ${CRED_SAID}"
echo "Using LE alias: ${LE_ALIAS}"
source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh le/le-acdc-present-le.ts \
    'docker' \
    "${LE_SALT}" \
    "${LE_ALIAS}" \
    "${CRED_SAID}" \
    "${VERIFIER_AID}"
