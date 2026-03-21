#!/bin/bash
# le-acdc-admit-le.sh - Admit the LE ACDC Credential
# Usage: le-acdc-admit-le.sh [alias]
#   alias: Optional unique alias for the LE AID (defaults to 'le')

set -e

# Accept optional alias parameter
LE_ALIAS=${1:-"le"}

GRANT_SAID=$(jq -r .grantSaid ./task-data/le-credential-info.json)
QVI_PREFIX=$(jq -r .aid ./task-data/qvi-info.json)
echo "Admitting LE ACDC with IPEX Grant SAID: ${GRANT_SAID}"
echo "Using LE alias: ${LE_ALIAS}"
source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh le/le-acdc-admit-le.ts \
    'docker' \
    "${LE_SALT}" \
    "${GRANT_SAID}" \
    "${QVI_PREFIX}" \
    "${LE_ALIAS}"
