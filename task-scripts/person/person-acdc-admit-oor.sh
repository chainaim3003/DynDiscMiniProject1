#!/bin/bash
# person-acdc-admit-oor.sh - Admit the OOR ACDC Credential
# Usage: person-acdc-admit-oor.sh [alias]
#   alias: Optional unique alias for the Person AID (defaults to 'person')

set -e

# Accept optional alias parameter
PERSON_ALIAS=${1:-"person"}

GRANT_SAID=$(jq -r .grantSaid ./task-data/oor-credential-info.json)
QVI_PREFIX=$(jq -r .aid ./task-data/qvi-info.json)
echo "Admitting OOR ACDC with IPEX Grant SAID: ${GRANT_SAID}"
echo "Using Person alias: ${PERSON_ALIAS}"
source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh person/person-acdc-admit-oor.ts \
    'docker' \
    "${PERSON_SALT}" \
    "${GRANT_SAID}" \
    "${QVI_PREFIX}" \
    "${PERSON_ALIAS}"
