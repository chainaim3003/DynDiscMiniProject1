#!/bin/bash
# person-acdc-present-oor.sh - Present the OOR ACDC to the verifier (Sally)
# Usage: person-acdc-present-oor.sh [alias]
#   alias: Optional unique alias for the Person AID (defaults to 'person')

set -e

# Accept optional alias parameter
PERSON_ALIAS=${1:-"person"}

CRED_SAID=$(jq -r .said ./task-data/oor-credential-info.json)
echo "Presenting OOR ACDC to verifier Credential SAID: ${CRED_SAID}"
echo "Using Person alias: ${PERSON_ALIAS}"
source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh person/person-acdc-present-oor.ts \
    'docker' \
    "${PERSON_SALT}" \
    "${PERSON_ALIAS}" \
    "${CRED_SAID}" \
    "${VERIFIER_AID}"
