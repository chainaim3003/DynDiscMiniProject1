#!/bin/bash
# le-acdc-issue-oor-auth.sh - Issue OOR Auth Credential from LE to QVI AID

# Create OOR Auth ACDC Credential
# This script issues the OOR Auth credential from the LE AID to the QVI AID

set -e

echo "Creating OOR Auth ACDC Credential"
source ./task-scripts/workshop-env-vars.sh

# Check if required info files exist
if [ ! -f "./task-data/person-info.json" ]; then
    echo "Required Person AID info file not found. Please run person-aid-create.sh first."
    exit 1
fi
if [ ! -f "./task-data/le-registry-info.json" ]; then
    echo "Required LE registry info file not found. Please run le-registry-create.sh first."
    exit 1
fi
if [ ! -f "./task-data/le-credential-info.json" ]; then
    echo "Required LE credential info file not found. Please run qvi-acdc-issue-le.sh first."
    exit 1
fi

# get QVI AID from qvi-info.json
QVI_AID=$(jq -r '.aid' ./task-data/qvi-info.json)

# get Person AID from person-info.json
PERSON_AID=$(jq -r '.aid' ./task-data/person-info.json)

# get LE Credential SAID from le-credential-info.json for the edge
LE_CRED_SAID=$(jq -r '.said' ./task-data/le-credential-info.json)

# Person data for OOR Auth credential - accept as parameters or use defaults
PERSON_NAME="${1:-John Smith}"
PERSON_OOR="${2:-Head of Standards}"
LE_LEI="${3:-254900OPPU84GM83MG36}"
LE_ALIAS="${4:-le}"  # Accept LE alias as 4th parameter, default to 'le'

# Issue the OOR Auth credential
if [ -n "$1" ]; then
    echo "Issuing OOR Auth credential to ${QVI_AID} for person ${PERSON_NAME} (${PERSON_OOR}) with LEI ${LE_LEI} (from parameters)"
    echo "Using LE alias: ${LE_ALIAS}"
else
    echo "Issuing OOR Auth credential to ${QVI_AID} for person ${PERSON_NAME} (${PERSON_OOR}) with LEI ${LE_LEI} (defaults - hardcoded)"
    echo "Using LE alias: ${LE_ALIAS}"
fi
# Use dynamic registry name based on LE alias
REGISTRY_NAME="${LE_ALIAS}-oor-registry"
echo "Using registry: ${REGISTRY_NAME}"

docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh le/le-acdc-issue-oor-auth.ts \
    'docker' \
    "${LE_ALIAS}" \
    "${LE_SALT}" \
    "${REGISTRY_NAME}" \
    "${OOR_AUTH_SCHEMA_SAID}" \
    "${QVI_AID}" \
    "${PERSON_AID}" \
    "${PERSON_NAME}" \
    "${PERSON_OOR}" \
    "${LE_CRED_SAID}" \
    "${LE_LEI}" \
    "/task-data/oor-auth-credential-info.json"

