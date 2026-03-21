#!/bin/bash
# qvi-acdc-issue-oor.sh - Issue OOR Credential from QVI to Person AID

# Create OOR ACDC Credential
# This script issues the OOR credential from the QVI AID to the Person AID

set -e

echo "Creating OOR ACDC Credential"
source ./task-scripts/workshop-env-vars.sh

# Check if required info files exist
if [ ! -f "./task-data/person-info.json" ]; then
    echo "Required Person AID info file not found. Please run person-aid-create.sh first."
    exit 1
fi
if [ ! -f "./task-data/qvi-registry-info.json" ]; then
    echo "Required QVI registry info file not found. Please run qvi-registry-create.sh first."
    exit 1
fi
if [ ! -f "./task-data/oor-auth-credential-info.json" ]; then
    echo "Required OOR Auth credential info file not found. Please run le-acdc-issue-oor-auth.sh first."
    exit 1
fi

# get Person AID from person-info.json
PERSON_AID=$(jq -r '.aid' ./task-data/person-info.json)

# get OOR Auth Credential SAID from oor-auth-credential-info.json for the edge
OOR_AUTH_CRED_SAID=$(jq -r '.said' ./task-data/oor-auth-credential-info.json)

# Person data for OOR credential - accept as parameters or use defaults
PERSON_NAME="${1:-John Smith}"
PERSON_OOR="${2:-Head of Standards}"
LE_LEI="${3:-254900OPPU84GM83MG36}"

# Issue the OOR credential
if [ -n "$1" ]; then
    echo "Issuing OOR credential to ${PERSON_AID} for person ${PERSON_NAME} (${PERSON_OOR}) with LEI ${LE_LEI} (from parameters)"
else
    echo "Issuing OOR credential to ${PERSON_AID} for person ${PERSON_NAME} (${PERSON_OOR}) with LEI ${LE_LEI} (defaults - hardcoded)"
fi
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh qvi/qvi-acdc-issue-oor.ts \
    'docker' \
    'qvi' \
    "${QVI_SALT}" \
    "qvi-le-registry" \
    "${OOR_SCHEMA_SAID}" \
    "${PERSON_AID}" \
    "${PERSON_NAME}" \
    "${PERSON_OOR}" \
    "${OOR_AUTH_CRED_SAID}" \
    "${LE_LEI}" \
    "/task-data/oor-credential-info.json"