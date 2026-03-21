#!/bin/bash
# le-acdc-issue-ecr-auth.sh - Issue ECR Auth Credential from LE to QVI AID

# Create ECR Auth ACDC Credential
# This script issues the ECR Auth credential from the LE AID to the QVI AID

set -e

echo "Creating ECR Auth ACDC Credential"
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

# Accept parameters or use defaults
PERSON_NAME="${1:-John Smith}"
PERSON_ECR="${2:-Project Manager}"
LE_ALIAS="${3:-le}"  # Accept LE alias as 3rd parameter

# Use dynamic registry name based on LE alias
REGISTRY_NAME="${LE_ALIAS}-oor-registry"

# Issue the ECR Auth credential
echo "Issuing ECR Auth credential to ${QVI_AID} for person ${PERSON_NAME}"
echo "Using LE alias: ${LE_ALIAS}"
echo "Using registry: ${REGISTRY_NAME}"

docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh le/le-acdc-issue-ecr-auth.ts \
    'docker' \
    "${LE_ALIAS}" \
    "${LE_SALT}" \
    "${REGISTRY_NAME}" \
    "${ECR_AUTH_SCHEMA_SAID}" \
    "${QVI_AID}" \
    "${PERSON_AID}" \
    "${PERSON_NAME}" \
    "${PERSON_ECR}" \
    "${LE_CRED_SAID}" \
    "/task-data/ecr-auth-credential-info.json"
