#!/bin/bash
# qvi-acdc-issue-ecr.sh - Issue ECR Credential from QVI to Person AID

# Create ECR ACDC Credential
# This script issues the ECR credential from the QVI AID to the Person AID

set -e

echo "Creating ECR ACDC Credential"
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
if [ ! -f "./task-data/ecr-auth-credential-info.json" ]; then
    echo "Required ECR Auth credential info file not found. Please run le-acdc-issue-ecr-auth.sh first."
    exit 1
fi

# get Person AID from person-info.json
PERSON_AID=$(jq -r '.aid' ./task-data/person-info.json)

# get ECR Auth Credential SAID from ecr-auth-credential-info.json for the edge
ECR_AUTH_CRED_SAID=$(jq -r '.said' ./task-data/ecr-auth-credential-info.json)

# Sample person data for ECR credential (same as ECR Auth)
PERSON_NAME="John Smith"
PERSON_ECR="Project Manager"

# Issue the ECR credential
echo "Issuing ECR credential to ${PERSON_AID} for person ${PERSON_NAME}"
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh qvi/qvi-acdc-issue-ecr.ts \
    'docker' \
    'qvi' \
    "${QVI_SALT}" \
    "qvi-le-registry" \
    "${ECR_SCHEMA_SAID}" \
    "${PERSON_AID}" \
    "${PERSON_NAME}" \
    "${PERSON_ECR}" \
    "${ECR_AUTH_CRED_SAID}" \
    "/task-data/ecr-credential-info.json"
