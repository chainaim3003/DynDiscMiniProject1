#!/bin/bash
# qvi-acdc-issue-le.sh - Issue LE Credential from QVI to LE AID

# Create LE ACDC Credential
# This script issues the LE credential from the QVI AID to the LE AID

set -e

echo "Creating LE ACDC Credential"
source ./task-scripts/workshop-env-vars.sh

# Check if required info files exist
if [ ! -f "./task-data/le-info.json" ]; then
    echo "Required LE AID info file not found. Please run le-aid-create.sh first."
    exit 1
fi
if [ ! -f "./task-data/qvi-registry-info.json" ]; then
    echo "Required QVI registry info file not found. Please run qvi-registry-create.sh first."
    exit 1
fi
if [ ! -f "./task-data/qvi-credential-info.json" ]; then
    echo "Required QVI credential info file not found. Please run geda-acdc-issue-qvi.sh first."
    exit 1
fi

# get LE AID from le-info.json
LE_AID=$(jq -r '.aid' ./task-data/le-info.json)

# LEI for LE credential - accept as parameter or use default
LE_LEI="${1:-254900OPPU84GM83MG36}"

# Issue the LE credential
if [ -n "$1" ]; then
    echo "Issuing LE credential to ${LE_AID} with LEI ${LE_LEI} (from parameter)"
else
    echo "Issuing LE credential to ${LE_AID} with LEI ${LE_LEI} (default - hardcoded)"
fi
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh qvi/qvi-acdc-issue-le.ts \
    'docker' \
    'qvi' \
    "${QVI_SALT}" \
    "${LE_AID}" \
    "${LE_LEI}" \
    "/task-data/le-credential-info.json"
