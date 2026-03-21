#!/bin/bash
# geda-issue-qvi.sh - Issue QVI Credential from GEDA to QVI AID

# Create QVI ACDC Credential
# This script issues the QVI credential from the GEDA AID to the QVI AID

set -e

echo "Creating QVI ACDC Credential"
source ./task-scripts/workshop-env-vars.sh

# Check if required info files exist (geda-registry-info.json, qvi-info.json, geda-info.json)
if [ ! -f "./task-data/qvi-info.json" ]; then
    echo "Required QVI AID info file not found. Please run create-qvi-aid.sh first."
    exit 1
fi
if [ ! -f "./task-data/geda-registry-info.json" ]; then
    echo "Required info file not found. Please run create-qvi-aid.sh first."
    exit 1
fi

# get QVI AID from qvi-info.json
QVI_AID=$(jq -r '.aid' ./task-data/qvi-info.json)

# LEI for QVI credential - sample
GLEIF_LEI="506700GE1G29325QX363"

# Issue the QVI credential
echo "Issuing QVI credential to ${QVI_AID} with LEI ${GLEIF_LEI}"
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh geda/geda-acdc-issue-qvi.ts \
    'docker' \
    'geda' \
    "${GEDA_SALT}" \
    "${GEDA_REGISTRY}" \
    "${QVI_SCHEMA_SAID}" \
    "${QVI_AID}" \
    "${GLEIF_LEI}" \
    "/task-data/qvi-credential-info.json"
