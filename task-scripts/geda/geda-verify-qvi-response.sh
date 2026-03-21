#!/bin/bash
# geda-verify-qvi-response.sh - GEDA verifies QVI challenge response

set -e

echo "GEDA verifying QVI challenge response"
source ./task-scripts/workshop-env-vars.sh

# Check if required info files exist
if [ ! -f "./task-data/geda-challenge-info.json" ]; then
    echo "Required GEDA challenge info file not found. Please run geda-challenge-qvi.sh first."
    exit 1
fi
if [ ! -f "./task-data/qvi-info.json" ]; then
    echo "Required QVI info file not found. Please run qvi-aid-delegate-finish.sh first."
    exit 1
fi

# get QVI AID from qvi-info.json
QVI_AID=$(jq -r '.aid' ./task-data/qvi-info.json)

# get challenge words from geda-challenge-info.json
CHALLENGE_WORDS=$(jq -r '.words | join(" ")' ./task-data/geda-challenge-info.json)

echo "GEDA verifying QVI AID: ${QVI_AID}"

# Verify the challenge response
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh geda/geda-verify-qvi-response.ts \
    'docker' \
    "${GEDA_SALT}" \
    "${QVI_AID}" \
    "${CHALLENGE_WORDS}"

echo "GEDA challenge verification completed"


