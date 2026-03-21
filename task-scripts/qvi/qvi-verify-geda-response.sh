#!/bin/bash
# qvi-verify-geda-response.sh - QVI verifies GEDA challenge response

set -e

echo "QVI verifying GEDA challenge response"
source ./task-scripts/workshop-env-vars.sh

# Check if required info files exist
if [ ! -f "./task-data/qvi-challenge-info.json" ]; then
    echo "Required QVI challenge info file not found. Please run qvi-challenge-geda.sh first."
    exit 1
fi
if [ ! -f "./task-data/geda-info.json" ]; then
    echo "Required GEDA info file not found. Please run geda-aid-create.sh first."
    exit 1
fi

# get GEDA AID from geda-info.json
GEDA_AID=$(jq -r '.aid' ./task-data/geda-info.json)

# get challenge words from qvi-challenge-info.json
CHALLENGE_WORDS=$(jq -r '.words | join(" ")' ./task-data/qvi-challenge-info.json)

echo "QVI verifying GEDA AID: ${GEDA_AID}"

# Verify the challenge response
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh qvi/qvi-verify-geda-response.ts \
    'docker' \
    "${QVI_SALT}" \
    "${GEDA_AID}" \
    "${CHALLENGE_WORDS}"

echo "QVI challenge verification completed"


