#!/bin/bash
# geda-respond-qvi-challenge.sh - GEDA responds to QVI challenge

set -e

echo "GEDA responding to QVI challenge"
source ./task-scripts/workshop-env-vars.sh

# Check if required info files exist
if [ ! -f "./task-data/qvi-challenge-info.json" ]; then
    echo "Required QVI challenge info file not found. Please run qvi-challenge-geda.sh first."
    exit 1
fi
if [ ! -f "./task-data/qvi-info.json" ]; then
    echo "Required QVI info file not found. Please run qvi-aid-delegate-finish.sh first."
    exit 1
fi

# get QVI AID from qvi-info.json
QVI_AID=$(jq -r '.aid' ./task-data/qvi-info.json)

# get challenge words from qvi-challenge-info.json
CHALLENGE_WORDS=$(jq -r '.words | join(" ")' ./task-data/qvi-challenge-info.json)

echo "GEDA responding to QVI AID: ${QVI_AID}"

# Respond to the challenge
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh geda/geda-respond-qvi-challenge.ts \
    'docker' \
    "${GEDA_SALT}" \
    "${QVI_AID}" \
    "${CHALLENGE_WORDS}"

echo "GEDA challenge response completed"


