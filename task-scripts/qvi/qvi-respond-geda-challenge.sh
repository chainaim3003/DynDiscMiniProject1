#!/bin/bash
# qvi-respond-geda-challenge.sh - QVI responds to GEDA challenge

set -e 

echo "QVI responding to GEDA challenge"
source ./task-scripts/workshop-env-vars.sh

# Check if required info files exist
if [ ! -f "./task-data/geda-challenge-info.json" ]; then
    echo "Required GEDA challenge info file not found. Please run geda-challenge-qvi.sh first."
    exit 1
fi
if [ ! -f "./task-data/geda-info.json" ]; then
    echo "Required GEDA info file not found. Please run geda-aid-create.sh first."
    exit 1
fi

# get GEDA AID from geda-info.json
GEDA_AID=$(jq -r '.aid' ./task-data/geda-info.json)
# get challenge words from geda-challenge-info.json
CHALLENGE_WORDS=$(jq -r '.words | join(" ")' ./task-data/geda-challenge-info.json)

echo "QVI responding to GEDA AID: ${GEDA_AID}"

# Respond to the challenge
docker compose exec tsx-shell \
    /vlei/tsx-script-runner.sh qvi/qvi-respond-geda-challenge.ts \
    'docker' \
    "${QVI_SALT}" \
    "${GEDA_AID}" \
    "${CHALLENGE_WORDS}"

echo "QVI challenge response completed"