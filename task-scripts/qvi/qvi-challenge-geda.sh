#!/bin/bash
# qvi-challenge-geda.sh - Generate challenge from QVI to GEDA

set -e

echo "QVI generating challenge for GEDA"
source ./task-scripts/workshop-env-vars.sh

# Check if required info files exist
if [ ! -f "./task-data/geda-info.json" ]; then
    echo "Required GEDA info file not found. Please run geda-aid-create.sh first."
    exit 1
fi

# get GEDA AID from geda-info.json
GEDA_AID=$(jq -r '.aid' ./task-data/geda-info.json)

echo "QVI challenging GEDA AID: ${GEDA_AID}"

# Generate challenge and store info
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh qvi/qvi-challenge-geda.ts \
    'docker' \
    "${QVI_SALT}" \
    "${GEDA_AID}" \
    "/task-data/qvi-challenge-info.json"

echo "Challenge generated successfully"


