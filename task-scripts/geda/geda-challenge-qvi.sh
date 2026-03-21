#!/bin/bash
# geda-challenge-qvi.sh - Generate challenge from GEDA to QVI

set -e

echo "GEDA generating challenge for QVI"
source ./task-scripts/workshop-env-vars.sh

# Check if required info files exist
if [ ! -f "./task-data/qvi-info.json" ]; then
    echo "Required QVI info file not found. Please run qvi-aid-delegate-finish.sh first."
    exit 1
fi

# get QVI AID from qvi-info.json
QVI_AID=$(jq -r '.aid' ./task-data/qvi-info.json)

echo "GEDA challenging QVI AID: ${QVI_AID}"

# Generate challenge and store info
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh geda/geda-challenge-qvi.ts \
    'docker' \
    "${GEDA_SALT}" \
    "${QVI_AID}" \
    "/task-data/geda-challenge-info.json"

echo "Challenge generated successfully"

