#!/bin/bash
# le-oobi-resolve-qvi.sh - Resolve the QVI OOBI for LE AID

set -e
QVI_OOBI=$(jq -r .oobi ./task-data/qvi-info.json)

echo "LE Resolving QVI OOBI: ${QVI_OOBI}"

source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh common/oobi-resolve.ts \
    'docker' \
    "${LE_SALT}" \
    "${QVI_OOBI}"
