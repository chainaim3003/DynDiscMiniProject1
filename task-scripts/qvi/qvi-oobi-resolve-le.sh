#!/bin/bash
# qvi-oobi-resolve-le.sh - Resolve the LE OOBI for QVI AID

set -e
LE_OOBI=$(jq -r .oobi ./task-data/le-info.json)

echo "QVI Resolving LE OOBI: ${LE_OOBI}"

source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh common/oobi-resolve.ts \
    'docker' \
    "${QVI_SALT}" \
    "${LE_OOBI}"
