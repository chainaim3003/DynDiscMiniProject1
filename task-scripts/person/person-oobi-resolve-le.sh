#!/bin/bash
# person-oobi-resolve-le.sh - Resolve LE OOBI for Person AID

set -e

LE_OOBI=$(jq -r '.oobi' ./task-data/le-info.json)

echo "Person Resolving LE OOBI: ${LE_OOBI}"

source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh common/oobi-resolve.ts \
    'docker' \
    "${PERSON_SALT}" \
    "${LE_OOBI}"
