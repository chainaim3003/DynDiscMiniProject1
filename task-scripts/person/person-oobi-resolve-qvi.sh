#!/bin/bash
# person-oobi-resolve-qvi.sh - Resolve QVI OOBI for Person AID

set -e

QVI_OOBI=$(jq -r '.oobi' ./task-data/qvi-info.json)

echo "Person Resolving QVI OOBI: ${QVI_OOBI}"

source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh common/oobi-resolve.ts \
    'docker' \
    "${PERSON_SALT}" \
    "${QVI_OOBI}"

