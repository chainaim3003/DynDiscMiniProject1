#!/bin/bash
# qvi-oobi-resolve-person.sh - Resolve Person OOBI for QVI AID

set -e

PERSON_OOBI=$(jq -r '.oobi' ./task-data/person-info.json)

echo "QVI Resolving Person OOBI: ${PERSON_OOBI}"

source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh common/oobi-resolve.ts \
    'docker' \
    "${QVI_SALT}" \
    "${PERSON_OOBI}"
