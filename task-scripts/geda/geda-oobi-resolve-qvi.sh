#!/bin/bash
# geda-oobi-resolve-qvi.sh - Refresh keystate for GEDA AID by resolving QVI AID to get end roles

set -e
QVI_OOBI=$(jq -r .oobi ./task-data/qvi-info.json)

echo "GEDA Resolving QVI OOBI: ${QVI_OOBI}"

source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh common/oobi-resolve.ts \
    'docker' \
    "${GEDA_SALT}" \
    "${QVI_OOBI}"