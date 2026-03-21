#!/bin/bash
# qvi-registry-create.sh - Create a registry in QVI AID

set -e
echo "Creating registry in QVI AID"
source ./task-scripts/workshop-env-vars.sh

# create ACDC registry
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh qvi/qvi-registry-create.ts \
    'docker' \
    "${QVI_SALT}" \
    "qvi" \
    "qvi-le-registry" \
    "/task-data/qvi-registry-info.json"
