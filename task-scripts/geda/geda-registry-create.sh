#!/bin/bash
# geda-registry-create.sh - Create a registry in GEDA AID

set -e
echo "Creating registry in GEDA AID"
source ./task-scripts/workshop-env-vars.sh

# create ACDC registry
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh geda/geda-registry-create.ts \
    'docker' \
    "${GEDA_SALT}" \
    "geda" \
    "geda-qvi-registry" \
    "/task-data/geda-registry-info.json"