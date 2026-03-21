#!/bin/bash
# le-registry-create.sh - Create a registry in LE AID
# Usage: le-registry-create.sh [alias]
#   alias: Optional unique alias for the LE AID (defaults to 'le')

set -e

# Accept optional alias parameter
LE_ALIAS=${1:-"le"}

echo "Creating registry in LE AID"
echo "Using LE alias: ${LE_ALIAS}"
source ./task-scripts/workshop-env-vars.sh

# create ACDC registry with unique name per organization
REGISTRY_NAME="${LE_ALIAS}-oor-registry"
echo "Creating registry: ${REGISTRY_NAME}"

docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh le/le-registry-create.ts \
    'docker' \
    "${LE_SALT}" \
    "${LE_ALIAS}" \
    "${REGISTRY_NAME}" \
    "/task-data/le-registry-info.json"
