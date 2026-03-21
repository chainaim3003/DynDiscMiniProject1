#!/bin/bash
# qvi-acdc-admit-oor-auth.sh - Admit the OOR Auth ACDC Credential

set -e

GRANT_SAID=$(jq -r .grantSaid ./task-data/oor-auth-credential-info.json)
LE_PREFIX=$(jq -r .aid ./task-data/le-info.json)
echo "Admitting OOR Auth ACDC with IPEX Grant SAID: ${GRANT_SAID}"
source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh qvi/qvi-acdc-admit-oor-auth.ts \
    'docker' \
    "${QVI_SALT}" \
    "${GRANT_SAID}" \
    "${LE_PREFIX}"
