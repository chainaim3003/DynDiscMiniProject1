#!/bin/bash
# qvi-acdc-admit-ecr-auth.sh - Admit the ECR Auth ACDC Credential

set -e

GRANT_SAID=$(jq -r .grantSaid ./task-data/ecr-auth-credential-info.json)
LE_PREFIX=$(jq -r .aid ./task-data/le-info.json)
echo "Admitting ECR Auth ACDC with IPEX Grant SAID: ${GRANT_SAID}"
source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh qvi/qvi-acdc-admit-ecr-auth.ts \
    'docker' \
    "${QVI_SALT}" \
    "${GRANT_SAID}" \
    "${LE_PREFIX}"
