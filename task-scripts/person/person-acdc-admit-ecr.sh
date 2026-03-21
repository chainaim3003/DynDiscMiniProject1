#!/bin/bash
# person-acdc-admit-ecr.sh - Admit the ECR ACDC Credential

set -e

GRANT_SAID=$(jq -r .grantSaid ./task-data/ecr-credential-info.json)
QVI_PREFIX=$(jq -r .aid ./task-data/qvi-info.json)
echo "Admitting ECR ACDC with IPEX Grant SAID: ${GRANT_SAID}"
source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh person/person-acdc-admit-ecr.ts \
    'docker' \
    "${PERSON_SALT}" \
    "${GRANT_SAID}" \
    "${QVI_PREFIX}"
