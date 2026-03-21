#!/bin/bash
# qvi-acdc-admit-qvi.sh - Admit the QVI ACDC Credential

set -e

GRANT_SAID=$(jq -r .grantSAID ./task-data/qvi-credential-info.json)
GEDA_PREFIX=$(cat ./task-data/geda-aid.txt | tr -d " \t\n\r")
echo "Admitting QVI ACDC with IPEX Grant SAID: ${GRANT_SAID}"
source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh qvi/qvi-acdc-admit-qvi.ts \
    'docker' \
    "${QVI_SALT}" \
    "${GRANT_SAID}" \
    "${GEDA_PREFIX}"
