#!/bin/bash
# qvi-acdc-present-qvi.sh - Present the QVI ACDC to the verifier (Sally)

set -e

CRED_SAID=$(jq -r .said ./task-data/qvi-credential-info.json)
echo "Presenting QVI ACDC to verifier Credential SAID: ${CRED_SAID}"
source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh qvi/qvi-acdc-present-qvi.ts \
    'docker' \
    "${QVI_SALT}" \
    "${QVI_AID_NAME}" \
    "${CRED_SAID}" \
    "${VERIFIER_AID}"