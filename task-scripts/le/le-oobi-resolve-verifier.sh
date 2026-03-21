#!/bin/bash
# le-oobi-resolve-verifier.sh - Resolve the verifier (Sally) OOBI for LE AID

set -e

echo "LE Resolving Verifier (Sally) OOBI: ${VERIFIER_OOBI}"

source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh common/oobi-resolve.ts \
    'docker' \
    "${LE_SALT}" \
    "${VERIFIER_OOBI}"
