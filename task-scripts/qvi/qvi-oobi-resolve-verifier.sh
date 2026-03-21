#!/bin/bash
# qvi-oobi-resolve-verifier.sh - Resolve the verifier (Sally) OOBI for QVI AID

set -e

echo "QVI Resolving Verifier (Sally) OOBI: ${VERIFIER_OOBI}"

source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh common/oobi-resolve.ts \
    'docker' \
    "${QVI_SALT}" \
    "${VERIFIER_OOBI}"
