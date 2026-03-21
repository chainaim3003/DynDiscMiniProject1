#!/bin/bash
# person-oobi-resolve-verifier.sh - Resolve the verifier (Sally) OOBI for Person AID

set -e

echo "Person Resolving Verifier (Sally) OOBI: ${VERIFIER_OOBI}"

source ./task-scripts/workshop-env-vars.sh
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh common/oobi-resolve.ts \
    'docker' \
    "${PERSON_SALT}" \
    "${VERIFIER_OOBI}"
