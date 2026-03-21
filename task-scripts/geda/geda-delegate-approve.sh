#!/bin/bash
# geda-approve-delegation.sh - Approve delegation from GEDA to QVI AID

set -e
echo "Approving delegation from GEDA to QVI AID"
source ./task-scripts/workshop-env-vars.sh

# approve delegation
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh geda/geda-delegate-approve.ts \
    'docker' \
    "${GEDA_SALT}" \
    "geda" \
    "/task-data/qvi-delegate-info.json"
