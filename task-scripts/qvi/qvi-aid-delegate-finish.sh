#!/bin/bash
# qvi-finish-delegation.sh - Finish delegation from GEDA to QVI AID

set -e

echo "Finishing delegation from GEDA to QVI AID"

source ./task-scripts/workshop-env-vars.sh

docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh qvi/qvi-aid-delegate-finish.ts \
    "docker" \
    "${QVI_SALT}" \
    "qvi" \
    "/task-data/geda-info.json" \
    "/task-data/qvi-delegate-info.json" \
    "/task-data/qvi-info.json" \
    "/task-data"

echo "GEDA -> QVI AID delegation complete"
