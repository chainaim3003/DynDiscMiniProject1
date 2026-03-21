#!/bin/bash
# recreate-with-geda-aid.sh - Recreate the Verifier container to have the correct GEDA AID
#   This GEDA AID is used to validate any presented AID credentials.

set -e
if [ ! -f ./task-data/geda-aid.txt ]; then
  echo "GEDA AID file not found: ./task-data/geda-aid.txt"
  export GEDA_PRE="EDDe8pD24aqd0dCZTQHaGpfcluPFD2ajGIY3ARgE5DD"
else
  GEDA_PRE=$(cat ./task-data/geda-aid.txt | tr -d " \t\n\r")
  export GEDA_PRE
fi

echo "Recreating Verifier container with GEDA AID Prefix: ${GEDA_PRE}"
docker compose -f docker-compose.yml down verifier
docker compose -f docker-compose.yml up -d verifier --wait
echo "Verifier container recreated."
