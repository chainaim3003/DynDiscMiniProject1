
# create-qvi-aid.sh - Create QVI AID as delegated identifier from GEDA
# This script uses the headless SignifyTS wallet via tsx to create the QVI AID

set -e

echo "Creating QVI AID as a delegate of GEDA using SignifyTS and KERIA"

source ./task-scripts/workshop-env-vars.sh

# ensure ./task-data/geda-info.json exists
if [ ! -f "./task-data/geda-info.json" ]; then
    echo "Error: ./task-data/geda-info.json not found. Please run geda-aid-create.sh first."
    exit 1
fi

# Run the QVI AID creation script
docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh qvi/qvi-aid-delegate-create.ts 'docker' "${QVI_SALT}" "/task-data"

echo "QVI AID created successfully"
