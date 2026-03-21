#!/bin/bash
# tsx-script-runner.sh - Receives a script in /tsx/sig-wallet/src and runs it with tsx
#   after sourcing the environment variables

set -e
source /vlei/workshop-env-vars.sh

# take first arg as typescript file name and pass the rest to the script
tsx "/vlei/sig-wallet/src/tasks/$1" "${@:2}"
