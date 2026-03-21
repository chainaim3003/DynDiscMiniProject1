#!/bin/bash
# run-all.sh - Run all scripts to create GEDA and QVI AIDs

# GEDA and QVI Setup
./task-scripts/geda/geda-aid-create.sh
./task-scripts/verifier/recreate-with-geda-aid.sh
./task-scripts/qvi/qvi-aid-delegate-create.sh
./task-scripts/geda/geda-delegate-approve.sh
./task-scripts/qvi/qvi-aid-delegate-finish.sh
./task-scripts/geda/geda-oobi-resolve-qvi.sh

# GEDA and QVI challenge and response
./task-scripts/geda/geda-challenge-qvi.sh
./task-scripts/qvi/qvi-respond-geda-challenge.sh
./task-scripts/geda/geda-verify-qvi-response.sh
./task-scripts/qvi/qvi-challenge-geda.sh
./task-scripts/geda/geda-respond-qvi-challenge.sh
./task-scripts/qvi/qvi-verify-geda-response.sh

# QVI Credential
./task-scripts/geda/geda-registry-create.sh
./task-scripts/geda/geda-acdc-issue-qvi.sh # includes the IPEX Grant
./task-scripts/qvi/qvi-acdc-admit-qvi.sh
./task-scripts/qvi/qvi-oobi-resolve-verifier.sh
./task-scripts/qvi/qvi-acdc-present-qvi.sh

# LE identifier setup
./task-scripts/le/le-aid-create.sh
./task-scripts/le/le-oobi-resolve-qvi.sh
./task-scripts/qvi/qvi-oobi-resolve-le.sh

# LE credentials
./task-scripts/qvi/qvi-registry-create.sh
./task-scripts/qvi/qvi-acdc-issue-le.sh
./task-scripts/le/le-acdc-admit-le.sh
./task-scripts/le/le-oobi-resolve-verifier.sh
./task-scripts/le/le-acdc-present-le.sh

# Person identifier setup
./task-scripts/person/person-aid-create.sh
./task-scripts/person/person-oobi-resolve-le.sh
./task-scripts/le/le-oobi-resolve-person.sh
./task-scripts/qvi/qvi-oobi-resolve-person.sh
./task-scripts/person/person-oobi-resolve-qvi.sh
./task-scripts/person/person-oobi-resolve-verifier.sh

# Person OOR Credential setup
./task-scripts/le/le-registry-create.sh
./task-scripts/le/le-acdc-issue-oor-auth.sh
./task-scripts/qvi/qvi-acdc-admit-oor-auth.sh
./task-scripts/qvi/qvi-acdc-issue-oor.sh
./task-scripts/person/person-acdc-admit-oor.sh

# Person present OOR Credential to verifier (Sally)
./task-scripts/person/person-acdc-present-oor.sh

# Person and ECR Credential
#./task-scripts/le/le-acdc-issue-ecr-auth.sh
#./task-scripts/qvi/qvi-acdc-admit-ecr-auth.sh
#./task-scripts/qvi/qvi-acdc-issue-ecr.sh
#./task-scripts/person/person-acdc-admit-ecr.sh
exit 0


