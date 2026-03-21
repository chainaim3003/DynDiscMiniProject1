## Step-by-Step Implementation Guide

### Phase 1: Setup (One-Time)

#### Step 1: Create Directory Structure

```bash
cd /path/to/vLEIWorkLinux1

# Create TypeScript task directories
mkdir -p sig-wallet/src/tasks/agent
mkdir -p task-scripts/agent

# Create Sally extension directory
mkdir -p config/verifier-sally/custom-sally

# Verify
ls -la sig-wallet/src/tasks/ | grep agent
ls -la task-scripts/ | grep agent
ls -la config/verifier-sally/ | grep custom-sally
```

#### Step 2: Copy TypeScript Files

```bash
# Copy files from AGENT-TYPESCRIPT-COMPLETE.md to:
# sig-wallet/src/tasks/person/person-delegate-agent-create.ts
# sig-wallet/src/tasks/person/person-approve-agent-delegation.ts
# sig-wallet/src/tasks/agent/agent-aid-delegate-finish.ts
# sig-wallet/src/tasks/agent/agent-oobi-resolve-qvi.ts
# sig-wallet/src/tasks/agent/agent-oobi-resolve-le.ts
# sig-wallet/src/tasks/agent/agent-oobi-resolve-verifier.ts
# sig-wallet/src/tasks/agent/agent-verify-delegation.ts
```

#### Step 3: Create Shell Scripts

```bash
# Create person scripts
cat > task-scripts/person/person-delegate-agent-create.sh << 'EOF'
#!/bin/bash
source ./task-scripts/tsx-script-runner.sh
run_task "person/person-delegate-agent-create.ts" "$@"
EOF

cat > task-scripts/person/person-approve-agent-delegation.sh << 'EOF'
#!/bin/bash
source ./task-scripts/tsx-script-runner.sh
run_task "person/person-approve-agent-delegation.ts" "$@"
EOF

# Create agent scripts
cat > task-scripts/agent/agent-aid-delegate-finish.sh << 'EOF'
#!/bin/bash
source ./task-scripts/tsx-script-runner.sh
run_task "agent/agent-aid-delegate-finish.ts" "$@"
EOF

cat > task-scripts/agent/agent-oobi-resolve-qvi.sh << 'EOF'
#!/bin/bash
source ./task-scripts/tsx-script-runner.sh
run_task "agent/agent-oobi-resolve-qvi.ts" "$@"
EOF

cat > task-scripts/agent/agent-oobi-resolve-le.sh << 'EOF'
#!/bin/bash
source ./task-scripts/tsx-script-runner.sh
run_task "agent/agent-oobi-resolve-le.ts" "$@"
EOF

cat > task-scripts/agent/agent-oobi-resolve-verifier.sh << 'EOF'
#!/bin/bash
source ./task-scripts/tsx-script-runner.sh
run_task "agent/agent-oobi-resolve-verifier.ts" "$@"
EOF

cat > task-scripts/agent/agent-verify-delegation.sh << 'EOF'
#!/bin/bash
source ./task-scripts/tsx-script-runner.sh
run_task "agent/agent-verify-delegation.ts" "$@"
EOF

# Make executable
chmod +x task-scripts/person/*.sh
chmod +x task-scripts/agent/*.sh
```

#### Step 4: Copy Sally Extension Files

```bash
# Copy Python files from AGENT-SALLY-PYTHON-COMPLETE.md to:
# config/verifier-sally/custom-sally/agent_verifying.py
# config/verifier-sally/custom-sally/handling_ext.py
# config/verifier-sally/custom-sally/__init__.py
# config/verifier-sally/entry-point-extended.sh

chmod +x config/verifier-sally/entry-point-extended.sh
```

#### Step 5: Update Docker Compose

```bash
# Backup original
cp docker-compose.yml docker-compose.yml.backup

# Edit verifier service in docker-compose.yml
# Change:
#   - ./config/verifier-sally/entry-point.sh:/sally/entry-point.sh
# To:
#   - ./config/verifier-sally/entry-point-extended.sh:/sally/entry-point.sh
#
# Add new volume mount:
#   - ./config/verifier-sally/custom-sally:/sally/custom-sally
```

#### Step 6: Deploy

```bash
# Stop existing containers
./stop.sh

# Rebuild TypeScript container
docker compose build tsx-shell

# Deploy all services
./deploy.sh

# Verify Sally loaded extensions
docker logs vleiworklinux1-verifier-1 | grep "Custom extensions"
# Should see:
# ✓ agent_verifying.py installed
# ✓ Custom extensions installed successfully
```

### Phase 2: Agent Delegation Execution

Prerequisite: You must have already run `./run-all-buyerseller-2.sh` to create:
- GEDA, QVI, LE (Jupiter Knitting), OOR Holder (Chief Sales Officer)

#### Step 1: Create Agent Delegation Request

```bash
docker exec tsx_shell bash -c "
  cd /vlei && \
  ./task-scripts/person/person-delegate-agent-create.sh \
    docker \
    AgentPass123 \
    /task-data \
    Jupiter_Chief_Sales_Officer \
    jupiterSellerAgent
"

# Verify output file
docker exec tsx_shell cat /task-data/jupiterSellerAgent-delegate-info.json
```

#### Step 2: OOR Holder Approves Delegation

```bash
docker exec tsx_shell bash -c "
  cd /vlei && \
  ./task-scripts/person/person-approve-agent-delegation.sh \
    docker \
    DoNotUseThis \
    Jupiter_Chief_Sales_Officer \
    /task-data/jupiterSellerAgent-delegate-info.json
"
```

#### Step 3: Agent Completes Delegation

```bash
docker exec tsx_shell bash -c "
  cd /vlei && \
  ./task-scripts/agent/agent-aid-delegate-finish.sh \
    docker \
    AgentPass123 \
    jupiterSellerAgent \
    /task-data/Jupiter_Chief_Sales_Officer-info.json \
    /task-data/jupiterSellerAgent-delegate-info.json \
    /task-data/jupiterSellerAgent-info.json
"

# Verify output
docker exec tsx_shell cat /task-data/jupiterSellerAgent-info.json
```

#### Step 4: Agent Resolves OOBIs

```bash
# Get QVI OOBI
QVI_OOBI=$(docker exec tsx_shell cat /task-data/qvi-info.json | jq -r '.oobi')

# Resolve QVI
docker exec tsx_shell bash -c "
  cd /vlei && \
  ./task-scripts/agent/agent-oobi-resolve-qvi.sh \
    docker \
    AgentPass123 \
    jupiterSellerAgent \
    \"$QVI_OOBI\"
"

# Get LE OOBI
LE_OOBI=$(docker exec tsx_shell cat /task-data/Jupiter_Knitting_Company-info.json | jq -r '.oobi')

# Resolve LE
docker exec tsx_shell bash -c "
  cd /vlei && \
  ./task-scripts/agent/agent-oobi-resolve-le.sh \
    docker \
    AgentPass123 \
    jupiterSellerAgent \
    \"$LE_OOBI\"
"

# Resolve Verifier
docker exec tsx_shell bash -c "
  cd /vlei && \
  ./task-scripts/agent/agent-oobi-resolve-verifier.sh \
    docker \
    AgentPass123 \
    jupiterSellerAgent
"
```

#### Step 5: Verify Delegation via Sally

```bash
docker exec tsx_shell bash -c "
  cd /vlei && \
  ./task-scripts/agent/agent-verify-delegation.sh \
    /task-data \
    jupiterSellerAgent \
    Jupiter_Chief_Sales_Officer
"

# Expected output:
# ============================================================
# SALLY VERIFICATION RESULT
# ============================================================
# {
#   "valid": true,
#   "agent_aid": "EAgent...",
#   "oor_holder_aid": "ECqsqWY...",
#   "verification": {
#     "delegation_valid": true,
#     "oor_credential_valid": true,
#     "oor_credential_said": "EAILKU2...",
#     "le_lei": "3358004DXAMRWRUIYJ05",
#     "qvi_aid": "EMK1ees...",
#     "geda_aid": "EECxLWG..."
#   }
# }
# ============================================================
# ✓ Agent delegation verified successfully
```

### Phase 3: Automated Workflow Script

Create: `run-agent-delegation-org1.sh`

```bash
#!/bin/bash
set -e

echo "========================================"
echo "  Agent Delegation Workflow"
echo "  Organization: Jupiter Knitting"
echo "========================================"

AGENT_NAME="jupiterSellerAgent"
OOR_HOLDER_NAME="Jupiter_Chief_Sales_Officer"
AGENT_PASSCODE="JupiterAgentPass123"
OOR_HOLDER_PASSCODE="DoNotUseThis"
DATA_DIR="/task-data"

echo "[1/5] Creating Agent Delegation Request..."
docker exec tsx_shell bash -c "
  cd /vlei && \
  ./task-scripts/person/person-delegate-agent-create.sh \
    docker \
    $AGENT_PASSCODE \
    $DATA_DIR \
    $OOR_HOLDER_NAME \
    $AGENT_NAME
"

echo "[2/5] OOR Holder Approving Delegation..."
docker exec tsx_shell bash -c "
  cd /vlei && \
  ./task-scripts/person/person-approve-agent-delegation.sh \
    docker \
    $OOR_HOLDER_PASSCODE \
    $OOR_HOLDER_NAME \
    $DATA_DIR/${AGENT_NAME}-delegate-info.json
"

echo "[3/5] Agent Completing Delegation..."
docker exec tsx_shell bash -c "
  cd /vlei && \
  ./task-scripts/agent/agent-aid-delegate-finish.sh \
    docker \
    $AGENT_PASSCODE \
    $AGENT_NAME \
    $DATA_DIR/${OOR_HOLDER_NAME}-info.json \
    $DATA_DIR/${AGENT_NAME}-delegate-info.json \
    $DATA_DIR/${AGENT_NAME}-info.json
"

echo "[4/5] Agent Resolving OOBIs..."
QVI_OOBI=$(docker exec tsx_shell cat $DATA_DIR/qvi-info.json | jq -r '.oobi')
docker exec tsx_shell bash -c "
  cd /vlei && \
  ./task-scripts/agent/agent-oobi-resolve-qvi.sh \
    docker \
    $AGENT_PASSCODE \
    $AGENT_NAME \
    \"$QVI_OOBI\"
"

if docker exec tsx_shell test -f "$DATA_DIR/Jupiter_Knitting_Company-info.json"; then
  LE_OOBI=$(docker exec tsx_shell cat $DATA_DIR/Jupiter_Knitting_Company-info.json | jq -r '.oobi')
else
  LE_OOBI=$(docker exec tsx_shell cat $DATA_DIR/le-info.json | jq -r '.oobi')
fi

docker exec tsx_shell bash -c "
  cd /vlei && \
  ./task-scripts/agent/agent-oobi-resolve-le.sh \
    docker \
    $AGENT_PASSCODE \
    $AGENT_NAME \
    \"$LE_OOBI\"
"

docker exec tsx_shell bash -c "
  cd /vlei && \
  ./task-scripts/agent/agent-oobi-resolve-verifier.sh \
    docker \
    $AGENT_PASSCODE \
    $AGENT_NAME
"

echo "[5/5] Verifying Agent Delegation via Sally..."
docker exec tsx_shell bash -c "
  cd /vlei && \
  ./task-scripts/agent/agent-verify-delegation.sh \
    $DATA_DIR \
    $AGENT_NAME \
    $OOR_HOLDER_NAME
"

echo ""
echo "========================================"
echo "✓ Agent Delegation Complete"
echo "========================================"
docker exec tsx_shell cat $DATA_DIR/${AGENT_NAME}-info.json | jq '.'
```

Make executable:
```bash
chmod +x run-agent-delegation-org1.sh
```

Run:
```bash
./run-agent-delegation-org1.sh
```

### Testing

#### Test Sally Endpoint Directly

```bash
# Get AIDs
AGENT_AID=$(docker exec tsx_shell cat /task-data/jupiterSellerAgent-info.json | jq -r '.aid')
OOR_AID=$(docker exec tsx_shell cat /task-data/Jupiter_Chief_Sales_Officer-info.json | jq -r '.aid')

# Call Sally API
curl -X POST http://127.0.0.1:9723/verify/agent-delegation \
  -H "Content-Type: application/json" \
  -d "{
    \"agent_aid\": \"$AGENT_AID\",
    \"oor_holder_aid\": \"$OOR_AID\",
    \"verify_oor_credential\": true
  }" | jq '.'
```

#### Verify Agent KEL

```bash
docker exec vlei_shell kli status \
  --name verifier \
  --alias verifier \
  --passcode 4TBjjhmKu9oeDp49J7Xdy \
  --prefix "$AGENT_AID"

# Look for "delegator" field showing OOR Holder AID
```

### Troubleshooting

#### Issue: Sally extensions not loading

```bash
# Check mount
docker exec vleiworklinux1-verifier-1 ls -la /sally/custom-sally

# Check installation
docker exec vleiworklinux1-verifier-1 ls -la /usr/local/lib/python3.12/site-packages/custom_sally/

# Check logs
docker logs vleiworklinux1-verifier-1 | grep -A5 "Installing custom"
```

#### Issue: Delegation fails

```bash
# Verify OOR Holder approved
docker logs tsx_shell | grep "approved delegation"

# Re-run approval
docker exec tsx_shell bash -c "
  cd /vlei && \
  ./task-scripts/person/person-approve-agent-delegation.sh ...
"
```

### Quick Reference

```bash
# Create agent
./task-scripts/person/person-delegate-agent-create.sh docker <pass> /task-data <oor> <agent>

# Approve
./task-scripts/person/person-approve-agent-delegation.sh docker <pass> <oor> <delegate-info>

# Finish
./task-scripts/agent/agent-aid-delegate-finish.sh docker <pass> <agent> <oor-info> <delegate-info> <output>

# Verify
./task-scripts/agent/agent-verify-delegation.sh /task-data <agent> <oor>

# Full workflow
./run-agent-delegation-org1.sh
```

### Sally API Reference

**Endpoint:** `POST http://127.0.0.1:9723/verify/agent-delegation`

**Request:**
```json
{
  "agent_aid": "EAgent...",
  "oor_holder_aid": "EC7pC...",
  "verify_oor_credential": true
}
```

**Response (Success):**
```json
{
  "valid": true,
  "agent_aid": "EAgent...",
  "oor_holder_aid": "EC7pC...",
  "verification": {
    "delegation_valid": true,
    "oor_credential_valid": true,
    "oor_credential_said": "EAILKU2...",
    "le_lei": "3358004DXAMRWRUIYJ05",
    "qvi_aid": "EMK1ees...",
    "geda_aid": "EECxLWG..."
  },
  "timestamp": "2025-11-10T22:00:00.000000+00:00"
}
```

**Response (Failure):**
```json
{
  "valid": false,
  "agent_aid": "EAgent...",
  "oor_holder_aid": "EC7pC...",
  "error": "Delegator mismatch. Expected EC7pC..., got EWrong...",
  "timestamp": "2025-11-10T22:00:00.000000+00:00"
}
```
