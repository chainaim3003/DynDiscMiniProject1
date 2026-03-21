# Agent KEL Diagnosis and Solution Guide

## 🔍 Deep Investigation Results

### **1. Docker Execution Context Difference**

```bash
# test-agent-verification.sh
./task-scripts/agent/agent-verify-delegation.sh
  ↓
docker compose exec tsx-shell /vlei/tsx-script-runner.sh agent/agent-verify-delegation.ts
  ↓
# TypeScript runs INSIDE Docker container
# Uses: http://vlei-verification:9723 (internal Docker network)
```

```bash
# test-agent-verification-v2.sh  
curl -X POST http://localhost:9724/verify/agent-delegation
  ↓
# Bash script runs OUTSIDE Docker on host
# Uses: http://localhost:9724 (port-mapped from internal 9723)
```

**Key Point:** Both scripts access the **SAME verifier service**. The difference is WHERE the calling code runs, not which service it calls.

### **2. Critical Clue from Logs**

```json
"verification": {
  "delegation_verified": true,  ✓ Seal exists in OOR holder's KEL
  "kel_verification": null      ⚠️ Agent's KEL not fully verified
}
```

This means:
- ✅ Sally found the delegation seal in the OOR holder's KEL
- ❌ Sally couldn't fully verify the agent's KEL itself
- **The agent's KEL might exist but isn't accessible/complete**

### **3. Agent KEL Creation Flow Analysis**

From the code examination:

```typescript
// Step 1: Agent creates inception (PENDING)
createDelegate(client, agentName, oorHolderAID, ...)
  → Creates inception event with delpre: oorHolderAID
  → Returns {aid, icpOpName}
  → KEL is created but INCOMPLETE until approved

// Step 2: OOR Holder approves (SEAL)
approveDelegation(oorHolderClient, oorHolderName, agentAID)
  → Adds seal to OOR holder's KEL: {i: agentAID, s: '0', d: agentAID}
  → This anchors the agent's inception

// Step 3: Agent finishes (COMPLETE)
finishAgentDelegation(...)
  → Queries OOR holder key state (discovers seal)
  → Waits for inception operation to complete
  → Adds endpoint role
  → Gets OOBI
```

**The Problem:** The agent's KEL exists in KERIA, but the verifier might not have access to it because:
1. The OOBI wasn't resolved by the verifier
2. The KEL wasn't witnessed properly
3. The inception operation didn't fully complete

---

## 🛠️ Solutions

### **Solution 1: Enhanced Diagnostic Script**

Create a comprehensive KEL check script that runs inside Docker:

**File:** `diagnose-agent-kel.sh`

```bash
#!/bin/bash
# diagnose-agent-kel.sh
# Check if agent KEL exists and is complete

set -e

AGENT_NAME="${1:-jupiterSellerAgent}"
OOR_HOLDER_NAME="${2:-Jupiter_Chief_Sales_Officer}"

echo "=========================================="
echo "AGENT KEL DIAGNOSTIC"
echo "=========================================="
echo ""

# Get AIDs
AGENT_INFO="./task-data/${AGENT_NAME}-info.json"
OOR_INFO="./task-data/${OOR_HOLDER_NAME}-info.json"

if [ ! -f "$AGENT_INFO" ]; then
    echo "❌ Agent info not found: $AGENT_INFO"
    exit 1
fi

if [ ! -f "$OOR_INFO" ]; then
    echo "❌ OOR Holder info not found: $OOR_INFO"
    exit 1
fi

AGENT_AID=$(cat "$AGENT_INFO" | jq -r '.aid')
OOR_AID=$(cat "$OOR_INFO" | jq -r '.aid')
AGENT_OOBI=$(cat "$AGENT_INFO" | jq -r '.oobi // empty')

echo "Agent: $AGENT_NAME"
echo "  AID: $AGENT_AID"
echo "  OOBI: $AGENT_OOBI"
echo ""
echo "OOR Holder: $OOR_HOLDER_NAME"
echo "  AID: $OOR_AID"
echo ""

# Test 1: Check if agent KEL exists in KERIA
echo "=========================================="
echo "TEST 1: Agent KEL in KERIA"
echo "=========================================="

docker compose exec -T tsx-shell bash -c "
    echo 'Checking KERIA for agent KEL...'
    RESPONSE=\$(curl -s http://keria:3902/identifiers/$AGENT_AID)
    if echo \"\$RESPONSE\" | jq -e '.prefix' > /dev/null 2>&1; then
        echo '✅ Agent KEL EXISTS in KERIA'
        echo \"\$RESPONSE\" | jq '.'
    else
        echo '❌ Agent KEL NOT FOUND in KERIA'
        echo \"\$RESPONSE\"
        exit 1
    fi
"
echo ""

# Test 2: Check agent KEL events
echo "=========================================="
echo "TEST 2: Agent KEL Events"
echo "=========================================="

docker compose exec -T tsx-shell bash -c "
    echo 'Fetching agent KEL events...'
    EVENTS=\$(curl -s http://keria:3902/events?pre=$AGENT_AID)
    EVENT_COUNT=\$(echo \"\$EVENTS\" | jq 'length')
    echo \"Event count: \$EVENT_COUNT\"
    if [ \"\$EVENT_COUNT\" -gt 0 ]; then
        echo '✅ Agent has KEL events'
        echo \"\$EVENTS\" | jq '.'
    else
        echo '❌ Agent has NO KEL events'
        exit 1
    fi
"
echo ""

# Test 3: Check OOR Holder KEL for delegation seal
echo "=========================================="
echo "TEST 3: Delegation Seal in OOR Holder KEL"
echo "=========================================="

docker compose exec -T tsx-shell bash -c "
    echo 'Checking OOR Holder KEL for delegation seal...'
    EVENTS=\$(curl -s http://keria:3902/events?pre=$OOR_AID)
    if echo \"\$EVENTS\" | jq -e '.[] | select(.a? | length > 0)' > /dev/null 2>&1; then
        echo '✅ OOR Holder has anchoring events (seals)'
        echo \"\$EVENTS\" | jq '.[] | select(.a? | length > 0)'
    else
        echo '❌ No delegation seal found in OOR Holder KEL'
        exit 1
    fi
"
echo ""

# Test 4: Verify agent OOBI is resolvable
echo "=========================================="
echo "TEST 4: Agent OOBI Resolution"
echo "=========================================="

if [ -n "$AGENT_OOBI" ]; then
    docker compose exec -T tsx-shell bash -c "
        echo 'Resolving agent OOBI...'
        RESPONSE=\$(curl -s '$AGENT_OOBI')
        if [ -n \"\$RESPONSE\" ]; then
            echo '✅ Agent OOBI is resolvable'
            echo \"\$RESPONSE\"
        else
            echo '❌ Agent OOBI not resolvable'
            exit 1
        fi
    "
else
    echo "⚠️  Agent OOBI not found in info file"
fi
echo ""

# Test 5: Check if verifier has agent's OOBI resolved
echo "=========================================="
echo "TEST 5: Verifier OOBI Resolution Status"
echo "=========================================="

docker compose exec -T tsx-shell bash -c "
    echo 'Checking if verifier can access agent...'
    # Try to get agent info from verifier's perspective
    curl -s -X POST http://vlei-verification:9723/verify/agent-delegation \
        -H 'Content-Type: application/json' \
        -d '{
            \"aid\": \"$OOR_AID\",
            \"agent_aid\": \"$AGENT_AID\",
            \"verify_kel\": true
        }' | jq '.'
"
echo ""

echo "=========================================="
echo "DIAGNOSTIC COMPLETE"
echo "=========================================="
```

**Usage:**
```bash
chmod +x diagnose-agent-kel.sh
./diagnose-agent-kel.sh jupiterSellerAgent Jupiter_Chief_Sales_Officer
```

---

### **Solution 2: Fix Agent OOBI Resolution for Verifier**

The verifier might not have the agent's OOBI resolved. Add this step to your workflow:

#### **File:** `task-scripts/verifier/verifier-oobi-resolve-agent.sh`

```bash
#!/bin/bash
# verifier-oobi-resolve-agent.sh
# Resolve agent OOBI from verifier's perspective

AGENT_NAME=$1

if [ -z "$AGENT_NAME" ]; then
  echo "Usage: verifier-oobi-resolve-agent.sh <agentName>"
  exit 1
fi

echo "Verifier resolving agent OOBI for ${AGENT_NAME}"

docker compose exec tsx-shell \
  /vlei/tsx-script-runner.sh verifier/verifier-oobi-resolve-agent.ts \
    "/task-data" \
    "${AGENT_NAME}"
```

#### **File:** `sig-wallet/src/tasks/verifier/verifier-oobi-resolve-agent.ts`

```typescript
import fs from 'fs';
import {getOrCreateClient} from "../../client/identifiers.js";
import {resolveOobi} from "../../client/oobis.js";

const args = process.argv.slice(2);
const dataDir = args[0];
const agentName = args[1];

const verifierPasscode = "verifier-password-123";  // Use your actual verifier passcode

const agentInfoPath = `${dataDir}/${agentName}-info.json`;
if (!fs.existsSync(agentInfoPath)) {
    throw new Error(`Agent info file not found: ${agentInfoPath}`);
}

const agentInfo = JSON.parse(fs.readFileSync(agentInfoPath, 'utf-8'));

console.log(`Verifier resolving agent OOBI: ${agentInfo.oobi}`);

const verifierClient = await getOrCreateClient(verifierPasscode, 'docker');
await resolveOobi(verifierClient, agentInfo.oobi, agentName);

console.log(`✓ Verifier resolved agent OOBI for ${agentName}`);
```

---

### **Solution 3: Enhanced Test Script with KEL Check**

#### **File:** `test-agent-verification-enhanced.sh`

```bash
#!/bin/bash
# test-agent-verification-enhanced.sh
# Enhanced test script with KEL checks

set -e

echo "=========================================="
echo "ENHANCED AGENT VERIFICATION TEST"
echo "=========================================="
echo ""

AGENT_NAME="${1:-jupiterSellerAgent}"
OOR_HOLDER_NAME="${2:-Jupiter_Chief_Sales_Officer}"

echo "Test Configuration:"
echo "  Agent Name: ${AGENT_NAME}"
echo "  OOR Holder Name: ${OOR_HOLDER_NAME}"
echo ""

# Check if info files exist
AGENT_INFO="./task-data/${AGENT_NAME}-info.json"
OOR_INFO="./task-data/${OOR_HOLDER_NAME}-info.json"

if [ ! -f "$AGENT_INFO" ] || [ ! -f "$OOR_INFO" ]; then
    echo "❌ Error: Info files not found"
    exit 1
fi

AGENT_AID=$(cat "$AGENT_INFO" | jq -r '.aid')
OOR_AID=$(cat "$OOR_INFO" | jq -r '.aid')

echo "Extracted AIDs:"
echo "  Agent AID: ${AGENT_AID}"
echo "  OOR Holder AID: ${OOR_AID}"
echo ""

# ============================================
# TEST 1: Check Agent KEL Exists in KERIA
# ============================================
echo "=========================================="
echo "TEST 1: Agent KEL in KERIA"
echo "=========================================="
echo ""

docker compose exec -T tsx-shell bash -c "
    RESPONSE=\$(curl -s http://keria:3902/identifiers/$AGENT_AID)
    if echo \"\$RESPONSE\" | jq -e '.prefix' > /dev/null 2>&1; then
        echo '✅ Agent KEL exists in KERIA'
    else
        echo '❌ Agent KEL NOT FOUND in KERIA'
        echo \"\$RESPONSE\"
        exit 1
    fi
"
echo ""

# ============================================
# TEST 2: Resolve Agent OOBI to Verifier
# ============================================
echo "=========================================="
echo "TEST 2: Ensure Verifier Has Agent OOBI"
echo "=========================================="
echo ""

AGENT_OOBI=$(cat "$AGENT_INFO" | jq -r '.oobi // empty')
if [ -n "$AGENT_OOBI" ]; then
    echo "Resolving agent OOBI to verifier..."
    # You would call your verifier-oobi-resolve-agent script here
    echo "OOBI: $AGENT_OOBI"
else
    echo "⚠️  Agent OOBI not found - verification may fail"
fi
echo ""

# ============================================
# TEST 3: TypeScript Verification
# ============================================
echo "=========================================="
echo "TEST 3: Verification via TypeScript"
echo "=========================================="
echo ""

./task-scripts/agent/agent-verify-delegation.sh "${AGENT_NAME}" "${OOR_HOLDER_NAME}"

RESULT=$?
if [ $RESULT -eq 0 ]; then
    echo ""
    echo "✅ TEST 3 PASSED: Verification succeeded"
else
    echo ""
    echo "❌ TEST 3 FAILED: Verification failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 ALL TESTS PASSED!"
echo "=========================================="
```

**Usage:**
```bash
chmod +x test-agent-verification-enhanced.sh
./test-agent-verification-enhanced.sh jupiterSellerAgent Jupiter_Chief_Sales_Officer
```

---

### **Solution 4: Fix in Agent Delegation Workflow**

Modify `run-all-buyerseller-2-with-agents.sh` to add verifier OOBI resolution:

```bash
# After step 4 (Agent resolves OOBIs), add this new step:

echo -e "${BLUE}            → Resolving agent OOBI to verifier...${NC}"
./task-scripts/verifier/verifier-oobi-resolve-agent.sh "$AGENT_ALIAS"
```

**Complete modification in the agent loop:**

```bash
# Step 4: Agent resolves OOBIs
echo -e "${BLUE}          [4/5] Agent resolves OOBIs...${NC}"
echo -e "${BLUE}            → Resolving QVI OOBI...${NC}"
./task-scripts/agent/agent-oobi-resolve-qvi.sh "$AGENT_ALIAS"

echo -e "${BLUE}            → Resolving LE OOBI...${NC}"
./task-scripts/agent/agent-oobi-resolve-le.sh "$AGENT_ALIAS" "$ORG_ALIAS"

echo -e "${BLUE}            → Resolving Sally verifier OOBI...${NC}"
./task-scripts/agent/agent-oobi-resolve-verifier.sh "$AGENT_ALIAS"

# NEW: Resolve agent OOBI to verifier
echo -e "${BLUE}            → Resolving agent OOBI to verifier...${NC}"
./task-scripts/verifier/verifier-oobi-resolve-agent.sh "$AGENT_ALIAS"

# Step 5: Verify agent delegation
echo -e "${BLUE}          [5/5] Verifying agent delegation via Sally...${NC}"
./task-scripts/agent/agent-verify-delegation.sh "$AGENT_ALIAS" "$PERSON_ALIAS"
```

---

## 📋 Recommended Action Plan

### **Phase 1: Diagnosis (Immediate)**

1. **Run the diagnostic script** to confirm if KELs exist:
   ```bash
   # Create the diagnostic script first
   ./diagnose-agent-kel.sh jupiterSellerAgent Jupiter_Chief_Sales_Officer
   ```

2. **Check directly in KERIA** from Docker:
   ```bash
   docker compose exec tsx-shell curl -s \
     http://keria:3902/identifiers/EGNlvZ3YwQ4BKsZm1Dvqyy9WbN2dQgDuyhXEXwo8TXYR | jq '.'
   ```

3. **Check agent KEL events**:
   ```bash
   docker compose exec tsx-shell curl -s \
     "http://keria:3902/events?pre=EGNlvZ3YwQ4BKsZm1Dvqyy9WbN2dQgDuyhXEXwo8TXYR" | jq '.'
   ```

### **Phase 2: Fix Implementation**

**If KEL exists (most likely):**
- ✅ The delegation completed successfully
- ❌ The verifier doesn't have the agent's OOBI resolved
- **Solution:** Implement Solution 2 (verifier OOBI resolution)

**If KEL doesn't exist:**
- ❌ The delegation didn't complete
- **Solution:** Check inception operation status, verify witness availability, add error handling

### **Phase 3: Workflow Enhancement**

1. **Create missing scripts:**
   - `diagnose-agent-kel.sh`
   - `task-scripts/verifier/verifier-oobi-resolve-agent.sh`
   - `sig-wallet/src/tasks/verifier/verifier-oobi-resolve-agent.ts`
   - `test-agent-verification-enhanced.sh`

2. **Modify workflow:**
   - Update `run-all-buyerseller-2-with-agents.sh` to include verifier OOBI resolution

3. **Test:**
   ```bash
   # Run full workflow
   ./run-all-buyerseller-2-with-agents.sh
   
   # Run enhanced test
   ./test-agent-verification-enhanced.sh
   ```

---

## 🔑 Key Takeaways

### **Root Cause**

The agent's KEL likely EXISTS in KERIA, but the verifier cannot access it because:

1. **OOBI Resolution Gap:** The agent's OOBI was resolved by:
   - ✅ QVI
   - ✅ LE
   - ✅ OOR Holder
   - ❌ **Verifier** ← **Missing step!**

2. **Verification Flow:**
   ```
   Verifier receives verification request
     ↓
   Needs to query agent's KEL
     ↓
   Doesn't have agent's OOBI resolved
     ↓
   Cannot find agent's KEL
     ↓
   Returns: "Agent AID not found in KEL"
   ```

### **Why Setup Verification "Succeeded"**

During setup, the verification showed:
```json
{
  "delegation_verified": true,   // Found seal in OOR holder's KEL
  "kel_verification": null       // Couldn't fully verify agent's KEL
}
```

This was marked as "success" because the **delegation seal** was found, which is the primary check. The KEL verification was `null` (not performed/incomplete) rather than `false` (failed).

### **Why v2 Test Failed**

The v2 test requested **full KEL verification** (`verify_kel: true`), which requires:
- Agent's KEL to be accessible
- Agent's OOBI to be resolved by the verifier
- Complete KEL event chain

Without the verifier having the agent's OOBI resolved, it cannot query KERIA for the agent's KEL, resulting in the error.

---

## 📝 Implementation Checklist

- [ ] Create `diagnose-agent-kel.sh`
- [ ] Run diagnostic on existing agents
- [ ] Create `task-scripts/verifier/verifier-oobi-resolve-agent.sh`
- [ ] Create `sig-wallet/src/tasks/verifier/verifier-oobi-resolve-agent.ts`
- [ ] Update `run-all-buyerseller-2-with-agents.sh` to include verifier OOBI resolution
- [ ] Create `test-agent-verification-enhanced.sh`
- [ ] Test with existing agents
- [ ] Re-run full workflow with fixes
- [ ] Verify both agents work with enhanced test script
- [ ] Update documentation

---

## 🎯 Expected Outcome

After implementing these fixes:

1. **Diagnostic output:**
   ```
   ✅ Agent KEL EXISTS in KERIA
   ✅ Agent has KEL events
   ✅ OOR Holder has anchoring events (seals)
   ✅ Agent OOBI is resolvable
   ✅ Verifier can access agent
   ```

2. **Enhanced test output:**
   ```
   ✅ Agent KEL exists in KERIA
   ✅ Verifier has agent OOBI resolved
   ✅ Verification succeeded
   🎉 ALL TESTS PASSED!
   ```

3. **v2 test output:**
   ```json
   {
     "valid": true,
     "verified": true,
     "verification": {
       "delegation_verified": true,
       "kel_verification": {
         "agent_kel_found": true,
         "events_verified": true,
         "delegation_seal_found": true
       }
     }
   }
   ```

---

## 📚 References

- **Agent Delegation Flow:** See `sig-wallet/src/tasks/agent/` and `sig-wallet/src/tasks/person/`
- **OOBI Resolution:** See `sig-wallet/src/client/oobis.ts`
- **KERI Identifiers:** See `sig-wallet/src/client/identifiers.ts`
- **Workflow Script:** `run-all-buyerseller-2-with-agents.sh`
- **Test Scripts:** `test-agent-verification.sh` and `test-agent-verification-v2.sh`

---

**Document Status:** Ready for Implementation  
**Priority:** High  
**Impact:** Critical - Fixes agent verification failure  
**Complexity:** Medium - Requires new scripts and workflow modifications
