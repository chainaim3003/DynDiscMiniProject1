# Invoice Credential Issuance - Complete Solution

## Problem Summary

The invoice credential issuance was failing with "fetch failed" errors when trying to resolve the schema OOBI. The key insight from analyzing the official vLEI documentation is that **schema OOBI resolution happens through KERIA, not directly from the host**.

## Root Causes Identified

### 1. SignifyClient Instance Isolation

Each SignifyClient instance (identified by its unique BRAN/passcode) maintains its own schema cache. When you:
- Created the GEDA/QVI/LE using passscode A → Schema OOBIs resolved for client A
- Created agent with unique BRAN B → Client B has NO schemas resolved

**Official documentation quote:**
> "Every client that needs to issue credentials must resolve the schema OOBI."

### 2. Schema Structure Issues

Comparing your invoice schema with official vLEI schemas (OOR, LE, EventPass), the schema needs:
- Proper SAIDification of ALL nested `$id` fields
- Rules section with proper structure

### 3. Docker Network Resolution

The test from `localhost:7723` can succeed while KERIA (inside Docker) fails to reach `http://schema:7723`. This is because:
- Host uses `localhost:7723` (port mapping)
- KERIA uses Docker DNS to resolve `schema` hostname

## Solution Components

### File 1: `fix-invoice-credential-complete.sh`

A comprehensive diagnostic and fix script that:
1. Checks/creates schema file with proper structure
2. SAIDifies the schema (using KLI or TypeScript)
3. Creates SAID-named copy for vLEI-server
4. Restarts schema container
5. Verifies accessibility from BOTH host AND KERIA
6. Provides detailed diagnostics

### File 2: `invoice-acdc-issue-v2.ts`

Fixed TypeScript that:
1. Uses multi-strategy schema resolution:
   - Direct resolution with agent client
   - Check schema cache
   - Use shared passcode client (same as OOR/LE issuance)
   - Direct schema fetch
2. Follows exact pattern from official documentation (102_20_KERIA_Signify_Credential_Issuance.md)
3. Provides detailed error diagnostics

### File 3: `self-attested-invoice-FIXED.json`

Properly structured schema matching official vLEI schema patterns.

## How to Apply the Fix

### Option A: Quick Fix (Recommended)

```bash
# On your server
cd ~/projects/algoTitanV61/LegentvLEI

# 1. Copy the fix script and make executable
chmod +x fix-invoice-credential-complete.sh

# 2. Run the fix
./fix-invoice-credential-complete.sh

# 3. Update 4C script to use V2 issuance
# Change: invoice-acdc-issue-self-attested-only.ts
# To:     invoice-acdc-issue-v2.ts

# 4. Re-run workflow
./run-all-buyerseller-4C-with-agents.sh
```

### Option B: Manual Fix

1. **Ensure schema file has proper SAIDs:**
   ```bash
   # Check current SAID
   jq '.["$id"]' schemas/self-attested-invoice.json
   
   # Should not be empty or null
   ```

2. **Create SAID-named copy:**
   ```bash
   SAID=$(jq -r '.["$id"]' schemas/self-attested-invoice.json)
   cp schemas/self-attested-invoice.json "schemas/${SAID}.json"
   ```

3. **Restart schema container:**
   ```bash
   docker compose restart schema
   sleep 15
   ```

4. **Verify from KERIA (CRITICAL):**
   ```bash
   docker compose exec keria wget -qO- http://schema:7723/oobi/$SAID | head -c 100
   ```

5. **Update TypeScript to use V2 script** or add this to existing script:
   ```typescript
   // Add shared passcode resolution as fallback
   const SHARED_PASSCODE = "AEqhkGD_S_SRiO-oeJA4x";
   
   // Try shared client for schema resolution
   const sharedClient = await getOrCreateClient(SHARED_PASSCODE, env);
   await resolveOobi(sharedClient, schemaOobi, 'invoice-schema');
   
   // Then resolve with agent client
   await resolveOobi(agentClient, schemaOobi, 'invoice-schema');
   ```

## Understanding the Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HOST MACHINE                              │
│                                                                  │
│  ┌─────────────┐     curl localhost:7723    ┌─────────────┐    │
│  │   Your      │ ──────────────────────────>│   Schema    │    │
│  │   Script    │         (Works!)           │  Container  │    │
│  └─────────────┘                            │  (port 7723)│    │
│                                             └─────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Port Mapping (7723:7723)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DOCKER NETWORK (vlei_workshop)               │
│                                                                  │
│  ┌─────────────┐                           ┌─────────────┐     │
│  │   KERIA     │   http://schema:7723      │   Schema    │     │
│  │  Container  │ ───────────────────────── │  Container  │     │
│  │             │   (Might FAIL - DNS!)     │  hostname:  │     │
│  └─────────────┘                           │   "schema"  │     │
│        │                                   └─────────────┘     │
│        │ SignifyClient                                          │
│        │ schema.resolve()                                       │
│        ▼                                                        │
│  ┌─────────────┐                                                │
│  │   tsx-shell │   Your TypeScript runs here                   │
│  │  Container  │   But SignifyClient talks to KERIA            │
│  └─────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Your TypeScript code calls `client.oobis().resolve()`, which sends a request to KERIA. KERIA then makes the HTTP request to `http://schema:7723/oobi/...`. So the schema container must be reachable from KERIA, not from the host.

## Credential Verification

After successful issuance, the credential can be verified using:

### IPEX Grant (Official Pattern)

```typescript
import { Serder } from "signify-ts";

// After issuance, grant to verifier
const [grant, gsigs, gend] = await client.ipex().grant({
    senderName: issuerAidAlias,
    acdc: new Serder(credential.sad),
    iss: new Serder(credential.iss),
    anc: new Serder(credential.anc),
    ancAttachment: credential.ancatc,
    recipient: verifierAid,
    datetime: createTimestamp(),
});

await client.ipex().submitGrant(
    issuerAidAlias,
    grant,
    gsigs,
    gend,
    [verifierAid]
);
```

### Sally Verification

Sally can verify credentials if:
1. The credential schema is known to Sally
2. The trust chain is complete (agent → LE → QVI → GEDA)

For self-attested credentials, the trust derives from the agent delegation chain, not from credential edges.

## Schema Requirements Summary

Based on official vLEI schemas:

| Field | Required | Notes |
|-------|----------|-------|
| `$id` | Yes | Top-level SAID |
| `$schema` | Yes | JSON Schema version |
| `title` | Yes | Human-readable name |
| `description` | Yes | Description |
| `type` | Yes | Always "object" |
| `credentialType` | Yes | Unique credential type name |
| `properties.a.oneOf[1].$id` | Yes | SAID for attributes block |
| `properties.r.oneOf[1].$id` | Optional | SAID for rules block if present |

## Files Created

1. `fix-invoice-credential-complete.sh` - Complete diagnostic and fix script
2. `invoice-acdc-issue-v2.ts` - Fixed TypeScript with multi-strategy schema resolution
3. `self-attested-invoice-FIXED.json` - Properly structured schema template

## Next Steps

1. Run `fix-invoice-credential-complete.sh` on your server
2. Verify schema is accessible from KERIA
3. Update your workflow to use `invoice-acdc-issue-v2.ts`
4. Re-run the 4C workflow

If issues persist, check:
- Docker network configuration: `docker network inspect vlei_workshop`
- Schema container logs: `docker compose logs schema`
- KERIA container logs: `docker compose logs keria`
