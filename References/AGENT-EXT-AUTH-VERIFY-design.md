# Agent Extension Authorization & Verification Design
## Hybrid KERI Delegation + ECR Credential Approach

**Version:** 1.0  
**Date:** November 12, 2025  
**Status:** Design Specification  
**Author:** vLEI Implementation Team  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Trust Chain Design](#trust-chain-design)
5. [Authorization Proof Mechanism](#authorization-proof-mechanism)
6. [Credential Schemas](#credential-schemas)
7. [Verification Flow](#verification-flow)
8. [Implementation Requirements](#implementation-requirements)
9. [Comparison with Alternatives](#comparison-with-alternatives)
10. [References](#references)

---

## Executive Summary

This design document specifies how AI agents (e.g., `jupiterSellerAgent`, `tommyBuyerAgent`) can be cryptographically proven to be authorized by specific Official Organizational Role (OOR) holders while maintaining full vLEI ecosystem compliance.

### Key Innovation

**Hybrid Approach:** Combines KERI AID delegation (for cryptographic authorization proof) with ECR credentials (for vLEI compliance and trust chain).

### Design Principles

1. **Cryptographic Proof**: Agent authorization must be cryptographically verifiable
2. **Specific Attribution**: Must prove authorization by specific OOR holder, not just LE
3. **vLEI Compliance**: Must follow official GLEIF vLEI patterns
4. **Full Trust Chain**: Must chain back to GEDA root of trust
5. **Revocability**: Both delegation and credentials must be revocable

---

## Problem Statement

### The Authorization Challenge

**Question:** How can it be cryptographically proven that `jupiterSellerAgent` is authorized by the **specific** Chief Sales Officer (OOR holder), not just by Jupiter Knitting Company (LE)?

### Inadequate Solutions

#### Option 1: Pure ECR Credential (No Delegation)
```
LE → ECR AUTH → QVI → ECR → Agent
```

**Problem:** 
- ECR AUTH credentials are typically issued by Legal Entity Authorized Representatives (LARs)
- No cryptographic link to the **specific** OOR holder who authorized the agent
- Cannot prove which person authorized which agent

**Example Issue:**
```
Jupiter Knitting has 3 sales officers:
  - Chief Sales Officer (John Smith)
  - Regional Sales Officer (Jane Doe)
  - Junior Sales Officer (Bob Lee)

If jupiterSellerAgent has only an ECR credential issued by LE:
  → Cannot prove it was authorized by John Smith specifically
  → Could have been authorized by any LAR
  → No cryptographic proof of specific authorization
```

#### Option 2: Pure KERI Delegation (Current Implementation)
```
OOR Holder AID → Delegates to → Agent AID
```

**Problem:**
- Agent has no vLEI credential
- Not part of official vLEI trust chain
- Sally cannot verify using standard vLEI credential verification
- Non-standard approach

**What it provides:**
- ✅ Cryptographic proof of delegation in KEL
- ❌ No vLEI credential
- ❌ Not GLEIF compliant

---

## Solution Architecture

### Hybrid Approach: Best of Both Worlds

Combine KERI AID delegation (for authorization proof) with ECR credentials (for vLEI compliance).

```
┌─────────────────────────────────────────────────────────────────┐
│                    GEDA (GLEIF Root)                            │
│                         │                                        │
│                    QVI Credential                               │
│                         │                                        │
│                   LE Credential                                 │
│                    (Jupiter)                                     │
│                         │                                        │
│                   OOR AUTH Credential                           │
│                         │                                        │
│    ┌───────────────────────────────────────────┐               │
│    │   OOR Credential                          │               │
│    │   (Chief Sales Officer: John Smith)       │               │
│    │   Holder: EOORHolder123...                │               │
│    └───────────────────────────────────────────┘               │
│                         │                                        │
│                         ├──► [KEL Event] ◄──┐                  │
│                         │    Delegation      │                  │
│                         │    Seal: EDel456...│                  │
│                         │                     │                  │
│                         │         Cryptographic Authorization   │
│                         │         Proof (in OOR Holder's KEL)   │
│                         │                                        │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ ECR AUTH Credential                                     │ │
│    │ Issuer: OOR Holder (John Smith)                        │ │
│    │ Recipient: QVI                                          │ │
│    │ Chains to: OOR Credential                              │ │
│    │ Contains:                                               │ │
│    │   - oorHolderAID: EOORHolder123...                     │ │
│    │   - agentAID: EAgent789...                             │ │
│    │   - delegationSeal: EDel456... ◄─────────────┐        │ │
│    │   - capabilities: [...]                       │        │ │
│    └───────────────────────────────────────────────│────────┘ │
│                         │                           │          │
│                         │                    References KEL    │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ ECR Credential (Agent Credential)                       │ │
│    │ Issuer: QVI                                             │ │
│    │ Recipient: Agent (EAgent789...)                         │ │
│    │ Chains to: LE Credential                                │ │
│    │ Contains:                                                │ │
│    │   - authorizedBy:                                        │ │
│    │       oorHolderAID: EOORHolder123...                    │ │
│    │       oorHolderName: "John Smith"                       │ │
│    │       oorHolderRole: "Chief Sales Officer"              │ │
│    │       delegationSeal: EDel456... ◄──────────────────────┤ │
│    │   - authorizationCredential: EECRAuth...                │ │
│    │   - capabilities: [...]                                  │ │
│    └──────────────────────────────────────────────────────────┘ │
│                                                                  │
│ Agent: jupiterSellerAgent                                       │
│   - AID: EAgent789... (Delegated from EOORHolder123...)        │
│   - ECR Credential: Full vLEI trust chain                      │
│   - Authorization: Cryptographically proven via KEL            │
└──────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **KERI Delegation** (Layer 1: Cryptographic Authorization)
   - OOR Holder delegates AID to Agent
   - Creates delegation seal in KEL
   - Provides cryptographic proof

2. **ECR AUTH Credential** (Layer 2: vLEI Authorization)
   - Issued by OOR Holder to QVI
   - References delegation seal
   - Authorizes ECR issuance

3. **ECR Credential** (Layer 3: Agent Identity)
   - Issued by QVI to Agent
   - Contains full authorization chain
   - Verifiable vLEI credential

---

## Trust Chain Design

### Complete Trust Chain

```
Level 1: Root of Trust
┌─────────────────────────────────────┐
│ GEDA (GLEIF External AID)           │
│ - Root of vLEI ecosystem            │
│ - Issues QVI credentials            │
└─────────────────────────────────────┘
            │
            │ delegates & issues
            ▼
Level 2: Qualified vLEI Issuer
┌─────────────────────────────────────┐
│ QVI Credential                      │
│ - Holder: QVI AID                   │
│ - Issuer: GEDA                      │
│ - Authority to issue LE & Role creds│
└─────────────────────────────────────┘
            │
            │ issues
            ▼
Level 3: Legal Entity
┌─────────────────────────────────────┐
│ LE Credential                       │
│ - Holder: Jupiter Knitting          │
│ - LEI: 3358004DXAMRWRUIYJ05        │
│ - Issuer: QVI                       │
└─────────────────────────────────────┘
            │
            │ authorizes & issues
            ▼
Level 4a: Official Role Authorization
┌─────────────────────────────────────┐
│ OOR AUTH Credential                 │
│ - Holder: QVI                       │
│ - Issuer: LE (Jupiter)              │
│ - Authorizes OOR issuance           │
└─────────────────────────────────────┘
            │
            │ issues
            ▼
Level 4b: Official Organizational Role
┌─────────────────────────────────────┐
│ OOR Credential                      │
│ - Holder: Chief Sales Officer       │
│ - Person: John Smith                │
│ - Role: Chief Sales Officer         │
│ - Issuer: QVI                       │
│ - AID: EOORHolder123...             │
└─────────────────────────────────────┘
            │
            │ ┌─────────────────────────┐
            │ │ [KEL Event]             │
            ├─┤ Delegation              │
            │ │ Seal: EDel456...        │
            │ │ Delegate: EAgent789...  │
            │ └─────────────────────────┘
            │
            │ issues (references delegation)
            ▼
Level 5a: Agent Authorization
┌─────────────────────────────────────┐
│ ECR AUTH Credential                 │
│ - Holder: QVI                       │
│ - Issuer: OOR Holder (John Smith)  │
│ - Chains to: OOR Credential         │
│ - Agent: EAgent789...               │
│ - Delegation Seal: EDel456...      │
│ - Capabilities: [...]               │
└─────────────────────────────────────┘
            │
            │ issues
            ▼
Level 5b: Agent Identity
┌─────────────────────────────────────┐
│ ECR Credential (Agent)              │
│ - Holder: jupiterSellerAgent        │
│ - AID: EAgent789...                 │
│ - Issuer: QVI                       │
│ - Chains to: LE Credential          │
│ - Authorized by: EOORHolder123...   │
│ - Delegation Seal: EDel456...      │
│ - Auth Credential: EECRAuth...     │
└─────────────────────────────────────┘
```

### Chain Verification Points

1. **ECR → LE**: Direct chain via `ri` field
2. **ECR → OOR Holder**: Via `authorizedBy` and delegation seal
3. **ECR AUTH → OOR**: Direct chain via `ri` field
4. **OOR → LE**: Via OOR AUTH → LE chain
5. **LE → QVI → GEDA**: Standard vLEI chain

---

## Authorization Proof Mechanism

### Three-Layer Proof System

#### Layer 1: KERI KEL Delegation (Cryptographic)

**Delegation Event in OOR Holder's KEL:**

```json
{
  "v": "KERI10JSON00011c_",
  "t": "dip",
  "d": "EDel456xyz...",
  "i": "EAgent789abc...",
  "s": "0",
  "kt": "1",
  "k": ["DAgent_Signing_Key_123..."],
  "nt": "1",
  "n": ["EAgent_Next_Key_456..."],
  "bt": "0",
  "b": [],
  "c": [],
  "a": [],
  "di": "EOORHolder123..."
}
```

**Fields:**
- `t: "dip"` - Delegated Inception Event
- `d` - Delegation Seal SAID (EDel456xyz...)
- `i` - Agent AID (EAgent789abc...)
- `di` - Delegator AID (EOORHolder123... = Chief Sales Officer)

**What this proves:**
- ✅ OOR Holder's private keys signed this event
- ✅ Event is in OOR Holder's KEL (immutable, ordered log)
- ✅ Specific Agent AID is authorized
- ✅ Timestamp of delegation
- ❌ Does NOT prove: vLEI credential chain

#### Layer 2: ECR AUTH Credential (Authorization)

**Purpose:** Bridge between KEL delegation and vLEI credential chain

**Issued by:** OOR Holder (John Smith) - holder of OOR credential  
**Recipient:** QVI  
**Chains to:** OOR Credential (not LE!)  

**Key Innovation:** References the KEL delegation seal

```json
{
  "v": "ACDC10JSON...",
  "d": "EECRAuth_SAID_123...",
  "i": "EQVI_AID_456...",
  "ri": "EOORCred_SAID_789...",
  "s": "EECRAuthSchema_SAID...",
  "a": {
    "d": "EData_SAID...",
    "i": "EAgent789abc...",
    "dt": "2025-11-12T10:00:00.000000+00:00",
    "LEI": "3358004DXAMRWRUIYJ05",
    
    "oorHolderAID": "EOORHolder123...",
    "oorHolderName": "John Smith",
    "oorHolderRole": "Chief Sales Officer",
    "oorCredential": "EOORCred_SAID_789...",
    
    "agentName": "Jupiter Seller Agent",
    "agentAID": "EAgent789abc...",
    "agentRole": "AI Trading Agent",
    "agentType": "AI_AUTONOMOUS",
    
    "delegationSeal": "EDel456xyz...",
    "delegatedAt": "2025-11-12T10:00:00.000000+00:00",
    
    "capabilities": [
      "execute_trades",
      "sign_contracts",
      "access_inventory",
      "negotiate_prices"
    ],
    
    "constraints": {
      "maxTransactionValue": "1000000 USD",
      "approvalRequired": true,
      "geographicScope": ["North America", "Europe"]
    }
  }
}
```

**What this proves:**
- ✅ OOR Holder issued this credential (signed with OOR AID keys)
- ✅ Chains to OOR Credential
- ✅ References KEL delegation seal
- ✅ Specifies agent capabilities and constraints
- ✅ Part of vLEI credential chain

#### Layer 3: ECR Credential (Agent Identity)

**Issued by:** QVI  
**Recipient:** Agent (jupiterSellerAgent)  
**Chains to:** LE Credential  

```json
{
  "v": "ACDC10JSON...",
  "d": "EECRAgent_SAID_999...",
  "i": "EAgent789abc...",
  "ri": "ELECred_SAID_555...",
  "s": "EECRSchema_SAID...",
  "a": {
    "d": "EData_SAID...",
    "i": "ELE_AID_Jupiter...",
    "dt": "2025-11-12T10:15:00.000000+00:00",
    "LEI": "3358004DXAMRWRUIYJ05",
    
    "agentName": "Jupiter Seller Agent",
    "agentRole": "AI Trading Agent",
    "agentType": "AI_AUTONOMOUS",
    "agentDescription": "Autonomous trading agent for textile sales",
    
    "authorizedBy": {
      "oorHolderAID": "EOORHolder123...",
      "oorHolderName": "John Smith",
      "oorHolderRole": "Chief Sales Officer",
      "oorCredential": "EOORCred_SAID_789...",
      "delegationSeal": "EDel456xyz...",
      "delegatedAID": "EAgent789abc..."
    },
    
    "authorizationCredential": "EECRAuth_SAID_123...",
    
    "capabilities": [
      "execute_trades",
      "sign_contracts",
      "access_inventory",
      "negotiate_prices"
    ],
    
    "constraints": {
      "maxTransactionValue": "1000000 USD",
      "approvalRequired": true,
      "geographicScope": ["North America", "Europe"],
      "operatingHours": "24/7",
      "escalationRequired": ["contract_value > 500000"]
    },
    
    "validFrom": "2025-11-12T10:15:00.000000+00:00",
    "validUntil": "2026-11-12T10:15:00.000000+00:00"
  }
}
```

**What this proves:**
- ✅ QVI issued this credential (QVI has authority from GEDA)
- ✅ Chains to LE Credential
- ✅ Contains complete authorization chain
- ✅ References ECR AUTH credential
- ✅ References delegation seal
- ✅ Full vLEI trust chain to GEDA

---

## Answer to Your Question

**Q: How can it be proved that jupiterSellerAgent is authorized by the Chief Sales Officer?**

**A: Through THREE linked proofs:**

1. **KEL Delegation Event** (Cryptographic)
   - Signed by Chief Sales Officer's keys
   - In Chief Sales Officer's KEL
   - Explicitly delegates to Agent AID

2. **ECR AUTH Credential** (Organizational)
   - Issued by Chief Sales Officer (holder of OOR credential)
   - References KEL delegation seal
   - Chains to OOR credential

3. **ECR Credential** (Final)
   - References both delegation seal and ECR AUTH
   - Verifiable back to GEDA
   - Sally can verify complete chain

**All three must be valid for the agent to be trusted.**

---

## Implementation Requirements

### Keep Existing (Phase 1 - DONE)
- ✅ OOR Holder delegates to Agent AID
- ✅ Creates delegation seal in KEL
- ✅ Agent AID is delegated AID

### Add New (Phase 2-4 - TO DO)

#### Phase 2: ECR AUTH Credential
1. Create ECR AUTH schema
2. OOR Holder issues ECR AUTH to QVI
3. QVI admits ECR AUTH

#### Phase 3: ECR Credential
1. Create ECR Agent schema
2. QVI issues ECR to Agent
3. Agent admits ECR

#### Phase 4: Sally Verifier
1. Add ECR schema recognition
2. Implement KEL delegation verification
3. Implement authorization chain verification

---

## References

- **GLEIF vLEI Workshop**: https://github.com/GLEIF-IT/vlei-hackathon-2025-workshop
- **GLEIF vLEI Trainings**: https://github.com/GLEIF-IT/vlei-trainings
- **Sally Verifier**: https://github.com/GLEIF-IT/sally
- **vLEI Ecosystem Governance Framework**: https://www.gleif.org/en/vlei/introducing-the-vlei-ecosystem-governance-framework

---

**Document Status:** Ready for Implementation  
**Next Steps:** Begin Phase 2 implementation (ECR AUTH credential creation)
