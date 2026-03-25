# A2A Negotiation + Dynamic Discounting — Module Overview

**Project:** DynDiscMiniProject1 / A2A / js  
**Stack:** Node.js · TypeScript · @a2a-js/sdk · Groq LLM · ACTUS REST  

---

## What This Module Does

Automates the full trade lifecycle between two AI agents:

```
Price Negotiation  →  Invoice  →  Dynamic Discounting  →  ACTUS Simulation
```

**Seller:** Jupiter Knitting Company (port 8080)  
**Buyer:** Tommy Hilfiger Europe B.V. (port 9090)

---

## Agents & Identity

| Agent | Organisation | LEI | Port |
|-------|-------------|-----|------|
| Jupiter Seller Agent | JUPITER KNITTING COMPANY | 3358004DXAMRWRUIYJ05 | 8080 |
| Tommy Buyer Agent | TOMMY HILFIGER EUROPE B.V. | 54930012QJWZMYHNJW95 | 9090 |

Both agents carry GLEIF vLEI credentials verified through the chain:  
`GLEIF_ROOT → QVI → Legal Entity → OOR Holder → Agent`

---

## Configuration

### Seller (private, hardcoded)
| Parameter | Value |
|-----------|-------|
| marginPrice | ₹350/unit — absolute floor, never accepted below |
| minProfitMargin | ₹5 — minimum profit above margin |
| targetPrice | ₹385/unit (10% above margin) |
| maxRounds | 3 |
| paymentTerms | Net 30 days |
| DD proposedEarlyPayDays | 10 days |
| DD safetyFactor | 0.5 (gives away at most 50% of profit as discount) |
| DD hurdleRate | 7.5% annualised |

### Buyer (private, hardcoded)
| Parameter | Value |
|-----------|-------|
| maxBudget | ₹400/unit — hard ceiling |
| targetPrice | ₹330/unit |
| quantity | 2,000 units |
| maxRounds | 3 |
| initialOfferRange | ₹250–₹320 (randomised each session) |

---

## Input — What Starts the Module

The CLI operator triggers everything. No external module input needed.

```
# Start negotiation (price optional)
start negotiation 340

# After negotiation completes and DD offer appears in Buyer terminal:
dd accept                   → accept seller's proposed date
dd accept YYYY-MM-DD        → choose your own early payment date
dd reject                   → decline discount, pay full on due date
```

---

## Message Flow (Full Sequence)

```
BUYER  → SELLER   OFFER               Opening price offer
SELLER → BUYER    COUNTER_OFFER       Up to 3 rounds of LLM/rule-based counters
BUYER  → SELLER   COUNTER_OFFER       (repeated per round)
SELLER → BUYER    ACCEPT_OFFER        Either side accepts
BUYER  → SELLER   ACCEPT_OFFER        Bilateral confirmation
BUYER  → SELLER   PURCHASE_ORDER      Formalises the deal
SELLER → BUYER    INVOICE             Full amount (subtotal + 18% GST, Net 30)
SELLER → BUYER    DD_OFFER            Early payment discount offer
BUYER  → SELLER   DD_ACCEPT           Buyer's chosen settlement date (via CLI)
SELLER → ACTUS    [4 HTTP calls]      Register contract & simulate
SELLER → BUYER    DD_INVOICE          Final discounted amount + ACTUS result
```

If all 3 rounds expire with no deal:
```
BUYER  → SELLER   ESCALATION_NOTICE   Human review report saved to disk
```

---

## Output — What This Module Produces

### 1. INVOICE
Sent by seller after Purchase Order received.

| Field | Example |
|-------|---------|
| invoiceId | INV-1774467379154 |
| terms.pricePerUnit | ₹370 |
| terms.quantity | 2000 |
| terms.subtotal | ₹740,000 |
| terms.tax (18% GST) | ₹133,200 |
| terms.total | ₹873,200 |
| paymentTerms | Net 30 days |
| deliveryDate | 60 days from session start |

### 2. DD_OFFER
Sent by seller immediately after Invoice (only if profit > 0).

| Field | Example |
|-------|---------|
| invoiceDate | 2026-03-26 |
| dueDate | 2026-04-25 (invoiceDate + 30) |
| originalTotal | ₹873,200 |
| maxDiscountRate | 0.027027 (~2.7%) |
| proposedSettlementDate | 2026-04-05 (invoiceDate + 10) |
| discountAtProposedDate.daysEarly | 20 |
| discountAtProposedDate.appliedRate | 0.018018 (~1.8%) |
| discountAtProposedDate.discountedAmount | ₹857,490 |
| discountAtProposedDate.savingAmount | ₹15,710 |

### 3. DD_INVOICE (Final Output)
Sent by seller after receiving DD_ACCEPT and running ACTUS simulation.

| Field | Example |
|-------|---------|
| originalTotal | ₹873,200 |
| discountedTotal | ₹857,490 |
| savingAmount | ₹15,710 |
| appliedRate | 0.018018 |
| settlementDate | 2026-04-05 (buyer's chosen date) |
| dueDate | 2026-04-25 |
| actusContractId | INV-1774467379154 |
| actusScenarioId | dd_scenario_79154 |
| actusSimulationStatus | SUCCESS / FAILED |

---

## Dynamic Discounting Formula

**Safe DD Rate** (how much seller can offer):
```
profitPerUnit  = agreedPrice - marginPrice
maxRate        = profitPerUnit / agreedPrice
safeDDRate     = maxRate × 0.5              ← seller keeps half the profit
```

**Linear Discount** (applied rate scales with days paid early):
```
totalDays      = dueDate - invoiceDate       (30 days)
daysEarly      = dueDate - settlementDate
appliedRate    = safeDDRate × (daysEarly / totalDays)
discountedAmt  = originalTotal × (1 - appliedRate)
```

Pay on due date → 0% discount. Pay on invoice date → full safeDDRate.

---

## ACTUS Integration

After DD_ACCEPT, seller fires 4 sequential HTTP POSTs:

| Step | Endpoint | Purpose |
|------|----------|---------|
| 1 | POST /addReferenceIndex | Flat daily cash series (seller revenue) |
| 2 | POST /addEarlySettlementModel | LINEAR discount model with safeDDRate |
| 3 | POST /addScenario | Links reference index + early settlement |
| 4 | POST rf2/scenarioSimulation | PAM contract (RPA role), simulate to dueDate |

**Default URLs:**
- Risk Service: `http://34.203.247.32:8082`
- ACTUS: `http://34.203.247.32:8083`

Override via env vars `ACTUS_RISK_URL` and `ACTUS_URL`.

> ACTUS failure is non-blocking. DD_INVOICE is still sent with `actusSimulationStatus: "FAILED"`.

---

## Decision Making

Both agents use **Hybrid LLM + Rule-based**:
1. Groq `llama-3.3-70b-versatile` generates action + price + reasoning
2. Constraint validator checks the LLM output against hard limits
3. If invalid → rule-based fallback kicks in

**Seller hard limits:**
- Never accept ≤ ₹355 (margin ₹350 + buffer ₹5)
- LLM is told floor is ₹355, not ₹350 (prevents zero-profit acceptance)
- Counter-offers must be monotonically decreasing

**Buyer hard limits:**
- Never accept > ₹400
- Counter-offers must be monotonically increasing
- Round 3 failure → escalation report

---

## Negotiation Status States

```
INITIATED → NEGOTIATING → ACCEPTED → COMPLETED → DD_COMPLETED
                               ↓
                           ESCALATED  (rounds exhausted, gap remains)
                           REJECTED   (seller price below margin in final round)
```

---

## File Structure

```
js/src/
├── agents/
│   ├── seller-agent/index.ts     Seller logic, DD offer, ACTUS submit
│   └── buyer-agent/index.ts      Buyer logic, CLI DD interaction
├── shared/
│   ├── negotiation-types.ts      All message + state interfaces
│   ├── dd-calculator.ts          Pure math: safeDDRate + linearDiscount
│   ├── actus-client.ts           HTTP client for ACTUS (4-step flow)
│   ├── llm-client.ts             Groq API wrapper + prompt builder
│   └── logger.ts                 Coloured terminal output
└── cli.ts                        Readline CLI → buyer agent

agent-cards/
├── jupiterSellerAgent-card.json  Seller GLEIF/vLEI identity
└── tommyBuyerAgent-card.json     Buyer GLEIF/vLEI identity
```

---

## How to Run

```bash
# From: A2A/js/

npm run agents:seller          # Terminal 1
npm run agents:buyer           # Terminal 2
npx tsx src/cli.ts http://localhost:9090   # Terminal 3

# In Terminal 3:
start negotiation 340
# (watch Terminals 1 & 2 for negotiation rounds)
# when DD Offer appears in Terminal 2...
dd accept
# or
dd accept 2026-04-10
```

---

## Integration Handoff Point

The **DD_INVOICE** is the final output of this module. Any downstream payment, settlement, or ledger module should consume:

- `discountedTotal` — actual amount buyer pays
- `settlementDate` — committed payment date
- `actusContractId` + `actusScenarioId` — for ACTUS cashflow correlation
- `actusSimulationStatus` — to know if ACTUS risk simulation succeeded
