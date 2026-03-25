# A2A Agentic Negotiation System - Quick Start Guide

## Overview

This implementation features **Hybrid AI negotiation agents** that use:
- 🧠 **LLM-based strategic reasoning** (GPT-4) for adaptive decision-making
- 🛡️ **Hard constraint validation** to protect business rules
- 🎲 **Randomized initial offers** for unpredictable negotiations
- 🤝 **Bilateral acceptance** (either party can accept at any round)
- 📊 **Comprehensive logging** of all negotiation rounds

## Architecture

```
Buyer Agent (Port 9090) ←→ A2A Messages ←→ Seller Agent (Port 8080)
       ↓                                           ↓
   LLM Decision                               LLM Decision
       ↓                                           ↓
 Constraint Check                           Constraint Check
   (Max Budget)                              (Min Margin)
```

## Prerequisites

1. **Node.js**: v18+ installed
2. **OpenAI API Key**: Get from https://platform.openai.com/api-keys
3. **Dependencies**: Installed via pnpm/npm

## Installation

### Step 1: Install Dependencies

```bash
cd C:\CHAINAIM3003\mcp-servers\algoTITANV6\Legent\A2A\js
pnpm install
# or
npm install
```

### Step 2: Configure Environment

```bash
# Copy the example environment file
copy .env.example .env

# Edit .env and add your OpenAI API key
notepad .env
```

Add your key:
```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx
```

## Running the System

### Option 1: Start Both Agents

**Terminal 1 - Seller Agent:**
```bash
pnpm run agents:seller
```

**Terminal 2 - Buyer Agent:**
```bash
pnpm run agents:buyer
```

**Terminal 3 - CLI to trigger negotiation:**
```bash
pnpm run a2a:cli http://localhost:9090
```

Then type:
```
start negotiation
```

### Option 2: Script Automation (Optional)

Create a PowerShell script to launch all three:

```powershell
# start-negotiation.ps1
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\CHAINAIM3003\mcp-servers\algoTITANV6\Legent\A2A\js; pnpm run agents:seller"
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\CHAINAIM3003\mcp-servers\algoTITANV6\Legent\A2A\js; pnpm run agents:buyer"
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\CHAINAIM3003\mcp-servers\algoTITANV6\Legent\A2A\js; pnpm run a2a:cli http://localhost:9090"
```

## Expected Behavior

### Negotiation Flow

```
ROUND 1:
  🛒 BUYER → SELLER: ₹285/unit (randomized initial offer)
  🏪 SELLER → BUYER: ₹430/unit (counter with margin protection)

ROUND 2:
  🛒 BUYER → SELLER: ₹325/unit (strategic concession)
  🏪 SELLER → BUYER: ₹380/unit (moving toward agreement)

ROUND 3:
  🛒 BUYER → SELLER: ✓ ACCEPT ₹380/unit
  🏪 SELLER → BUYER: ✓✓ AUTO-ACCEPT (bilateral acceptance)

POST-NEGOTIATION:
  📝 BUYER → SELLER: Purchase Order
  🧾 SELLER → BUYER: Invoice (with 18% tax)
```

### Console Output Example

```
═══════════════════════════════════════════════════════════
  NEGOTIATION SESSION STARTED
═══════════════════════════════════════════════════════════
  Negotiation ID: NEG-1737635000000
  Context ID: ctx-abc123xyz
  Start Time: 1/23/2026, 4:30:00 PM
═══════════════════════════════════════════════════════════

[ROUND 1] ────────────────────────────────────────────────
  
  🛒 BUYER → SELLER
     Action: INITIAL OFFER
     Price: ₹285/unit
     Strategy: Starting conservative, creating negotiation room

  🏪 SELLER → BUYER
     Action: COUNTER OFFER
     Previous: ₹285
     Offered: ₹430
     Change: ↑ +₹145 (+50.9%)
     Strategy: Anchoring high, protecting ₹350 margin
```

## Configuration

### Buyer Agent Settings

Located in `src/agents/buyer-agent/index.ts`:

```typescript
const BUYER_CONFIG = {
  maxBudget: 400,              // Never exceed this
  targetQuantity: 2000,         // Units to purchase
  maxRounds: 3,                 // Maximum negotiation rounds
  initialOfferRange: { 
    min: 250,                   // Minimum random start
    max: 320                    // Maximum random start
  },
  targetPrice: 330,             // Ideal outcome
};
```

### Seller Agent Settings

Located in `src/agents/seller-agent/index.ts`:

```typescript
const SELLER_CONFIG = {
  marginPrice: 350,             // NEVER go below this (cost)
  targetProfitPercentage: 0.1,  // 10% profit target
  maxRounds: 3,
  strategyParams: {
    minProfitMargin: 5,         // Minimum ₹5 profit per unit
  },
};
```

## How the Hybrid AI Works

### Buyer Agent Decision Process

1. **LLM Strategic Reasoning**:
   - Analyzes seller's offer
   - Considers round number (urgency)
   - Evaluates negotiation history
   - Proposes action: ACCEPT, COUNTER, or REJECT

2. **Hard Constraint Validation**:
   ```typescript
   if (decision.action === "ACCEPT") {
     if (sellerOffer > maxBudget) {
       // Override: Convert to COUNTER or REJECT
       decision.action = "COUNTER";
       decision.price = maxBudget;
     }
   }
   ```

3. **Rule-Based Fallback**:
   - If LLM fails or produces invalid response
   - Uses deterministic algorithm
   - Guarantees negotiation continues

### Seller Agent Decision Process

1. **LLM Strategic Reasoning**:
   - Analyzes buyer's offer vs. margin (₹350)
   - Calculates profit opportunity
   - Considers deal closure risk
   - Proposes counter-offer

2. **Hard Constraint Validation** (CRITICAL):
   ```typescript
   if (decision.price < marginPrice) {
     // NEVER allow below margin
     decision.price = marginPrice + minProfitMargin;
     decision.action = "COUNTER"; // or "REJECT" in final round
   }
   ```

3. **Rule-Based Fallback**:
   - Ensures margin is always protected
   - Calculates strategic concessions
   - Balances profit vs. deal probability

## Key Features

### ✅ Truly Agentic
- **Every negotiation is different** (randomized start)
- **LLM makes strategic decisions** based on context
- **No hardcoded negotiation paths**

### ✅ Bilateral Acceptance
```
If Buyer accepts Seller's offer:
  → Seller MUST auto-accept
  → Proceed to Purchase Order + Invoice

If Seller accepts Buyer's offer:
  → Buyer MUST auto-accept
  → Proceed to Purchase Order + Invoice
```

### ✅ Comprehensive Logging
- Every offer/counter logged
- Price movements tracked
- Gap analysis calculated
- Final summary with performance metrics

### ✅ A2A Protocol Compliance
- Uses `Message` with `DataPart` for structured data
- Maintains `contextId` throughout negotiation
- Proper multi-turn conversation handling
- Follows official A2A specification

## Troubleshooting

### Issue: "OpenAI API Error"

**Solution**: Check your `.env` file has valid `OPENAI_API_KEY`

```bash
# Verify environment variable is loaded
notepad .env
```

### Issue: "Cannot connect to seller/buyer"

**Solution**: Ensure both agents are running

```bash
# Check if ports are in use
netstat -ano | findstr :8080
netstat -ano | findstr :9090
```

Both should show LISTENING.

### Issue: "LLM decision invalid, using fallback"

**Cause**: LLM returned unexpected format or API failed

**Result**: System automatically uses rule-based fallback

**Impact**: Negotiation continues normally with deterministic logic

### Issue: "Negotiation failed - Max rounds exceeded"

**Cause**: Agents couldn't reach agreement in 3 rounds

**Possible reasons**:
- Buyer's budget too low (< ₹355)
- Seller's margin too high vs. buyer's randomized start
- LLM being too stubborn

**Solution**: Adjust configurations or run again (different random start)

## Advanced Usage

### Run Without LLM (Rule-Based Only)

Comment out the LLM call in both agents:

```typescript
// In makeNegotiationDecision method
// const llmDecision = await this.getLLMDecision(state);
// return this.applyConstraints(llmDecision, state);

// Use rule-based directly
return this.ruleBasedDecision(state);
```

### Modify Negotiation Parameters

Edit the config constants at the top of each agent file:

```typescript
// Buyer: More aggressive initial offers
initialOfferRange: { min: 300, max: 350 }

// Seller: Higher margin
marginPrice: 400
```

### Add More Rounds

```typescript
// Both agents
maxRounds: 5  // Allow 5 rounds instead of 3
```

## Testing Scenarios

### Scenario 1: Quick Agreement
- Buyer starts high (₹355+)
- Seller accepts immediately in Round 1
- Tests bilateral acceptance flow

### Scenario 2: Full Negotiation
- Buyer starts low (₹260-₹280)
- Seller counters high (₹420+)
- 3 rounds of back-and-forth
- Final acceptance in Round 3

### Scenario 3: Failed Negotiation
- Buyer max budget: ₹340
- Seller margin: ₹350
- No overlap → Rejection

### Scenario 4: Seller Accepts Mid-Negotiation
- Buyer offers ₹365 in Round 2
- Seller accepts (above margin)
- Tests truncated flow

## File Structure

```
src/
├── shared/
│   ├── negotiation-types.ts    # Type definitions
│   ├── llm-client.ts           # OpenAI integration
│   └── logger.ts               # Console logging
│
├── agents/
│   ├── buyer-agent/
│   │   └── index.ts           # Buyer agent implementation
│   └── seller-agent/
│       └── index.ts           # Seller agent implementation
│
└── cli.ts                     # A2A CLI client
```

## Next Steps

1. **Run multiple negotiations** to see variance
2. **Adjust configurations** to test different scenarios
3. **Monitor LLM decisions** in console output
4. **Experiment with prompts** in `llm-client.ts`
5. **Add database persistence** for negotiation history

## Support

For issues or questions:
1. Check console logs for detailed error messages
2. Verify both agents are running
3. Confirm OpenAI API key is valid
4. Review negotiation logs for decision reasoning

---

**Happy Negotiating! 🤝**
