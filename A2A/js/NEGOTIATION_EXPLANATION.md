# AI Agent-to-Agent Negotiation System - Technical Explanation

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [How Negotiation Works](#how-negotiation-works)
4. [AI Decision Making](#ai-decision-making)
5. [Communication Protocol](#communication-protocol)
6. [Code Flow](#code-flow)
7. [Key Features](#key-features)

---

## 🎯 System Overview

This is an **autonomous AI agent negotiation system** where two independent AI agents (Buyer and Seller) negotiate a trade deal without human intervention.

### Real-World Use Case
Imagine two companies negotiating a purchase order:
- **Buyer Agent**: Represents a company wanting to buy 2,000 units of goods
- **Seller Agent**: Represents a supplier with margin constraints
- **Goal**: Reach a mutually beneficial price through strategic negotiation

### Key Innovation
- **Hybrid Intelligence**: Combines LLM (Large Language Model) reasoning with rule-based constraints
- **Autonomous**: Agents make decisions independently
- **Protocol-Based**: Uses A2A (Agent-to-Agent) communication standard

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER (CLI Interface)                      │
│                  npx tsx src/cli.ts                          │
└────────────────────────┬────────────────────────────────────┘
                         │ "start negotiation"
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              BUYER AGENT (Port 9090)                         │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Configuration:                                    │       │
│  │ • Max Budget: ₹400/unit                          │       │
│  │ • Target Price: ₹330/unit                        │       │
│  │ • Quantity: 2,000 units                          │       │
│  │ • Max Rounds: 3                                  │       │
│  └──────────────────────────────────────────────────┘       │
│                                                               │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Decision Engine:                                  │       │
│  │ 1. LLM (Groq API) - Strategic reasoning          │       │
│  │ 2. Constraint Validator - Budget checks          │       │
│  │ 3. Rule-Based Fallback - Backup logic            │       │
│  └──────────────────────────────────────────────────┘       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ A2A Protocol (HTTP/SSE)
                         │ Messages: OFFER, COUNTER, ACCEPT
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              SELLER AGENT (Port 8080)                        │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Configuration:                                    │       │
│  │ • Margin Price: ₹350/unit (PROTECTED)            │       │
│  │ • Target Price: ₹385/unit                        │       │
│  │ • Target Profit: 10%                             │       │
│  │ • Max Rounds: 3                                  │       │
│  └──────────────────────────────────────────────────┘       │
│                                                               │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Decision Engine:                                  │       │
│  │ 1. LLM (Groq API) - Strategic reasoning          │       │
│  │ 2. Constraint Validator - Margin protection      │       │
│  │ 3. Rule-Based Fallback - Backup logic            │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤝 How Negotiation Works

### Step-by-Step Flow

#### **Round 1: Opening Positions**

```
BUYER: "I'll pay ₹285/unit"
  ↓ (sends OFFER message)
SELLER: Receives offer
  ↓ (analyzes with LLM)
SELLER: "Too low! I want ₹430/unit"
  ↓ (sends COUNTER_OFFER)
BUYER: Receives counter
```

#### **Round 2: Convergence**

```
BUYER: "Let me increase to ₹330/unit"
  ↓ (sends COUNTER_OFFER)
SELLER: Receives counter
  ↓ (analyzes with LLM)
SELLER: "Getting closer... ₹370/unit"
  ↓ (sends COUNTER_OFFER)
BUYER: Receives counter
```

#### **Round 3: Final Decision**

```
BUYER: "Final offer: ₹350/unit"
  ↓ (sends COUNTER_OFFER or ACCEPT)
SELLER: Receives offer
  ↓ (analyzes with LLM)
SELLER: "Deal! I accept ₹350/unit"
  ↓ (sends ACCEPT_OFFER)
BUYER: Receives acceptance
  ↓ (auto-accepts - bilateral rule)
BUYER: Sends Purchase Order
SELLER: Sends Invoice
  ↓
✅ NEGOTIATION COMPLETE
```

### Message Types

1. **OFFER**: Initial proposal from buyer
2. **COUNTER_OFFER**: Alternative price proposal
3. **ACCEPT_OFFER**: Agreement to terms
4. **REJECT_OFFER**: Termination of negotiation
5. **PURCHASE_ORDER**: Buyer's formal order (post-acceptance)
6. **INVOICE**: Seller's billing document (post-acceptance)

---

## 🧠 AI Decision Making

### Hybrid Intelligence Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  DECISION REQUEST                        │
│  "Seller offered ₹370, should I accept or counter?"     │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              STEP 1: LLM REASONING (Groq)                │
│  ┌───────────────────────────────────────────────┐      │
│  │ Model: Llama 3.3 70B                          │      │
│  │                                                │      │
│  │ Prompt includes:                               │      │
│  │ • Current round (e.g., 2 of 3)                │      │
│  │ • Negotiation history                         │      │
│  │ • Constraints (budget/margin)                 │      │
│  │ • Strategic considerations                    │      │
│  │                                                │      │
│  │ LLM Response (JSON):                          │      │
│  │ {                                              │      │
│  │   "action": "COUNTER",                        │      │
│  │   "price": 350,                               │      │
│  │   "reasoning": "Moving to ₹350 shows         │      │
│  │                 flexibility while staying     │      │
│  │                 within budget...",            │      │
│  │   "confidence": 0.85                          │      │
│  │ }                                              │      │
│  └───────────────────────────────────────────────┘      │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│         STEP 2: CONSTRAINT VALIDATION                    │
│  ┌───────────────────────────────────────────────┐      │
│  │ Hard Constraints (MUST PASS):                 │      │
│  │                                                │      │
│  │ For BUYER:                                    │      │
│  │ ✓ Price ≤ Max Budget (₹400)                  │      │
│  │ ✓ Price > Last Buyer Offer                   │      │
│  │                                                │      │
│  │ For SELLER:                                   │      │
│  │ ✓ Price ≥ Margin Price (₹350)                │      │
│  │ ✓ Price < Last Seller Offer                  │      │
│  │                                                │      │
│  │ If LLM violates constraints:                  │      │
│  │ → Adjust price to meet constraints            │      │
│  │ → OR reject if impossible                     │      │
│  └───────────────────────────────────────────────┘      │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│         STEP 3: FALLBACK (if LLM fails)                  │
│  ┌───────────────────────────────────────────────┐      │
│  │ Rule-Based Logic:                             │      │
│  │                                                │      │
│  │ • Calculate gap between offers                │      │
│  │ • Apply concession rate (40-60%)              │      │
│  │ • Check round-based thresholds                │      │
│  │ • Make strategic counter-offer                │      │
│  │                                                │      │
│  │ Example:                                       │      │
│  │ Gap = ₹370 - ₹330 = ₹40                       │      │
│  │ Concession = 40% of ₹40 = ₹16                 │      │
│  │ New Offer = ₹330 + ₹16 = ₹346                 │      │
│  └───────────────────────────────────────────────┘      │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              FINAL DECISION EXECUTED                     │
│  Action: COUNTER_OFFER at ₹350/unit                     │
└─────────────────────────────────────────────────────────┘
```

### Why Hybrid Approach?

| Component | Purpose | Benefit |
|-----------|---------|---------|
| **LLM** | Strategic reasoning, context understanding | Human-like negotiation tactics |
| **Constraints** | Business rule enforcement | Prevents financial losses |
| **Fallback** | Reliability when LLM unavailable | System always works |

---

## 📡 Communication Protocol

### A2A (Agent-to-Agent) Protocol

Based on industry standard for autonomous agent communication.

#### Message Structure

```json
{
  "messageId": "uuid-1234",
  "kind": "message",
  "role": "agent",
  "contextId": "negotiation-context-id",
  "parts": [
    {
      "kind": "data",
      "data": {
        "type": "COUNTER_OFFER",
        "negotiationId": "NEG-1234567890",
        "round": 2,
        "pricePerUnit": 350,
        "previousPrice": 330,
        "from": "BUYER",
        "reasoning": "Strategic counter showing flexibility"
      }
    },
    {
      "kind": "text",
      "text": "Negotiation COUNTER_OFFER - Round 2"
    }
  ]
}
```

#### Transport Layer

- **Protocol**: HTTP with Server-Sent Events (SSE)
- **Format**: JSON
- **Streaming**: Real-time event updates
- **Ports**: 
  - Buyer: 9090
  - Seller: 8080

#### Agent Discovery

Each agent exposes an **Agent Card** at `/.well-known/agent-card.json`:

```json
{
  "name": "Tommy Buyer Agent",
  "version": "1.0.0",
  "description": "Autonomous buyer agent for trade negotiations",
  "capabilities": {
    "streaming": true,
    "negotiation": true
  }
}
```

---

## 💻 Code Flow

### Buyer Agent Flow

```typescript
// 1. USER INITIATES
User types: "start negotiation"
  ↓
// 2. BUYER GENERATES INITIAL OFFER
generateInitialOffer() 
  → Random between ₹250-₹320
  → Creates negotiation state
  ↓
// 3. SEND TO SELLER
sendToSeller(OFFER_DATA)
  → HTTP POST to seller:8080
  → Wait for response (with timeout)
  ↓
// 4. RECEIVE SELLER COUNTER
handleSellerMessage(COUNTER_OFFER_DATA)
  ↓
// 5. MAKE DECISION
makeNegotiationDecision()
  ├─→ getLLMDecision() // Ask Groq API
  ├─→ applyBuyerConstraints() // Validate
  └─→ ruleBasedDecision() // Fallback if needed
  ↓
// 6. EXECUTE DECISION
if (decision.action === "ACCEPT") {
  sendAcceptance()
  sendPurchaseOrder()
  printSummary()
} else if (decision.action === "COUNTER") {
  sendCounterOffer()
  // Wait for seller response
}
```

### Seller Agent Flow

```typescript
// 1. RECEIVE BUYER OFFER
handleBuyerOffer(OFFER_DATA)
  → Log the offer
  → Create negotiation state
  ↓
// 2. MAKE DECISION
makeNegotiationDecision()
  ├─→ getLLMDecision() // Ask Groq API
  ├─→ applySellerConstraints() // Validate
  └─→ ruleBasedDecision() // Fallback if needed
  ↓
// 3. EXECUTE DECISION
if (decision.action === "ACCEPT") {
  sendAcceptance()
  // Wait for Purchase Order
} else if (decision.action === "COUNTER") {
  sendCounterOffer()
  // Wait for buyer response
}
  ↓
// 4. RECEIVE PURCHASE ORDER
handlePurchaseOrder()
  → Generate invoice
  → Send to buyer
  → Complete negotiation
```

### LLM Integration (Groq)

```typescript
// llm-client.ts
class LLMNegotiationClient {
  async getNegotiationDecision(context) {
    // 1. Build detailed prompt
    const prompt = buildPrompt({
      role: "BUYER",
      round: 2,
      maxRounds: 3,
      lastOwnOffer: 330,
      lastTheirOffer: 370,
      history: [...],
      constraints: { maxBudget: 400 }
    });
    
    // 2. Call Groq API
    const response = await groqClient.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      messages: [
        { role: "system", content: "You are a negotiation expert..." },
        { role: "user", content: prompt }
      ],
      temperature: 0.7,
      response_format: { type: "json_object" }
    });
    
    // 3. Parse JSON response
    return {
      action: "COUNTER",
      price: 350,
      reasoning: "Strategic move...",
      confidence: 0.85
    };
  }
}
```

---

## ✨ Key Features

### 1. **Autonomous Operation**
- No human intervention required
- Agents make independent decisions
- Fully automated negotiation process

### 2. **Strategic Intelligence**
- LLM analyzes negotiation context
- Considers round progression
- Adapts strategy based on opponent behavior
- Balances risk vs. reward

### 3. **Safety Constraints**
- **Buyer**: Never exceeds budget
- **Seller**: Never goes below margin
- Prevents financial losses
- Validates every decision

### 4. **Bilateral Acceptance**
- When one agent accepts, the other must accept
- Prevents acceptance loops
- Ensures mutual agreement

### 5. **Graceful Degradation**
- LLM fails → Rule-based fallback
- Network timeout → Continue anyway
- Always completes negotiation

### 6. **Transparency**
- Detailed logging of every decision
- Shows reasoning for each move
- Tracks price movements
- Displays negotiation summary

### 7. **Configurable**
```typescript
// Easy to adjust parameters
BUYER_CONFIG = {
  maxBudget: 400,
  targetPrice: 330,
  maxRounds: 3,
  aggressiveness: 0.6
}

SELLER_CONFIG = {
  marginPrice: 350,
  targetProfitPercentage: 0.10,
  maxRounds: 3,
  flexibility: 0.5
}
```

---

## 📊 Example Negotiation Output

```
═══════════════════════════════════════════════════════════
  NEGOTIATION SESSION STARTED
═══════════════════════════════════════════════════════════
Negotiation ID: NEG-1770474476378
Context ID: a21feac2-0e4d-4a88-8fb1-52bd4e46a14f
Start Time: 7/2/2026, 7:57:56 pm
═══════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════
  ROUND 1 of 3
═══════════════════════════════════════════════════════════

🛒 BUYER → SELLER
   Action: INITIAL OFFER
   Price: ₹285/unit
   Strategy: Starting conservative, creating negotiation room

🏪 SELLER → BUYER
   Action: COUNTER OFFER
   Price: ₹430/unit
   Previous: ₹285
   Change: ↑ +₹145 (+50.9%)
   Strategy: Anchoring high to protect margin and profit

═══════════════════════════════════════════════════════════
  ROUND 2 of 3
═══════════════════════════════════════════════════════════

🛒 BUYER → SELLER
   Action: COUNTER OFFER
   Price: ₹330/unit
   Previous: ₹285
   Change: ↑ +₹45 (+15.8%)
   Gap Closed: 31.0%
   Strategy: Significant increase showing serious intent

🏪 SELLER → BUYER
   Action: COUNTER OFFER
   Price: ₹370/unit
   Previous: ₹430
   Change: ↓ -₹60 (-14.0%)
   Strategy: Moving toward buyer while maintaining profit

═══════════════════════════════════════════════════════════
  ROUND 3 of 3 (FINAL ROUND)
═══════════════════════════════════════════════════════════

🛒 BUYER → SELLER
   Action: COUNTER OFFER
   Price: ₹350/unit
   Previous: ₹330
   Change: ↑ +₹20 (+6.1%)
   Strategy: Final push - meeting at margin threshold

🏪 SELLER → BUYER
   Action: ✓ ACCEPT OFFER
   Price: ₹350/unit
   Strategy: Exactly at margin - minimal profit but deal secured

🛒 BUYER → SELLER
   Action: ✓ ACCEPT OFFER (AUTO)
   Price: ₹350/unit
   Strategy: Bilateral acceptance rule

═══════════════════════════════════════════════════════════
  NEGOTIATION SUMMARY
═══════════════════════════════════════════════════════════
Status: ✓ COMPLETED
Rounds Used: 3 / 3
Final Price: ₹350/unit

Starting Positions:
  Buyer: ₹285/unit
  Seller: ₹430/unit
  Gap: ₹145

Final Agreement:
  Price: ₹350/unit
  Buyer Movement: +₹65 (+22.8%)
  Seller Movement: -₹80 (-18.6%)

Financial Details:
  Quantity: 2,000 units
  Total Value: ₹700,000
  Buyer Savings: ₹100,000 (vs budget)
  Seller Profit: ₹0/unit (at margin)
═══════════════════════════════════════════════════════════
```

---

## 🎓 Key Concepts to Explain

### 1. **Agent Autonomy**
"Each agent operates independently with its own goals and constraints. They don't share information - just like real business negotiations."

### 2. **Hybrid Intelligence**
"We combine AI reasoning (LLM) with business rules (constraints). The AI provides strategy, the rules ensure safety."

### 3. **Protocol-Based Communication**
"Agents use a standard protocol (A2A) to communicate. This means any agent following the protocol can negotiate with any other agent."

### 4. **Graceful Degradation**
"If the AI fails, the system falls back to rule-based logic. The negotiation always completes successfully."

### 5. **Real-World Applicability**
"This demonstrates how AI agents can handle complex business processes autonomously - procurement, sales, contract negotiation, etc."

---

## 🚀 Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Runtime** | Node.js + TypeScript | Server execution |
| **AI Model** | Groq (Llama 3.3 70B) | Strategic reasoning |
| **Protocol** | A2A (Agent-to-Agent) | Communication standard |
| **Transport** | HTTP + SSE | Message delivery |
| **Framework** | Express.js | Web server |
| **SDK** | @a2a-js/sdk | Agent implementation |

---

## 📈 Benefits & Use Cases

### Benefits
1. **Automation**: Reduces manual negotiation time
2. **Consistency**: Applies same logic every time
3. **Scalability**: Can handle multiple negotiations simultaneously
4. **Transparency**: Every decision is logged and explainable
5. **Safety**: Hard constraints prevent bad deals

### Use Cases
1. **B2B Procurement**: Automated supplier negotiations
2. **Dynamic Pricing**: Real-time price negotiations
3. **Contract Negotiation**: Terms and conditions
4. **Resource Allocation**: Distributed systems
5. **Supply Chain**: Multi-party negotiations

---

## 🎯 Demonstration Points

When presenting to your coordinator, emphasize:

1. **"This is autonomous AI-to-AI negotiation"** - No human in the loop
2. **"Hybrid intelligence ensures safety"** - AI + Rules = Reliable
3. **"Protocol-based means interoperable"** - Works with any A2A agent
4. **"Real-world applicable"** - Can be deployed in production
5. **"Transparent and explainable"** - Every decision has reasoning

---

## 📝 Quick Demo Script

```
1. "Let me show you two AI agents negotiating autonomously..."
   → Start both agents

2. "The buyer wants to buy 2,000 units with a ₹400 budget..."
   → Show buyer configuration

3. "The seller has a ₹350 margin and wants 10% profit..."
   → Show seller configuration

4. "Now I'll start the negotiation..."
   → Run: start negotiation

5. "Watch how they strategically move toward agreement..."
   → Point out LLM reasoning in logs

6. "Notice the constraints are always respected..."
   → Show budget/margin checks

7. "And they reach agreement in 3 rounds!"
   → Show final summary
```

---

## 🔧 Technical Highlights

### Code Quality
- **TypeScript**: Type-safe implementation
- **Modular Design**: Separate concerns (LLM, constraints, communication)
- **Error Handling**: Graceful degradation at every level
- **Logging**: Comprehensive audit trail

### Performance
- **Streaming**: Real-time updates via SSE
- **Timeouts**: Prevents hanging connections
- **Async/Await**: Non-blocking operations
- **Efficient**: Completes in seconds

### Maintainability
- **Configurable**: Easy to adjust parameters
- **Extensible**: Can add new strategies
- **Testable**: Clear separation of logic
- **Documented**: Inline comments and guides

---

## 📚 Further Reading

- A2A Protocol Specification
- Groq API Documentation
- Agent-Based Negotiation Theory
- Multi-Agent Systems

---

**Created for presentation to project coordinator**
**System demonstrates autonomous AI agent negotiation with hybrid intelligence**
