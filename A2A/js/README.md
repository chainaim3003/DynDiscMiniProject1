# AI Agent-to-Agent (A2A) Negotiation System

An autonomous AI negotiation system where a **Buyer Agent** and **Seller Agent** negotiate trade deals independently using LLM-powered decision making via [Groq](https://groq.com/).

```
🛒 BUYER AGENT  ←──── A2A Protocol ────→  🏪 SELLER AGENT
  Port 9090                                   Port 8080
  Max Budget: ₹400                            Margin: ₹350
  Target: ₹330                                Target: ₹385
       │                                           │
       └──────── Groq LLM (Llama 3.3 70B) ────────┘
                   Strategic Reasoning
```

---

## Prerequisites

- [Node.js](https://nodejs.org/) v18 or higher
- [npm](https://www.npmjs.com/) or [pnpm](https://pnpm.io/)
- A free [Groq API key](https://console.groq.com/)

---

## Step 1 — Get Your Groq API Key

Groq is **free** and much faster than OpenAI or Google Gemini.

1. Go to **[https://console.groq.com/](https://console.groq.com/)**
2. Sign up with Google, GitHub, or email (free account)
3. Navigate to **[https://console.groq.com/keys](https://console.groq.com/keys)**
4. Click **"Create API Key"**
5. Give it a name (e.g. `a2a-negotiation`)
6. Copy the key — it starts with `gsk_...`

> Keep this key safe. You'll need it in the next step.

---

## Step 2 — Clone & Install

```bash
git clone <your-repo-url>
cd Legent/A2A/js
npm install
```

---

## Step 3 — Configure Environment Variables

You need to set your Groq API key in **three** `.env` files.

### File 1: `Legent/A2A/js/.env`

```bash
GROQ_API_KEY=gsk_your_actual_key_here
```

### File 2: `Legent/A2A/js/src/agents/buyer-agent/.env`

Create this file and copy the code from .env.local
 and add the below with existing .env file:

```env
GROQ_API_KEY=gsk_your_actual_key_here
PORT=9090
```

### File 3: `Legent/A2A/js/src/agents/seller-agent/.env`

Create this file and copy the code from .env.local
 and add the below with existing .env file:

```env
GROQ_API_KEY=gsk_your_actual_key_here
PORT=8080
```

> Replace `gsk_your_actual_key_here` with your real Groq API key in all three files.

---

## Step 4 — Run the System

You need **3 separate terminal windows**, all opened in `Legent/A2A/js`.

### Terminal 1 — Start Seller Agent

```bash
npm run agents:seller
```

Wait until you see:
```
✅ Initializing Groq with API key: gsk_...
🟢 Seller Agent running on http://localhost:8080
💰 Margin Price: ₹350/unit (PROTECTED)
```

### Terminal 2 — Start Buyer Agent

```bash
npm run agents:buyer
```

Wait until you see:
```
✅ Initializing Groq with API key: gsk_...
🟢 Buyer Agent running on http://localhost:9090
💰 Max Budget: ₹400/unit
```

### Terminal 3 — Start CLI (connect to Buyer Agent)

```bash
npx tsx src/cli.ts http://localhost:9090
```

You should see:
```
A2A Terminal Client
✓ Agent Card Found:
  Name: Tommy Buyer Agent
  Streaming: Supported

Tommy Buyer Agent > You:
```

---

## Step 5 — Start a Negotiation

In Terminal 3, type one of the following:

### Random starting price (₹250–₹320)
```
start negotiation
```

### Specify your own starting price
```
start negotiation 300
```

The agents will negotiate automatically across 3 rounds and close a deal.

---

## How It Works

### Negotiation Flow

```
User: "start negotiation 300"
         │
         ▼
BUYER sends OFFER ₹300 ──────────────────────→ SELLER receives
                                                SELLER asks Groq LLM
                                                "Should I accept or counter?"
                                                LLM: "Counter at ₹420"
BUYER receives ←──────────── SELLER sends COUNTER ₹420
BUYER asks Groq LLM
"Should I accept or counter?"
LLM: "Counter at ₹340"
BUYER sends COUNTER ₹340 ────────────────────→ SELLER receives
                                                LLM: "Counter at ₹370"
BUYER receives ←──────────── SELLER sends COUNTER ₹370
LLM: "Final round - accept ₹370"
BUYER sends ACCEPT ₹370 ─────────────────────→ SELLER receives
                                                SELLER sends ACCEPT back
                                                SELLER status = COMPLETED
BUYER receives ACCEPT
BUYER status = COMPLETED
BUYER sends PURCHASE ORDER ──────────────────→ SELLER sends INVOICE
                                                ✅ DEAL CLOSED
```

### Decision Engine (per agent)

Each agent uses a 3-layer decision system:

```
1. LLM (Groq / Llama 3.3 70B)
   → Analyzes context, history, round number
   → Returns: { action, price, reasoning, confidence }
        │
        ▼
2. Constraint Validator
   → Buyer: price must be ≤ ₹400 (budget)
   → Seller: price must be ≥ ₹350 (margin)
   → Adjusts or overrides LLM if violated
        │
        ▼
3. Rule-Based Fallback (if LLM fails)
   → Calculates concession based on gap
   → Applies round-based thresholds
   → Always produces a valid decision
```

### Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `OFFER` | Buyer → Seller | Initial price proposal |
| `COUNTER_OFFER` | Both ways | Alternative price |
| `ACCEPT_OFFER` | Both ways | Agreement to terms |
| `REJECT_OFFER` | Both ways | Termination |
| `PURCHASE_ORDER` | Buyer → Seller | Formal order after deal |
| `INVOICE` | Seller → Buyer | Billing document |

---

## Configuration

### Buyer Agent (`src/agents/buyer-agent/index.ts`)

```typescript
const BUYER_CONFIG = {
  maxBudget: 400,        // Hard limit — never exceeded
  targetQuantity: 2000,  // Units to purchase
  maxRounds: 3,          // Max negotiation rounds
  initialOfferRange: { min: 250, max: 320 }, // Random range
  targetPrice: 330,      // Ideal outcome
};
```

### Seller Agent (`src/agents/seller-agent/index.ts`)

```typescript
const SELLER_CONFIG = {
  marginPrice: 350,              // Hard floor — never goes below
  targetProfitPercentage: 0.10,  // 10% profit target
  maxRounds: 3,
};
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `start negotiation` | Start with random offer (₹250–₹320) |
| `start negotiation 300` | Start with specific offer of ₹300 |
| `/new` | Clear session, start fresh |
| `/exit` | Quit the CLI |

---

## Project Structure

```
Legent/A2A/js/
├── src/
│   ├── agents/
│   │   ├── buyer-agent/
│   │   │   ├── index.ts          # Buyer agent logic
│   │   │   └── .env              # Buyer env config
│   │   └── seller-agent/
│   │       ├── index.ts          # Seller agent logic
│   │       └── .env              # Seller env config
│   ├── shared/
│   │   ├── llm-client.ts         # Groq API integration
│   │   ├── negotiation-types.ts  # Shared TypeScript types
│   │   └── logger.ts             # Negotiation logger
│   └── cli.ts                    # CLI interface
├── agent-cards/
│   ├── tommyBuyerAgent-card.json
│   └── jupiterSellerAgent-card.json
├── .env                          # Root env (Groq key)
├── .env.example                  # Template
├── package.json
└── tsconfig.json
```

---

## Troubleshooting

### "GROQ_API_KEY is required"
Your `.env` file is missing or the key name is wrong. Make sure all three `.env` files have:
```
GROQ_API_KEY=gsk_your_key_here
```

### "Address already in use"
A previous agent is still running on that port. Kill it:
```bash
# Windows
netstat -ano | findstr :8080
taskkill /PID <pid> /F

netstat -ano | findstr :9090
taskkill /PID <pid> /F
```

### "Cannot find module"
Make sure you're in the right directory and have installed dependencies:
```bash
cd Legent/A2A/js
npm install
```

### Stream timeout errors (`UND_ERR_BODY_TIMEOUT`)
This is handled gracefully — the negotiation continues. If it happens repeatedly, restart all three terminals.

### LLM using "fallback strategy"
This means Groq API failed. Check:
- Your API key is valid
- You haven't exceeded rate limits (30 req/min on free tier)
- Your internet connection is working

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | TypeScript / Node.js |
| LLM | Groq API (Llama 3.3 70B) |
| Agent Protocol | A2A (`@a2a-js/sdk`) |
| Transport | HTTP + Server-Sent Events |
| Web Framework | Express.js |
| LLM Client | OpenAI SDK (Groq-compatible) |

---

## Groq Free Tier Limits

| Limit | Value |
|-------|-------|
| Requests per minute | 30 |
| Requests per day | 14,400 |
| Tokens per minute | 7,000 |

A full 3-round negotiation uses ~6 API calls — well within free tier limits.

---

## License

This project is for demonstration purposes. See disclaimer below.

> When building production applications, treat any agent operating outside your direct control as a potentially untrusted entity. Validate and sanitize all data received from external agents before use.
