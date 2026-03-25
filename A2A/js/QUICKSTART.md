# A2A Agentic Negotiation - Quick Start

## ✅ Prerequisites Check

You already have `node_modules` installed, so you're ready to go!

## 🚀 Running the System

You need **3 separate terminal windows**. Open them all in:
```
C:\CHAINAIM3003\mcp-servers\algoTITANV6\Legent\A2A\js
```

### Terminal 1: Seller Agent (Port 8080)
```bash
npm run agents:seller
```

Wait for:
```
🟢 Seller Agent running on http://localhost:8080
💰 Margin Price: ₹350/unit (PROTECTED)
```

### Terminal 2: Buyer Agent (Port 9090)
```bash
npm run agents:buyer
```

Wait for:
```
🟢 Buyer Agent running on http://localhost:9090
💰 Max Budget: ₹400/unit
```

### Terminal 3: CLI to Start Negotiation
```bash
npx tsx src/cli.ts http://localhost:9090
```

When CLI loads, type:
```
start negotiation
```

## 🎯 What You'll See

The negotiation will play out across Terminal 1 and 2 with beautiful formatted output:

```
═══════════════════════════════════════════════════════════
  ROUND 1 of 3
═══════════════════════════════════════════════════════════

  🛒 BUYER → SELLER
     Action: INITIAL OFFER
     Price: ₹285/unit
     Strategy: Starting conservative...

  🏪 SELLER → BUYER
     Action: COUNTER OFFER
     Price: ₹430/unit
     Change: ↑ +₹145 (+50.9%)
```

## ⚙️ Configuration

### Set Your OpenAI API Key

**IMPORTANT**: For hybrid LLM decision-making, you need an OpenAI API key.

1. Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

2. Edit `.env` and add your key:
```
OPENAI_API_KEY=sk-proj-your-key-here
```

### Without LLM (Rule-Based Only)

If you don't have an OpenAI key, the system will automatically fall back to rule-based negotiation with this warning:
```
⚠️ LLM decision invalid, using rule-based fallback
```

The negotiation will still work perfectly!

## 🎲 Test Different Scenarios

Run the negotiation multiple times - **the buyer's starting price is randomized** between ₹250-₹320, so every negotiation will be different!

## 🐛 Troubleshooting

### Error: "Cannot find module"
**Solution**: Run from the correct directory:
```bash
cd C:\CHAINAIM3003\mcp-servers\algoTITANV6\Legent\A2A\js
```

### Error: "Address already in use"
**Solution**: Close previous agent instances and restart.

### Error: "OpenAI API Error"
**Result**: System automatically uses rule-based fallback. Negotiation continues normally.

## 📊 Advanced Usage

### Monitor Seller's View (Terminal 4 - Optional)
```bash
npx tsx src/cli.ts http://localhost:8080
```

Type:
```
show negotiations
```

This shows the seller's perspective of active negotiations.

## 🎉 Success Indicators

- ✅ Both agents started on correct ports
- ✅ Negotiation progresses through rounds
- ✅ Deal closes with invoice generation
- ✅ Detailed logs show price movements and reasoning

---

**Ready to negotiate! 🤝**
