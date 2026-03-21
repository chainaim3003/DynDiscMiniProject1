# Agent Card Generation - Quick Reference

## 🚀 Quick Start

```bash
# 1. Run your vLEI workflow
./run-all-buyerseller-2-with-agents.sh

# 2. Generate agent cards
chmod +x generate-agent-cards.sh  # First time only
./generate-agent-cards.sh

# 3. View results
ls -lh agent-cards/
cat agent-cards/jupiterSellerAgent-card.json | jq .
```

## 📋 What You Get

### Input (From vLEI Workflow)
```
task-data/
├── jupiterSellerAgent-info.json       ← Agent AID & OOBI
├── Jupiter_Chief_Sales_Officer-info.json  ← Delegator AID
├── Jupiter_Knitting_Company-info.json     ← LE AID & LEI
├── tommyBuyerAgent-info.json          ← Agent AID & OOBI
├── Tommy_Chief_Procurement_Officer-info.json
├── Tommy_Hilfiger_Europe-info.json
├── oor-credential-info.json           ← Credential SAID
└── qvi-info.json                      ← QVI AID
```

### Output (Generated)
```
agent-cards/
├── jupiterSellerAgent-card.json  ← Complete agent card
└── tommyBuyerAgent-card.json     ← Complete agent card
```

## 🎯 Key Features

✅ **No Code Changes** - Completely separate from your vLEI workflow  
✅ **No A2A Server** - Pure file generation  
✅ **Automatic Mapping** - Reads all output files and maps data  
✅ **Complete vLEI Data** - All AIDs, SAIDs, OOBIs included  
✅ **Customizable** - Easy to modify template  

## 📝 Agent Card Contains

```json
{
  "name": "Jupiter Seller Agent",
  "description": "...",
  "provider": { "organization": "JUPITER KNITTING COMPANY" },
  "skills": [ /* procurement, invoicing, etc */ ],
  "extensions": {
    "gleifIdentity": {
      "lei": "3358004DXAMRWRUIYJ05",
      "officialRole": "ChiefSalesOfficer"
    },
    "vLEImetadata": {
      "delegatorAID": "EJ8sMfueQ...",  // OOR holder
      "delegateeAID": "EMhQNVE2R...",  // Agent
      "delegatorSAID": "EETDbCBDo...", // OOR credential
      "verificationPath": [
        "GLEIF_ROOT → QVI",
        "QVI → JUPITER → Chief Sales Officer → Agent"
      ],
      "status": "verified"
    },
    "keriIdentifiers": {
      "agentAID": "EMhQNVE2R...",
      "oorHolderAID": "EJ8sMfueQ...",
      "legalEntityAID": "ENo_qB02q...",
      "qviAID": "ECzkXCdHw..."
    }
  }
}
```

## 🔄 Workflow Integration

### Option 1: Automatic (Recommended)
```bash
# Add to end of run-all-buyerseller-2-with-agents.sh
./generate-agent-cards.sh
```

### Option 2: Manual
```bash
# Run after main workflow
./generate-agent-cards.sh
```

## 🛠️ Customization

### Add Custom Fields
Edit `generate-agent-cards.js`:

```javascript
extensions: {
  // ... existing fields ...
  myCustomData: {
    field1: "value1",
    field2: "value2"
  }
}
```

### Modify Skills
```javascript
skills: [
  {
    id: "custom_skill",
    name: "Custom Skill",
    description: "Does something specific",
    tags: ["custom", "skill"]
  }
]
```

## ❓ Common Issues

| Issue | Solution |
|-------|----------|
| `task-data directory not found` | Run vLEI workflow first |
| `Configuration file not found` | Check `appconfig/configBuyerSellerAIAgent1.json` exists |
| `Failed to read agent info` | Ensure workflow completed successfully |
| Missing AIDs | Check workflow logs for errors |

## 📚 Full Documentation

See: `AGENT-CARD-GENERATION-GUIDE.md`

## ✨ Example Output

```
═══════════════════════════════════════════════
  vLEI Agent Card Generator
═══════════════════════════════════════════════

✓ Configuration loaded

─────────────────────────────────────────────
Processing: JUPITER KNITTING COMPANY
─────────────────────────────────────────────

→ Generating agent card for: jupiterSellerAgent
✓ Agent info loaded: jupiterSellerAgent
✓ Person info loaded: Jupiter_Chief_Sales_Officer
✓ Agent card generated for jupiterSellerAgent
  Agent AID: EMhQNVE2RFIhaf9j4WfbxGA7xvDRoxcUoTSH0IfsxK6k
  Delegator AID: EJ8sMfueQNpg5gkJZAP476Ee4TgxgvfSFMg1R8JIZD6A
  LEI: 3358004DXAMRWRUIYJ05
✓ Agent card saved: ./agent-cards/jupiterSellerAgent-card.json

═══════════════════════════════════════════════
  ✅ Generation Complete
  Generated 2 agent card(s)
═══════════════════════════════════════════════
```

## 🎁 Benefits

1. **Standards Compliant** - Follows vLEI and KERI specs
2. **Verifiable** - All credentials traceable to GLEIF root
3. **Interoperable** - Works with any vLEI-compliant system
4. **Auditable** - Complete trust chain documented
5. **Automated** - No manual data entry needed

## 🚦 Ready to Use

Your agent cards are ready for:
- ✅ Agent-to-Agent (A2A) communication
- ✅ Credential verification
- ✅ Trust chain validation
- ✅ OOBI resolution
- ✅ Discovery protocols

## 📞 Need Help?

1. Read full guide: `AGENT-CARD-GENERATION-GUIDE.md`
2. Check workflow docs: `AGENT-DELEGATION-IMPLEMENTATION-COMPLETE.md`
3. Review vLEI specs: https://www.gleif.org/en/vlei
