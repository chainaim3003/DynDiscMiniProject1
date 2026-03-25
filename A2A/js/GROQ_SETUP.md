# Groq API Setup Guide

## Why Groq?

✅ **FREE** - Generous free tier with high limits
✅ **FAST** - Extremely fast inference (up to 10x faster than OpenAI)
✅ **RELIABLE** - No quota issues like Google's free tier
✅ **POWERFUL** - Uses Llama 3.3 70B model

## Get Your Groq API Key

### Step 1: Sign Up
1. Go to: https://console.groq.com/
2. Click "Sign Up" (free account)
3. Sign up with Google, GitHub, or email

### Step 2: Create API Key
1. After login, go to: https://console.groq.com/keys
2. Click "Create API Key"
3. Give it a name (e.g., "A2A Negotiation")
4. Copy the key (starts with `gsk_...`)

### Step 3: Add to .env
1. Open `Legent/A2A/js/.env`
2. Replace `your_groq_api_key_here` with your actual key:
   ```
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

## Free Tier Limits

Groq's free tier is VERY generous:
- **30 requests per minute**
- **14,400 requests per day**
- **7,000 tokens per minute**

This is more than enough for testing negotiations!

## Supported Models

The system uses `llama-3.3-70b-versatile` by default, but you can change it in `src/shared/llm-client.ts`:

Available models:
- `llama-3.3-70b-versatile` (Default - Best balance)
- `llama-3.1-70b-versatile` (Alternative)
- `mixtral-8x7b-32768` (Faster, smaller)
- `gemma2-9b-it` (Fastest, smallest)

## Test Your Setup

After adding your key, restart the agents:

**Terminal 1:**
```bash
npm run agents:seller
```

**Terminal 2:**
```bash
npm run agents:buyer
```

You should see:
```
✅ Initializing Groq with API key: gsk_...
```

If you see errors, check:
1. Key is correct (starts with `gsk_`)
2. No extra spaces in .env file
3. .env file is in `Legent/A2A/js/` directory

## Troubleshooting

### Error: "GROQ_API_KEY is required"
- Make sure .env file exists in `Legent/A2A/js/`
- Check the key name is exactly `GROQ_API_KEY`
- No quotes around the key value

### Error: "Invalid API key"
- Regenerate key at https://console.groq.com/keys
- Make sure you copied the entire key

### Error: "Rate limit exceeded"
- Wait 1 minute (free tier: 30 req/min)
- Unlikely with negotiation (only ~6 requests per negotiation)

## Success Indicators

When working correctly, you'll see in the terminal:
```
✅ Initializing Groq with API key: gsk_...
🛒 BUYER → SELLER
   Strategy: [LLM reasoning here - NOT "fallback strategy"]
```

The LLM reasoning will be strategic and contextual, not generic fallback messages.

## Cost

**FREE** for testing and development!

If you exceed free tier limits (unlikely), Groq's paid tier is very affordable:
- $0.05 per 1M input tokens
- $0.08 per 1M output tokens

(About 100x cheaper than GPT-4)
