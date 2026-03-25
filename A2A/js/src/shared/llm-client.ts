// ================= LLM CLIENT FOR AGENTIC DECISION MAKING =================
import OpenAI from "openai";
import { LLMResponse, AgentRole } from "./negotiation-types.js";

export interface LLMPromptContext {
    role: AgentRole;
    round: number;
    maxRounds: number;
    lastOwnOffer?: number;
    lastTheirOffer?: number;
    history: any[];
    constraints: {
        marginPrice?: number; // Seller only
        maxBudget?: number; // Buyer only
        quantity: number;
    };
    targetPrice?: number;
}

export class LLMNegotiationClient {
    private client: OpenAI;
    private model: string = "llama-3.3-70b-versatile"; // Fast and powerful Groq model

    constructor(apiKey?: string) {
        const key = apiKey || process.env.GROQ_API_KEY;

        if (!key) {
            console.error("❌ No API key found! GROQ_API_KEY env var:", process.env.GROQ_API_KEY);
            throw new Error("GROQ_API_KEY is required");
        }

        console.log("✅ Initializing Groq with API key:", key?.substring(0, 10) + "...");

        // Groq uses OpenAI-compatible API
        this.client = new OpenAI({
            apiKey: key,
            baseURL: "https://api.groq.com/openai/v1"
        });
    }

    async getNegotiationDecision(
        context: LLMPromptContext
    ): Promise<LLMResponse> {
        const prompt = this.buildPrompt(context);

        try {
            const systemPrompt = `You are an expert negotiation AI ${context.role === "BUYER" ? "buyer" : "seller"} agent. 
You must make strategic decisions to maximize your goals while being realistic about deal closure.
Always respond with valid JSON only, no additional text.`;

            const response = await this.client.chat.completions.create({
                model: this.model,
                messages: [
                    { role: "system", content: systemPrompt },
                    { role: "user", content: prompt }
                ],
                temperature: 0.7,
                response_format: { type: "json_object" }
            });

            const content = response.choices[0]?.message?.content || "{}";
            const parsed = JSON.parse(content);

            return {
                action: parsed.action || "COUNTER",
                price: parsed.price ? Math.round(parsed.price) : undefined,
                reasoning: parsed.reasoning || "Strategic decision",
                confidence: parsed.confidence || 0.7,
            };
        } catch (error) {
            console.error("❌ LLM API Error:", error);
            // Return fallback decision
            return {
                action: "COUNTER",
                price: context.lastTheirOffer,
                reasoning: "LLM unavailable - using fallback strategy",
                confidence: 0.3,
            };
        }
    }

    private buildPrompt(context: LLMPromptContext): string {
        const isBuyer = context.role === "BUYER";

        return `You are negotiating a trade deal as the ${context.role}.

CURRENT SITUATION:
- Round: ${context.round} of ${context.maxRounds} ${context.round === context.maxRounds ? "(FINAL ROUND)" : ""}
- Quantity: ${context.constraints.quantity} units
${context.lastOwnOffer ? `- Your last offer: ₹${context.lastOwnOffer}/unit` : ""}
${context.lastTheirOffer ? `- Their last offer: ₹${context.lastTheirOffer}/unit` : ""}

${context.history.length > 0
                ? `
NEGOTIATION HISTORY:
${context.history
                    .map(
                        (h) =>
                            `  Round ${h.round}: Buyer ₹${h.buyerOffer || "?"} → Seller ₹${h.sellerOffer || "?"}`
                    )
                    .join("\n")}
`
                : ""
            }

YOUR CONSTRAINTS:
${isBuyer
                ? `- Maximum Budget: ₹${context.constraints.maxBudget}/unit (NEVER exceed this)
- Target Price: ₹${context.targetPrice}/unit (ideal outcome)
- Goal: Minimize total cost while securing the deal`
                : `- Margin Price: ₹${context.constraints.marginPrice}/unit (NEVER go below this - you lose money)
- Target Price: ₹${context.targetPrice}/unit (ideal outcome with good profit)
- Goal: Maximize profit while closing the deal`
            }

STRATEGIC CONSIDERATIONS:
${context.round === 1
                ? `- First impressions matter - set the tone
${isBuyer ? "- Start lower to create negotiation room" : "- Start higher to anchor expectations"}
- Don't be too extreme or you'll lose credibility`
                : context.round === context.maxRounds
                    ? `- THIS IS THE FINAL ROUND - deal will fail if not accepted
- Consider: Is this the best offer you'll get?
- Accepting a "good enough" deal is better than no deal
${isBuyer ? `- If their offer is within budget, seriously consider accepting` : `- If their offer is STRICTLY ABOVE ₹${context.constraints.marginPrice} (your minimum floor), you may accept
- NEVER accept at exactly ₹${context.constraints.marginPrice} — that is zero profit
- Any deal with at least ₹1 profit is worth taking in the final round`}`
                    : `- Middle rounds are for convergence
- Show flexibility but don't concede too quickly
- Analyze their pattern: Are they moving toward you?
- Calculate: Will we reach agreement by round ${context.maxRounds}?`
            }

DECISION ANALYSIS:
${context.lastTheirOffer
                ? `
1. Their offer (₹${context.lastTheirOffer}) vs your constraints:
${isBuyer
                    ? `   - ${context.lastTheirOffer <= context.constraints.maxBudget! ? "✓ Within budget" : "✗ Exceeds budget"}
   - ${context.lastTheirOffer <= context.targetPrice! ? "✓ Below target (EXCELLENT)" : `${((context.lastTheirOffer - context.targetPrice!) / context.targetPrice! * 100).toFixed(1)}% above target`}`
                    : `   - ${context.lastTheirOffer > context.constraints.marginPrice! ? "✓ Above minimum (ACCEPTABLE)" : "✗ At or below minimum floor (MUST REJECT or COUNTER)"}
   - Profit: ₹${context.lastTheirOffer - context.constraints.marginPrice!}/unit (${(((context.lastTheirOffer - context.constraints.marginPrice!) / context.constraints.marginPrice!) * 100).toFixed(1)}%)
   - NOTE: ₹${context.constraints.marginPrice}/unit is your MINIMUM — any offer at or below this has zero or negative profit`
                }

2. Gap analysis:
${context.lastOwnOffer
                    ? `   - Current gap: ₹${Math.abs(context.lastTheirOffer - context.lastOwnOffer)}
   - You need to ${isBuyer ? "increase" : "decrease"} by ${Math.abs(context.lastTheirOffer - context.lastOwnOffer)} to meet their price`
                    : ""
                }

3. Should you ACCEPT or make a COUNTER-OFFER?
   - If ACCEPT: Explain why this is the right price
   - If COUNTER: Calculate a strategic new price that:
     ${isBuyer ? `* Moves toward their offer (show willingness)
     * Stays within budget
     * Increases pressure on them to accept` : `* Moves toward their offer (show flexibility)
     * Stays above margin
     * Signals you're serious about closing`}
`
                : `
This is the ${isBuyer ? "initial offer" : "first counter-offer"}.
Calculate a strong opening position that gives you negotiation room.
`
            }

RESPOND WITH JSON ONLY (no other text):
{
  "action": "ACCEPT" or "COUNTER" or "REJECT",
  "price": <number if COUNTER, omit if ACCEPT/REJECT>,
  "reasoning": "<1-2 sentence explanation of your strategic thinking>",
  "confidence": <0.0 to 1.0 how confident you are in this decision>
}

Example responses:
{"action": "ACCEPT", "reasoning": "Their offer of ₹360 is within budget and further negotiation risks deal failure in final round", "confidence": 0.85}
{"action": "COUNTER", "price": 340, "reasoning": "Moving up from ₹320 to ₹340 shows flexibility while staying well below budget, testing their price sensitivity", "confidence": 0.75}
{"action": "REJECT", "reasoning": "Their offer of ₹330 is below our margin of ₹350, accepting would result in a loss", "confidence": 1.0}`;
    }
}
