// ================= ROLE-AWARE NEGOTIATION LOGGER =================

import fs   from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { NegotiationLog, AgentRole, NegotiationAction } from "./negotiation-types.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ANSI color codes
const C = {
    reset:    "\x1b[0m",
    bold:     "\x1b[1m",
    dim:      "\x1b[2m",
    red:      "\x1b[31m",
    green:    "\x1b[32m",
    yellow:   "\x1b[33m",
    blue:     "\x1b[34m",
    magenta:  "\x1b[35m",
    cyan:     "\x1b[36m",
    white:    "\x1b[37m",
    bgRed:    "\x1b[41m",
    bgGreen:  "\x1b[42m",
    bgBlue:   "\x1b[44m",
    bgYellow: "\x1b[43m",
};

// ─── Width constant for all boxes ────────────────────────────────────────────
const W = 58;

function hline(char = "─") { return char.repeat(W); }

// ── Suppress @a2a-js/sdk internal stdout noise ───────────────────────────────
const SDK_NOISE_PATTERNS = [
    "ResultManager:",
    "Error reading or parsing SSE stream:",
];

export function suppressSDKNoise(): void {
    const originalWrite = process.stdout.write.bind(process.stdout);

    (process.stdout.write as any) = function (
        chunk: any,
        encodingOrCallback?: any,
        callback?: any
    ): boolean {
        const text = typeof chunk === "string" ? chunk : chunk?.toString?.() ?? "";
        const isNoise = SDK_NOISE_PATTERNS.some((pattern) => text.includes(pattern));
        if (isNoise) {
            if (typeof encodingOrCallback === "function") encodingOrCallback();
            else if (typeof callback === "function") callback();
            return true;
        }
        return originalWrite(chunk, encodingOrCallback, callback);
    };
}

// ── Subtle internal-log helper — dimmed, no box ───────────────────────────────
export function logInternal(msg: string) {
    console.log(`${C.dim}  ⋯ ${msg}${C.reset}`);
}

export class NegotiationLogger {
    private logs: NegotiationLog[] = [];
    private negotiationId: string;
    private startTime: Date;
    private myRole: AgentRole;

    // Running price trail for the summary table
    private priceTrail: { round: number; buyer?: number; seller?: number }[] = [];

    constructor(negotiationId: string, myRole: AgentRole) {
        this.negotiationId = negotiationId;
        this.startTime     = new Date();
        this.myRole        = myRole;
    }

    // ── Public log entry point ────────────────────────────────────────────────
    log(entry: Omit<NegotiationLog, "timestamp" | "negotiationId">) {
        const logEntry: NegotiationLog = {
            ...entry,
            timestamp:     new Date().toISOString(),
            negotiationId: this.negotiationId,
        };
        this.logs.push(logEntry);
        this.printLog(logEntry);
    }

    // ── Core log printer ──────────────────────────────────────────────────────
    private printLog(log: NegotiationLog) {
        const isMine = log.from === this.myRole;

        const isEchoAccept =
            log.decision === "ACCEPT" &&
            log.reasoning?.toLowerCase().includes("bilateral");
        if (isEchoAccept) {
            logInternal(`Bilateral acceptance confirmed at ₹${log.offeredPrice}`);
            return;
        }

        let headerBg:   string;
        let headerText: string;
        let priceColor: string;

        switch (log.decision) {
            case "ACCEPT":
                headerBg   = C.bgGreen + C.bold + C.white;
                headerText = isMine ? "✓  ACCEPTED  ── sent to counterpart" : "✓  ACCEPTED  ── received";
                priceColor = C.green + C.bold;
                break;
            case "REJECT":
                headerBg   = C.bgRed + C.bold + C.white;
                headerText = isMine ? "✗  REJECTED  ── sent" : "✗  REJECTED  ── received";
                priceColor = C.red + C.bold;
                break;
            case "OFFER":
                headerBg   = C.bgBlue + C.bold + C.white;
                headerText = isMine ? "▶  INITIAL OFFER  ── sent to seller" : "▶  INITIAL OFFER  ── received from buyer";
                priceColor = C.cyan + C.bold;
                break;
            default:
                if (isMine) {
                    headerBg   = C.cyan  + C.bold;
                    headerText = "↑  COUNTER-OFFER  ── sent";
                    priceColor = C.cyan  + C.bold;
                } else {
                    headerBg   = C.yellow + C.bold;
                    headerText = "↓  COUNTER-OFFER  ── received";
                    priceColor = C.yellow + C.bold;
                }
                break;
        }

        console.log("");
        console.log(`  ${headerBg}  ${headerText.padEnd(W - 2)}  ${C.reset}`);

        if (log.offeredPrice !== undefined) {
            console.log(`  ${priceColor}    ₹${log.offeredPrice} / unit${C.reset}`);
        }

        if (log.previousPrice !== undefined && log.priceMovement !== undefined) {
            const arrow    = log.priceMovement >= 0 ? "▲" : "▼";
            const sign     = log.priceMovement >= 0 ? "+" : "";
            const movColor = log.priceMovement >= 0 ? C.green : C.red;
            const pct      = log.priceMovementPercent?.toFixed(1) ?? "0.0";
            console.log(
                `  ${C.dim}    was ₹${log.previousPrice}  ${C.reset}${movColor}${arrow} ${sign}₹${Math.abs(log.priceMovement)} (${sign}${pct}%)${C.reset}`
            );
        }

        if (log.gap !== undefined && log.gap > 0) {
            console.log(`  ${C.dim}    gap left : ₹${log.gap}${C.reset}`);
        }

        if (log.reasoning) {
            console.log(`  ${C.dim}    reason   : ${log.reasoning}${C.reset}`);
        }

        this.updatePriceTrail(log);
    }

    // ── Track prices per round ────────────────────────────────────────────────
    private updatePriceTrail(log: NegotiationLog) {
        if (log.offeredPrice === undefined) return;
        let entry = this.priceTrail.find(e => e.round === log.round);
        if (!entry) {
            entry = { round: log.round };
            this.priceTrail.push(entry);
        }
        if (log.from === "BUYER") entry.buyer  = log.offeredPrice;
        else                      entry.seller = log.offeredPrice;
    }

    // ── Round header ──────────────────────────────────────────────────────────
    printRoundHeader(round: number, maxRounds: number) {
        const isFinal = round === maxRounds;
        const label   = isFinal
            ? `  ROUND ${round} / ${maxRounds}  ◀  FINAL ROUND`
            : `  ROUND ${round} / ${maxRounds}`;
        const color   = isFinal ? C.magenta + C.bold : C.bold;
        console.log("");
        console.log(`${C.dim}  ${hline()}${C.reset}`);
        console.log(`${color}${label}${C.reset}`);
        console.log(`${C.dim}  ${hline()}${C.reset}`);
    }

    // ── Session header ────────────────────────────────────────────────────────
    printSessionHeader(_contextId: string) {
        const roleEmoji = this.myRole === "BUYER" ? "🛒" : "🏪";
        const roleName  = this.myRole === "BUYER" ? "BUYER AGENT" : "SELLER AGENT";

        console.log("");
        console.log(`${C.bold}  ╔${"═".repeat(W)}╗${C.reset}`);
        console.log(`${C.bold}  ║  ${roleEmoji}  ${roleName.padEnd(W - 5)}║${C.reset}`);
        console.log(`${C.bold}  ║  Negotiation : ${this.negotiationId.padEnd(W - 16)}║${C.reset}`);
        console.log(`${C.bold}  ║  Started     : ${this.startTime.toLocaleTimeString().padEnd(W - 16)}║${C.reset}`);
        console.log(`${C.bold}  ╚${"═".repeat(W)}╝${C.reset}`);
    }

    // ── Price trail table ─────────────────────────────────────────────────────
    private printPriceTrail() {
        if (this.priceTrail.length === 0) return;

        const col      = { round: 8, buyer: 14, seller: 14, gap: 12 };
        const rowWidth = col.round + col.buyer + col.seller + col.gap;

        console.log("");
        console.log(`${C.dim}  ┌─ Price Trail ${"─".repeat(rowWidth - 1)}┐${C.reset}`);
        console.log(
            `${C.dim}  │  ${"Rnd".padEnd(col.round)}${"Buyer".padEnd(col.buyer)}` +
            `${"Seller".padEnd(col.seller)}${"Gap".padEnd(col.gap)}│${C.reset}`
        );
        console.log(`${C.dim}  │  ${"─".repeat(rowWidth)}│${C.reset}`);

        for (const e of this.priceTrail) {
            const b   = e.buyer  !== undefined ? `₹${e.buyer}`  : "—";
            const s   = e.seller !== undefined ? `₹${e.seller}` : "—";
            const gap = (e.buyer !== undefined && e.seller !== undefined)
                ? `₹${Math.abs(e.seller - e.buyer)}`
                : "—";
            console.log(
                `${C.dim}  │  ${String(e.round).padEnd(col.round)}${b.padEnd(col.buyer)}` +
                `${s.padEnd(col.seller)}${gap.padEnd(col.gap)}│${C.reset}`
            );
        }
        console.log(`${C.dim}  └${"─".repeat(rowWidth + 2)}┘${C.reset}`);
    }

    // ── Negotiation summary ───────────────────────────────────────────────────
    printNegotiationSummary(
        status: "COMPLETED" | "FAILED",
        details: {
            roundsUsed:        number;
            maxRounds:         number;
            finalPrice?:       number;
            buyerStartPrice?:  number;
            sellerStartPrice?: number;
            totalCost?:        number;
            totalRevenue?:     number;
            profitMargin?:     number;
            quantity:          number;
        }
    ) {
        this.printPriceTrail();

        const isOk     = status === "COMPLETED";
        const bgColor  = isOk ? C.bgGreen + C.bold + C.white : C.bgRed + C.bold + C.white;
        const headline = isOk ? "✅  DEAL CLOSED" : "❌  NO DEAL REACHED";

        console.log("");
        console.log(`  ${bgColor}  ${headline.padEnd(W - 2)}  ${C.reset}`);
        console.log(`${C.dim}  Rounds: ${details.roundsUsed} / ${details.maxRounds}${C.reset}`);

        if (isOk && details.finalPrice !== undefined) {
            console.log(`  ${C.green + C.bold}  Final Price  →  ₹${details.finalPrice} / unit${C.reset}`);
            console.log(`  ${C.bold}  Quantity     →  ${details.quantity.toLocaleString()} units${C.reset}`);

            const total = details.totalCost ?? details.totalRevenue;
            if (total) {
                console.log(`  ${C.bold}  Total Value  →  ₹${total.toLocaleString()}${C.reset}`);
            }
            if (details.buyerStartPrice !== undefined) {
                const c   = details.finalPrice - details.buyerStartPrice;
                const pct = ((c / details.buyerStartPrice) * 100).toFixed(1);
                console.log(`${C.dim}  Buyer   : ₹${details.buyerStartPrice} → ₹${details.finalPrice}  (conceded ₹${c}, ${pct}%)${C.reset}`);
            }
            if (details.sellerStartPrice !== undefined) {
                const c   = details.sellerStartPrice - details.finalPrice;
                const pct = ((c / details.sellerStartPrice) * 100).toFixed(1);
                console.log(`${C.dim}  Seller  : ₹${details.sellerStartPrice} → ₹${details.finalPrice}  (conceded ₹${c}, ${pct}%)${C.reset}`);
            }
            if (details.profitMargin !== undefined) {
                console.log(`${C.dim}  Seller profit: ₹${details.profitMargin} / unit${C.reset}`);
            }
        } else {
            console.log(`  ${C.red}  No agreement after ${details.roundsUsed} round(s).${C.reset}`);
        }
        console.log(`  ${C.dim}${"═".repeat(W)}${C.reset}`);
    }

    // ── Escalation terminal notice ────────────────────────────────────────────
    printEscalationNotice(
        buyerFinalOffer:  number,
        sellerFinalOffer: number,
        gap:              number,
        reportPath:       string
    ) {
        this.printPriceTrail();

        console.log("");
        console.log(`  ${C.bgYellow + C.bold + C.white}  ⚠  ESCALATED TO HUMAN${"".padEnd(W - 21)}  ${C.reset}`);
        console.log(`${C.dim}  Gap of ₹${gap} remains after ${this.priceTrail.length} round(s)${C.reset}`);
        console.log(`  ${C.yellow}  Buyer final offer  : ₹${buyerFinalOffer}${C.reset}`);
        console.log(`  ${C.yellow}  Seller final offer : ₹${sellerFinalOffer}${C.reset}`);
        console.log(`  ${C.bold}  Report saved → ${reportPath}${C.reset}`);
        console.log(`  ${C.dim}${"═".repeat(W)}${C.reset}`);
    }

    // ── Seller escalation received notice ─────────────────────────────────────
    printEscalationReceived(gap: number, reportPath: string) {
        console.log("");
        console.log(`  ${C.bgYellow + C.bold + C.white}  ⚠  ESCALATION NOTICE RECEIVED${"".padEnd(W - 29)}  ${C.reset}`);
        console.log(`${C.dim}  Buyer could not close ₹${gap} gap — human review requested${C.reset}`);
        console.log(`  ${C.bold}  Report → ${reportPath}${C.reset}`);
        console.log(`  ${C.dim}${"═".repeat(W)}${C.reset}`);
    }

    // ── Write escalation report to disk ──────────────────────────────────────
    saveEscalationReport(params: {
        buyerFinalOffer:  number;
        sellerFinalOffer: number;
        gap:              number;
        rounds:           number;
        maxRounds:        number;
        quantity:         number;
        deliveryDate:     string;
        logs:             NegotiationLog[];
    }): string {
        // Ensure escalations directory exists next to this file (src/escalations/)
        const escalationsDir = path.resolve(__dirname, "..", "escalations");
        if (!fs.existsSync(escalationsDir)) {
            fs.mkdirSync(escalationsDir, { recursive: true });
        }

        const filePath = path.join(escalationsDir, `${this.negotiationId}_escalation.txt`);
        const now      = new Date();

        const lines: string[] = [];
        const hr = "─".repeat(60);

        lines.push("╔══════════════════════════════════════════════════════════════╗");
        lines.push("║           NEGOTIATION ESCALATION REPORT                     ║");
        lines.push("╚══════════════════════════════════════════════════════════════╝");
        lines.push("");
        lines.push(`Negotiation ID   : ${this.negotiationId}`);
        lines.push(`Date / Time      : ${now.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" })} ${now.toLocaleTimeString()}`);
        lines.push(`Status           : ESCALATED — No agreement reached after ${params.rounds} round(s)`);
        lines.push(`Quantity         : ${params.quantity.toLocaleString()} units`);
        lines.push(`Delivery Date    : ${params.deliveryDate}`);
        lines.push("");
        lines.push(hr);
        lines.push("PRICE TRAIL");
        lines.push(hr);
        lines.push(`${"Rnd".padEnd(8)}${"Buyer".padEnd(14)}${"Seller".padEnd(14)}${"Gap".padEnd(12)}`);
        lines.push("─".repeat(48));

        // Rebuild trail from logs
        const trail = new Map<number, { buyer?: number; seller?: number }>();
        for (const log of params.logs) {
            if (log.offeredPrice === undefined) continue;
            if (!trail.has(log.round)) trail.set(log.round, {});
            const entry = trail.get(log.round)!;
            if (log.from === "BUYER")  entry.buyer  = log.offeredPrice;
            if (log.from === "SELLER") entry.seller = log.offeredPrice;
        }
        for (const [round, e] of [...trail.entries()].sort((a, b) => a[0] - b[0])) {
            const b   = e.buyer  !== undefined ? `₹${e.buyer}`  : "—";
            const s   = e.seller !== undefined ? `₹${e.seller}` : "—";
            const gap = (e.buyer !== undefined && e.seller !== undefined)
                ? `₹${Math.abs(e.seller - e.buyer)}`
                : "—";
            lines.push(`${String(round).padEnd(8)}${b.padEnd(14)}${s.padEnd(14)}${gap.padEnd(12)}`);
        }

        lines.push("");
        lines.push(hr);
        lines.push("FINAL POSITIONS");
        lines.push(hr);
        lines.push(`Buyer's last offer  : ₹${params.buyerFinalOffer}`);
        lines.push(`Seller's last offer : ₹${params.sellerFinalOffer}`);
        lines.push(`Remaining gap       : ₹${params.gap}`);
        lines.push(`Gap as % of seller  : ${((params.gap / params.sellerFinalOffer) * 100).toFixed(1)}%`);

        lines.push("");
        lines.push(hr);
        lines.push("AGENT REASONING — FINAL ROUND");
        lines.push(hr);
        const finalRoundLogs = params.logs.filter(l => l.round === params.rounds);
        for (const log of finalRoundLogs) {
            if (log.reasoning) {
                lines.push(`${log.from.padEnd(8)}: ${log.reasoning}`);
            }
        }

        lines.push("");
        lines.push(hr);
        lines.push("HUMAN ACTION REQUIRED");
        lines.push(hr);
        lines.push("Please review the negotiation above and choose one of:");
        lines.push("");
        lines.push(`  A)  Accept SELLER price     →  ₹${params.sellerFinalOffer} / unit`);
        lines.push(`  B)  Accept BUYER price      →  ₹${params.buyerFinalOffer} / unit`);
        lines.push(`  C)  Split the difference    →  ₹${Math.round((params.buyerFinalOffer + params.sellerFinalOffer) / 2)} / unit`);
        lines.push(`  D)  Reject — do not proceed`);
        lines.push("");
        lines.push(hr);
        lines.push(`Generated : ${now.toISOString()}`);
        lines.push(hr);

        fs.writeFileSync(filePath, lines.join("\n"), "utf8");
        return filePath;
    }

    // ── Purchase Order block ──────────────────────────────────────────────────
    printPurchaseOrder(poData: any) {
        console.log("");
        console.log(`  ${C.blue + C.bold}  📝  PURCHASE ORDER${"".padEnd(W - 19)}  ${C.reset}`);
        console.log(`${C.dim}  ${hline()}${C.reset}`);
        console.log(`${C.dim}  PO ID    : ${poData.poId}${C.reset}`);
        console.log(`  ${C.bold}  Price    →  ₹${poData.terms.pricePerUnit} / unit  ×  ${poData.terms.quantity} units${C.reset}`);
        console.log(`  ${C.bold}  Total    →  ₹${poData.terms.total.toLocaleString()}${C.reset}`);
        console.log(`${C.dim}  Delivery : ${poData.deliveryDate}${C.reset}`);
        console.log(`${C.dim}  ${hline()}${C.reset}`);
    }

    // ── Invoice block ─────────────────────────────────────────────────────────
    printInvoice(invoiceData: any) {
        console.log("");
        console.log(`  ${C.magenta + C.bold}  📄  INVOICE GENERATED${"".padEnd(W - 22)}  ${C.reset}`);
        console.log(`${C.dim}  ${hline()}${C.reset}`);
        console.log(`${C.dim}  Invoice  : ${invoiceData.invoiceId}${C.reset}`);
        console.log(`${C.dim}  PO Ref   : ${invoiceData.poId}${C.reset}`);
        console.log(`  ${C.bold}  Price    →  ₹${invoiceData.terms.pricePerUnit} / unit  ×  ${invoiceData.terms.quantity}${C.reset}`);
        console.log(`${C.dim}  Subtotal : ₹${invoiceData.terms.subtotal.toLocaleString()}${C.reset}`);
        console.log(`${C.dim}  GST 18%  : ₹${invoiceData.terms.tax.toLocaleString()}${C.reset}`);
        console.log(`  ${C.green + C.bold}  TOTAL    →  ₹${invoiceData.terms.total.toLocaleString()}${C.reset}`);
        console.log(`${C.dim}  Payment  : ${invoiceData.paymentTerms}${C.reset}`);
        console.log(`${C.dim}  Delivery : ${invoiceData.deliveryDate}${C.reset}`);
        console.log(`${C.dim}  ${hline()}${C.reset}`);
    }

    // ── Dynamic Discounting: DD_OFFER ────────────────────────────────────────
    printDDOffer(data: any) {
        const pct = (data.maxDiscountRate * 100).toFixed(3);
        const propPct = (data.discountAtProposedDate.appliedRate * 100).toFixed(3);
        console.log("");
        console.log(`  ${C.cyan + C.bold}  💰  DD OFFER SENT${''.padEnd(W - 18)}  ${C.reset}`);
        console.log(`${C.dim}  ${hline()}${C.reset}`);
        console.log(`${C.dim}  Invoice      : ${data.invoiceId}${C.reset}`);
        console.log(`${C.dim}  Original     : ₹${data.originalTotal.toLocaleString()}${C.reset}`);
        console.log(`  ${C.bold}  Max DD Rate  →  ${pct}% (linear)${C.reset}`);
        console.log(`  ${C.bold}  Proposed pay : ${data.proposedSettlementDate}  (${data.discountAtProposedDate.daysEarly} days early)${C.reset}`);
        console.log(`  ${C.green + C.bold}  If accepted  →  ₹${data.discountAtProposedDate.discountedAmount.toLocaleString()}  (save ₹${data.discountAtProposedDate.savingAmount.toLocaleString()} @ ${propPct}%)${C.reset}`);
        console.log(`${C.dim}  ${hline()}${C.reset}`);
    }

    // ── Dynamic Discounting: DD_ACCEPT ───────────────────────────────────────
    printDDAccept(data: any) {
        console.log("");
        console.log(`  ${C.green + C.bold}  ✓  DD ACCEPTED BY BUYER${''.padEnd(W - 24)}  ${C.reset}`);
        console.log(`${C.dim}  ${hline()}${C.reset}`);
        console.log(`${C.dim}  Invoice    : ${data.invoiceId}${C.reset}`);
        console.log(`  ${C.bold}  Settlement : ${data.chosenSettlementDate}${C.reset}`);
        console.log(`${C.dim}  Computing discounted amount and submitting to ACTUS...${C.reset}`);
        console.log(`${C.dim}  ${hline()}${C.reset}`);
    }

    // ── Dynamic Discounting: DD_INVOICE (final) ───────────────────────────────
    printDDInvoice(data: any) {
        const pct = (data.appliedRate * 100).toFixed(3);
        const statusColor = data.actusSimulationStatus === "SUCCESS" ? C.green + C.bold : C.red + C.bold;
        console.log("");
        console.log(`  ${C.magenta + C.bold}  📄  DD INVOICE — FINAL${''.padEnd(W - 23)}  ${C.reset}`);
        console.log(`${C.dim}  ${hline()}${C.reset}`);
        console.log(`${C.dim}  Invoice ID   : ${data.invoiceId}${C.reset}`);
        console.log(`${C.dim}  Original     : ₹${data.originalTotal.toLocaleString()}${C.reset}`);
        console.log(`  ${C.bold}  Applied Rate →  ${pct}%${C.reset}`);
        console.log(`  ${C.green + C.bold}  PAYABLE      →  ₹${data.discountedTotal.toLocaleString()}  (saved ₹${data.savingAmount.toLocaleString()})${C.reset}`);
        console.log(`  ${C.bold}  Settle by   →  ${data.settlementDate}${C.reset}`);
        console.log(`${C.dim}  ACTUS ID     : ${data.actusContractId}${C.reset}`);
        console.log(`  ${statusColor}  ACTUS Status →  ${data.actusSimulationStatus}${data.actusError ? ' — ' + data.actusError : ''}${C.reset}`);
        console.log(`${C.dim}  ${hline()}${C.reset}`);
    }

    getLogs(): NegotiationLog[] {
        return this.logs;
    }
}
