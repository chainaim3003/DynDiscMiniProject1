// ================= DYNAMIC DISCOUNTING CALCULATOR =================
// Pure math — no I/O, no side effects.

export interface LinearDDResult {
    daysEarly:        number;
    totalDays:        number;
    appliedRate:      number;   // fraction, e.g. 0.0093
    discountedAmount: number;
    savingAmount:     number;
}

/**
 * Compute the maximum safe discount rate the seller can offer.
 *
 *   profitPerUnit = agreedPrice − marginPrice
 *   maxRate       = profitPerUnit / agreedPrice       (100% of margin as discount)
 *   safeDDRate    = maxRate × safetyFactor            (default: give away 50%)
 *
 * safetyFactor = 0.5 means the seller keeps at least half the negotiated profit
 * even in the best-case (buyer pays on day 0).
 */
export function computeSafeDDRate(
    agreedPrice:   number,
    marginPrice:   number,
    safetyFactor:  number = 0.5
): number {
    const profitPerUnit = agreedPrice - marginPrice;
    if (profitPerUnit <= 0) return 0;
    const maxRate = profitPerUnit / agreedPrice;
    return parseFloat((maxRate * safetyFactor).toFixed(6));
}

/**
 * Compute the discounted payable under the LINEAR function.
 *
 *   totalDays      = dueDate − invoiceDate
 *   daysEarly      = dueDate − settlementDate   (clamped ≥ 0)
 *   appliedRate    = maxDiscountRate × (daysEarly / totalDays)
 *   discountedAmt  = originalTotal × (1 − appliedRate)
 */
export function computeLinearDiscount(
    originalTotal:    number,
    maxDiscountRate:  number,
    invoiceDate:      string,   // ISO date or datetime
    dueDate:          string,
    settlementDate:   string
): LinearDDResult {
    const MS_PER_DAY = 86_400_000;

    const inv  = new Date(invoiceDate).getTime();
    const due  = new Date(dueDate).getTime();
    const setl = new Date(settlementDate).getTime();

    const totalDays = Math.max(1, Math.round((due  - inv)  / MS_PER_DAY));
    const daysEarly = Math.max(0,  Math.round((due  - setl) / MS_PER_DAY));

    const appliedRate      = maxDiscountRate * (daysEarly / totalDays);
    const discountedAmount = parseFloat((originalTotal * (1 - appliedRate)).toFixed(2));
    const savingAmount     = parseFloat((originalTotal - discountedAmount).toFixed(2));

    return {
        daysEarly,
        totalDays,
        appliedRate:      parseFloat(appliedRate.toFixed(6)),
        discountedAmount,
        savingAmount,
    };
}

/**
 * Add N calendar days to an ISO date string.
 * Returns "YYYY-MM-DD".
 */
export function addDays(isoDate: string, days: number): string {
    const d = new Date(isoDate);
    d.setDate(d.getDate() + days);
    return d.toISOString().split("T")[0];
}
