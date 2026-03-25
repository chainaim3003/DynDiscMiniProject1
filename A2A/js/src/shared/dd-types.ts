// ================= DYNAMIC DISCOUNTING MESSAGE TYPES =================

export interface DDOfferData {
    type: "DD_OFFER";
    invoiceId: string;
    negotiationId: string;
    invoiceDate: string;            // ISO date  e.g. "2026-03-26"
    dueDate: string;                // invoiceDate + paymentTermsDays
    originalTotal: number;          // full invoice total (incl. tax)
    maxDiscountRate: number;        // safeDDRate computed from seller margin
    paymentTermsDays: number;       // e.g. 30 for Net 30
    proposedSettlementDate: string; // seller suggests: invoiceDate + 10 days
    discountAtProposedDate: {
        daysEarly: number;
        totalDays: number;
        appliedRate: number;        // e.g. 0.0093
        discountedAmount: number;
        savingAmount: number;
    };
}

export interface DDAcceptData {
    type: "DD_ACCEPT";
    invoiceId: string;
    negotiationId: string;
    chosenSettlementDate: string;   // buyer may accept proposed or pick a later date
    from: "BUYER";
}

export interface DDInvoiceData {
    type: "DD_INVOICE";
    invoiceId: string;
    negotiationId: string;
    originalTotal: number;
    discountedTotal: number;
    savingAmount: number;
    appliedRate: number;
    settlementDate: string;
    dueDate: string;
    actusContractId: string;
    actusScenarioId: string;
    actusSimulationStatus: "SUCCESS" | "FAILED";
    actusError?: string;
}
