// ================= SHARED NEGOTIATION TYPES =================

export type NegotiationStatus =
    | "INITIATED"
    | "NEGOTIATING"
    | "ACCEPTED"
    | "COMPLETED"
    | "FAILED"
    | "REJECTED"
    | "ESCALATED"
    | "DD_COMPLETED";   // negotiation + dynamic discounting fully settled

export type NegotiationAction = "OFFER" | "COUNTER_OFFER" | "ACCEPT" | "REJECT";

export type AgentRole = "BUYER" | "SELLER";

// ================= NEGOTIATION DATA SCHEMAS =================

export interface NegotiationDataBase {
    negotiationId: string;
    round: number;
    timestamp: string;
}

export interface OfferData extends NegotiationDataBase {
    type: "OFFER";
    pricePerUnit: number;
    quantity: number;
    from: AgentRole;
    deliveryDate: string;
}

export interface CounterOfferData extends NegotiationDataBase {
    type: "COUNTER_OFFER";
    pricePerUnit: number;
    previousPrice: number;
    from: AgentRole;
    reasoning?: string;
}

export interface AcceptanceData extends NegotiationDataBase {
    type: "ACCEPT_OFFER";
    acceptedPrice: number;
    from: AgentRole;
    finalTerms: {
        pricePerUnit: number;
        quantity: number;
        totalAmount: number;
        deliveryDate: string;
    };
}

export interface RejectionData extends NegotiationDataBase {
    type: "REJECT_OFFER";
    from: AgentRole;
    reason: string;
    finalRound: boolean;
}

export interface EscalationNoticeData extends NegotiationDataBase {
    type: "ESCALATION_NOTICE";
    from: AgentRole;
    buyerFinalOffer:  number;
    sellerFinalOffer: number;
    gap:              number;
    reportPath:       string;
}

export interface InvoiceData {
    type: "INVOICE";
    invoiceId: string;
    negotiationId: string;
    poId: string;
    invoiceDate: string;
    terms: {
        pricePerUnit: number;
        quantity: number;
        subtotal: number;
        tax: number;
        total: number;
    };
    paymentTerms: string;
    deliveryDate: string;
}

export interface PurchaseOrderData {
    type: "PURCHASE_ORDER";
    poId: string;
    negotiationId: string;
    orderDate: string;
    terms: {
        pricePerUnit: number;
        quantity: number;
        total: number;
    };
    deliveryDate: string;
}

// ================= DYNAMIC DISCOUNTING MESSAGE TYPES =================

export interface DDOfferData {
    type: "DD_OFFER";
    invoiceId: string;
    negotiationId: string;
    invoiceDate: string;
    dueDate: string;
    originalTotal: number;
    maxDiscountRate: number;
    paymentTermsDays: number;
    proposedSettlementDate: string;
    discountAtProposedDate: {
        daysEarly: number;
        totalDays: number;
        appliedRate: number;
        discountedAmount: number;
        savingAmount: number;
    };
}

export interface DDAcceptData {
    type: "DD_ACCEPT";
    invoiceId: string;
    negotiationId: string;
    chosenSettlementDate: string;
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

export type NegotiationData =
    | OfferData
    | CounterOfferData
    | AcceptanceData
    | RejectionData
    | EscalationNoticeData
    | InvoiceData
    | PurchaseOrderData
    | DDOfferData
    | DDAcceptData
    | DDInvoiceData;

// ================= STATE MANAGEMENT =================

export interface RoundHistory {
    round: number;
    buyerOffer?: number;
    sellerOffer?: number;
    buyerAction?: NegotiationAction;
    sellerAction?: NegotiationAction;
    timestamp: string;
    reasoning?: string;
}

export interface BuyerNegotiationState {
    negotiationId: string;
    contextId: string;
    status: NegotiationStatus;

    // Parameters
    targetQuantity: number;
    maxBudget: number;
    deliveryDate: string;

    // Round tracking
    currentRound: number;
    maxRounds: number;

    // History
    history: RoundHistory[];

    // Current state
    lastBuyerOffer?: number;
    lastSellerOffer?: number;

    // Final agreement
    agreedPrice?: number;
    totalCost?: number;

    // Strategy
    strategyParams: {
        aggressiveness: number; // 0-1
        riskTolerance: number; // 0-1
        initialOfferRange: { min: number; max: number };
    };
}

export interface SellerNegotiationState {
    negotiationId: string;
    contextId: string;
    status: NegotiationStatus;

    // Business constraints (PRIVATE)
    marginPrice: number;
    targetProfitPercentage: number;

    // Parameters
    quantity: number;
    deliveryDate: string;

    // Round tracking
    currentRound: number;
    maxRounds: number;

    // History
    history: RoundHistory[];

    // Current state
    lastBuyerOffer?: number;
    lastSellerOffer?: number;

    // Final agreement
    agreedPrice?: number;
    profitPerUnit?: number;
    totalRevenue?: number;

    // Strategy
    strategyParams: {
        flexibility: number; // 0-1
        dealPriority: number; // 0-1
        minProfitMargin: number;
    };
}

// ================= DECISION MAKING =================

export interface NegotiationDecision {
    action: "ACCEPT" | "COUNTER" | "REJECT";
    price?: number;
    reasoning: string;
}

export interface LLMResponse {
    action: "ACCEPT" | "COUNTER" | "REJECT";
    price?: number;
    reasoning: string;
    confidence?: number;
}

// ================= LOGGING =================

export interface NegotiationLog {
    timestamp: string;
    negotiationId: string;
    round: number;
    messageType: string;
    from: AgentRole;

    // Price information
    offeredPrice?: number;
    previousPrice?: number;
    priceMovement?: number;
    priceMovementPercent?: number;

    // Decision context
    decision: NegotiationAction;
    reasoning?: string;

    // Metrics
    gap?: number;
    gapClosed?: number;
}
