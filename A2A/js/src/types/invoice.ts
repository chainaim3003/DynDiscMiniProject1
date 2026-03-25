// Invoice Schema with required attributes
export interface InvoiceSchema {
    amount: number;                    // e.g., 5000.00
    currency: string;                  // e.g., "USD", "EUR"
    dueDate: string;                   // ISO 8601 format: "2025-12-31"
    refUri: RefUri;                    // Reference URI
    destinationAccount: DestinationAccount;
}

// Reference URI - can be transaction hash, IPFS link, or S3 link
export type RefUri =
    | { type: 'transaction_hash'; value: string }
    | { type: 'ipfs_encrypted'; value: string }
    | { type: 's3_storage'; value: string };

// Destination account for digital asset payment
export interface DestinationAccount {
    type: 'digital_asset';
    chainId: string;                   // e.g., "ethereum-mainnet"
    walletAddress: string;             // e.g., "0x742d35Cc..."
}

// Invoice message wrapper
export interface InvoiceMessage {
    invoiceId: string;
    invoice: InvoiceSchema;
    timestamp: string;
    senderAgent: {
        name: string;
        agentAID: string;
    };
}

// ============================================================================
// PURCHASE ORDER (Buyer → Seller) - Stage 1: Buyer initiates trade
// ============================================================================
export interface PurchaseOrderMessage {
    poId: string;                      // e.g., "PO-20251120-ABC123"
    stage: 1;                          // Always 1 for PO
    buyer: {
        name: string;                  // Buyer agent name
        agentAID: string;              // Buyer agent AID
        walletAddress: string;         // Buyer's wallet (for receiving goods)
    };
    seller: {
        name: string;                  // Seller agent name (target)
        agentAID: string;              // Seller agent AID
        walletAddress: string;         // Seller's wallet (for receiving payment)
    };
    orderDetails: {
        items: Array<{
            description: string;
            quantity: number;
            unitPrice: number;
            totalPrice: number;
        }>;
        currency: string;              // "ALGO", "USDC", etc.
        totalAmount: number;           // Total trade amount
        deliveryDate: string;          // ISO 8601 format
        paymentTerms: string;          // Description of payment terms
    };
    paymentStages: {
        poStage: {
            percent: number;           // e.g., 20
            amount: number;            // e.g., 2.0 ALGO
            description: string;
        };
        invoiceStage: {
            percent: number;           // e.g., 50
            amount: number;            // e.g., 5.0 ALGO
            description: string;
        };
        receiptStage: {
            percent: number;           // e.g., 30
            amount: number;            // e.g., 3.0 ALGO
            description: string;
        };
    };
    destinationAccount: DestinationAccount; // Where buyer will send payment
    timestamp: string;                 // ISO 8601 format
    attachments?: Array<{              // Optional: attach PO.json or other files
        filename: string;
        mimeType: string;
        data: string;                  // Base64 encoded file data
    }>;
}

// ============================================================================
// PO ACCEPTANCE (Seller → Buyer) - Response to Purchase Order
// ============================================================================
export interface POAcceptanceMessage {
    acceptanceId: string;              // e.g., "POA-20251120-XYZ789"
    poId: string;                      // Reference to original PO
    status: 'accepted' | 'rejected' | 'pending';
    seller: {
        name: string;                  // Seller agent name
        agentAID: string;              // Seller agent AID
        walletAddress: string;         // Seller's payment wallet
    };
    buyer: {
        name: string;                  // Buyer agent name (from original PO)
        agentAID: string;              // Buyer agent AID
    };
    acceptanceDetails?: {
        estimatedDeliveryDate?: string; // ISO 8601 format
        paymentInstructions?: string;   // Additional payment details
        notes?: string;                 // Any seller notes
    };
    rejectionReason?: string;          // If status is 'rejected'
    timestamp: string;                 // ISO 8601 format
    nextStage?: {
        stage: 'invoice' | 'warehouse_receipt';
        expectedDate?: string;
    };
}

// Warehouse Receipt Message (Stage 3: 30% payment)
export interface WarehouseReceiptMessage {
    receiptId: string;                 // e.g., "WR-20251120-XYZ789"
    stage: 3;                          // Always 3 for Receipt
    amount: number;                    // 30% of trade total
    currency: string;                  // "ALGO" or "USDC"
    tradeTotal: number;                // Full trade amount
    destinationAccount: DestinationAccount;
    senderAgent: {
        name: string;
        agentAID: string;
        oorAID?: string;
    };
    timestamp?: string;
}

// ============================================================================
// NEGOTIATION MESSAGES (Pre-PO Stage)
// ============================================================================

// Negotiation Initiate (Buyer → Seller) - Start negotiation
export interface NegotiationInitiateMessage {
    negotiationId: string;             // e.g., "NEG-20251225-ABC123"
    buyerOffer: number;                // Buyer's initial price offer
    currency: string;                  // "ALGO" or "USDC"
    buyer: {
        name: string;
        agentAID: string;
        walletAddress: string;
    };
    seller: {
        name: string;
        agentAID: string;
    };
    itemDescription: string;           // What's being purchased
    quantity: number;
    timestamp: string;
    round: number;                     // Always 1 for initiation
}

// Negotiation Counter (Seller → Buyer or Buyer → Seller) - Counter-offer
export interface NegotiationCounterMessage {
    negotiationId: string;             // Reference to original negotiation
    counterOffer: number;              // New price offer
    currency: string;
    previousOffer: number;             // Previous price for reference
    round: number;                     // Current round (1-3)
    counterParty: {
        name: string;
        agentAID: string;
    };
    timestamp: string;
    message?: string;                  // Optional negotiation message
}

// Negotiation Accept (Either party) - Accept current offer
export interface NegotiationAcceptMessage {
    negotiationId: string;
    acceptedPrice: number;
    currency: string;
    round: number;                     // Round when accepted
    acceptedBy: {
        name: string;
        agentAID: string;
    };
    timestamp: string;
    message?: string;
}

// Negotiation Reject (Either party) - Reject and end negotiation
export interface NegotiationRejectMessage {
    negotiationId: string;
    rejectedPrice: number;
    currency: string;
    round: number;                     // Round when rejected
    rejectedBy: {
        name: string;
        agentAID: string;
    };
    timestamp: string;
    reason?: string;
}

// Union type for all trade documents
export type TradeDocumentMessage = 
    | PurchaseOrderMessage 
    | POAcceptanceMessage
    | InvoiceMessage 
    | WarehouseReceiptMessage
    | NegotiationInitiateMessage
    | NegotiationCounterMessage
    | NegotiationAcceptMessage
    | NegotiationRejectMessage;
