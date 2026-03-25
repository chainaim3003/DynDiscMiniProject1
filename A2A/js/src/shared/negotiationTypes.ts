// src/agents/shared/negotiationTypes.ts

/* ===========================
   COMMON ENUMS
=========================== */

export type NegotiationStatus =
  | "negotiating"
  | "accepted"
  | "rejected";

export type NegotiationMessageType =
  | "NEGOTIATION_INIT"
  | "COUNTER_OFFER"
  | "ACCEPT_OFFER"
  | "REJECT_OFFER"
  | "PURCHASE_ORDER";

/* ===========================
   NEGOTIATION MESSAGES
=========================== */

export interface NegotiationInitMessage {
  type: "NEGOTIATION_INIT";
  negotiationId: string;
  quantity: number;
  pricePerUnit: number;
  deliveryDate: string; // ISO date
  round: number;
}

export interface CounterOfferMessage {
  type: "COUNTER_OFFER";
  negotiationId: string;
  pricePerUnit: number;
  deliveryDate: string;
  round: number;
}

export interface AcceptOfferMessage {
  type: "ACCEPT_OFFER";
  negotiationId: string;
  pricePerUnit: number;
  deliveryDate: string;
}

export interface RejectOfferMessage {
  type: "REJECT_OFFER";
  negotiationId: string;
  reason: string;
}

/* ===========================
   PURCHASE ORDER
=========================== */

export interface PurchaseOrderMessage {
  type: "PURCHASE_ORDER";
  poId: string;
  negotiationId: string;
  quantity: number;
  pricePerUnit: number;
  deliveryDate: string;
  total: number;
}

/* ===========================
   UNION (VERY IMPORTANT)
=========================== */

export type NegotiationMessage =
  | NegotiationInitMessage
  | CounterOfferMessage
  | AcceptOfferMessage
  | RejectOfferMessage
  | PurchaseOrderMessage;
