// ================= SELLER AGENT WITH HYBRID LLM + RULE-BASED DECISION MAKING =================
import express from "express";
import cors from "cors";
import { v4 as uuidv4 } from "uuid";
import fs from "fs";
import path from "path";
import dotenv from "dotenv";
import { fileURLToPath } from "url";

import { A2AClient } from "@a2a-js/sdk/client";
import {
  AgentCard,
  TaskStatusUpdateEvent,
  Message,
  MessageSendParams,
} from "@a2a-js/sdk";

import {
  InMemoryTaskStore,
  AgentExecutor,
  RequestContext,
  ExecutionEventBus,
  DefaultRequestHandler,
} from "@a2a-js/sdk/server";

import { A2AExpressApp } from "@a2a-js/sdk/server/express";

import {
  SellerNegotiationState,
  NegotiationDecision,
  OfferData,
  CounterOfferData,
  AcceptanceData,
  EscalationNoticeData,
  InvoiceData,
  PurchaseOrderData,
  NegotiationData,
  DDOfferData,
  DDAcceptData,
} from "../../shared/negotiation-types.js";

import { LLMNegotiationClient, LLMPromptContext } from "../../shared/llm-client.js";
import { NegotiationLogger, logInternal, suppressSDKNoise } from "../../shared/logger.js";
import { computeSafeDDRate, computeLinearDiscount, addDays } from "../../shared/dd-calculator.js";
import { ActusClient } from "../../shared/actus-client.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, ".env") });

// Suppress @a2a-js/sdk internal stdout noise (ResultManager logs etc.)
suppressSDKNoise();

// ================= SELLER AGENT CONFIGURATION =================
const SELLER_CONFIG = {
  marginPrice:            350,   // PRIVATE — never go below this
  targetProfitPercentage: 0.1,   // 10%
  maxRounds:              3,
  strategyParams: {
    flexibility:     0.5,
    dealPriority:    0.7,
    minProfitMargin: 5,
  },
  // Dynamic Discounting config
  dd: {
    paymentTermsDays:        30,   // Net 30
    proposedEarlyPayDays:    10,   // seller proposes buyer pays within 10 days
    safetyFactor:            0.5,  // give away at most 50% of profit as discount
    hurdleRateAnnualized:    0.075,
  },
};

const TARGET_PRICE = Math.round(
  SELLER_CONFIG.marginPrice * (1 + SELLER_CONFIG.targetProfitPercentage)
);

// ================= SELLER AGENT EXECUTOR =================
class SellerAgentExecutor implements AgentExecutor {
  private negotiations = new Map<string, SellerNegotiationState>();
  private loggers      = new Map<string, NegotiationLogger>();
  private llmClient: LLMNegotiationClient;
  private actusClient: ActusClient;

  constructor() {
    this.llmClient   = new LLMNegotiationClient();
    this.actusClient = new ActusClient();
  }

  async cancelTask(taskId: string): Promise<void> {
    logInternal(`Task cancellation requested: ${taskId}`);
  }

  // ================= MAIN EXECUTION =================
  async execute(ctx: RequestContext, bus: ExecutionEventBus) {
    const taskId    = ctx.task?.id        || uuidv4();
    const contextId = ctx.task?.contextId || uuidv4();

    const dataParts = ctx.userMessage.parts.filter((p) => p.kind === "data");

    if (dataParts.length === 0) {
      this.respond(bus, taskId, contextId, "🏪 Seller Agent Ready. Waiting for buyer...");
      return;
    }

    const data = (dataParts[0] as any).data as NegotiationData;

    switch (data.type) {
      case "OFFER":
        await this.handleBuyerOffer(data as OfferData, contextId, bus, taskId);
        break;
      case "COUNTER_OFFER":
        await this.handleBuyerCounterOffer(data as CounterOfferData, contextId, bus, taskId);
        break;
      case "ACCEPT_OFFER":
        await this.handleBuyerAcceptance(data as AcceptanceData, contextId, bus, taskId);
        break;
      case "PURCHASE_ORDER":
        await this.handlePurchaseOrder(data as PurchaseOrderData, contextId, bus, taskId);
        break;
      case "ESCALATION_NOTICE":
        await this.handleEscalationNotice(data as EscalationNoticeData, contextId, bus, taskId);
        break;
      case "DD_ACCEPT":
        await this.handleDDAccept(data as DDAcceptData, contextId, bus, taskId);
        break;
      default:
        logInternal(`Unknown message type: ${(data as any).type}`);
    }
  }

  // ================= HANDLE BUYER INITIAL OFFER =================
  private async handleBuyerOffer(
    data: OfferData,
    contextId: string,
    bus: ExecutionEventBus,
    taskId: string
  ) {
    const { negotiationId, pricePerUnit, quantity, deliveryDate } = data;

    const logger = new NegotiationLogger(negotiationId, "SELLER");
    this.loggers.set(negotiationId, logger);

    logger.printSessionHeader(contextId);
    logger.printRoundHeader(1, SELLER_CONFIG.maxRounds);

    logger.log({
      round:        1,
      messageType:  "OFFER",
      from:         "BUYER",
      offeredPrice: pricePerUnit,
      decision:     "OFFER",
    });

    const state: SellerNegotiationState = {
      negotiationId,
      contextId,
      status:                 "NEGOTIATING",
      marginPrice:            SELLER_CONFIG.marginPrice,
      targetProfitPercentage: SELLER_CONFIG.targetProfitPercentage,
      quantity,
      deliveryDate,
      currentRound:           1,
      maxRounds:              SELLER_CONFIG.maxRounds,
      history:                [],
      lastBuyerOffer:         pricePerUnit,
      strategyParams:         SELLER_CONFIG.strategyParams,
    };

    this.negotiations.set(negotiationId, state);

    const decision = await this.makeNegotiationDecision(state);

    if (decision.action === "ACCEPT") {
      await this.sendAcceptance(state, logger, contextId);
      this.respond(
        bus, taskId, contextId,
        `✓ Accepting buyer's offer: ₹${pricePerUnit}/unit\nProfit: ₹${pricePerUnit - SELLER_CONFIG.marginPrice}/unit\nWaiting for buyer confirmation...`
      );
    } else if (decision.action === "COUNTER") {
      await this.sendCounterOffer(state, decision.price!, decision.reasoning, logger, contextId);
      this.respond(
        bus, taskId, contextId,
        `↓ Counter-offer sent: ₹${decision.price}/unit  (buyer offered ₹${pricePerUnit})\nWaiting for buyer response...`
      );
    } else {
      state.status = "REJECTED";
      this.respond(bus, taskId, contextId, "✗ Offer rejected — below acceptable threshold");
    }
  }

  // ================= HANDLE BUYER COUNTER OFFER =================
  private async handleBuyerCounterOffer(
    data: CounterOfferData,
    contextId: string,
    bus: ExecutionEventBus,
    taskId: string
  ) {
    const state  = this.negotiations.get(data.negotiationId);
    const logger = this.loggers.get(data.negotiationId);

    if (!state || !logger) {
      logInternal(`Negotiation state not found: ${data.negotiationId}`);
      return;
    }

    state.lastBuyerOffer = data.pricePerUnit;

    // ── Use seller's own last offer as the baseline for delta display ─────────
    const sellerLastPrice      = state.lastSellerOffer;
    const priceMovement        = sellerLastPrice !== undefined
      ? data.pricePerUnit - sellerLastPrice
      : 0;
    const priceMovementPercent = sellerLastPrice !== undefined && sellerLastPrice !== 0
      ? (priceMovement / sellerLastPrice) * 100
      : 0;

    logger.log({
      round:                state.currentRound,
      messageType:          "COUNTER_OFFER",
      from:                 "BUYER",
      offeredPrice:         data.pricePerUnit,
      previousPrice:        sellerLastPrice,
      priceMovement,
      priceMovementPercent,
      decision:             "COUNTER_OFFER",
      reasoning:            data.reasoning,
    });

    const currentHistory = state.history.find((h) => h.round === state.currentRound);
    if (currentHistory) {
      currentHistory.buyerOffer  = data.pricePerUnit;
      currentHistory.buyerAction = "COUNTER_OFFER";
    }

    state.currentRound += 1;

    if (state.currentRound > state.maxRounds) {
      // Seller side just waits — buyer will send ESCALATION_NOTICE shortly
      state.status = "ESCALATED";
      logInternal(`Max rounds reached — awaiting escalation notice from buyer`);
      this.respond(bus, taskId, contextId, "⚠ Max rounds reached — awaiting escalation notice...");
      return;
    }

    logger.printRoundHeader(state.currentRound, state.maxRounds);

    const decision = await this.makeNegotiationDecision(state);

    if (decision.action === "ACCEPT") {
      await this.sendAcceptance(state, logger, contextId);
      const profit = data.pricePerUnit - SELLER_CONFIG.marginPrice;
      this.respond(
        bus, taskId, contextId,
        `✓ Accepting buyer's offer: ₹${data.pricePerUnit}/unit\nProfit: ₹${profit}/unit (${((profit / SELLER_CONFIG.marginPrice) * 100).toFixed(1)}%)\nWaiting for buyer confirmation...`
      );
    } else if (decision.action === "COUNTER") {
      await this.sendCounterOffer(state, decision.price!, decision.reasoning, logger, contextId);
      this.respond(
        bus, taskId, contextId,
        `↓ Counter-offer sent: ₹${decision.price}/unit  (Round ${state.currentRound}/${state.maxRounds})\nWaiting for buyer response...`
      );
    } else {
      state.status = "REJECTED";
      logger.printNegotiationSummary("FAILED", {
        roundsUsed: state.currentRound,
        maxRounds:  state.maxRounds,
        quantity:   state.quantity,
      });
      this.respond(bus, taskId, contextId, "✗ Offer rejected — below margin price");
    }
  }

  // ================= HANDLE ESCALATION NOTICE =================
  private async handleEscalationNotice(
    data: EscalationNoticeData,
    contextId: string,
    bus: ExecutionEventBus,
    taskId: string
  ) {
    const logger = this.loggers.get(data.negotiationId);
    const state  = this.negotiations.get(data.negotiationId);

    if (state)  state.status = "ESCALATED";

    if (logger) {
      logger.printEscalationReceived(data.gap, data.reportPath);
    } else {
      logInternal(`Escalation received for ${data.negotiationId} — gap ₹${data.gap} — report: ${data.reportPath}`);
    }

    this.respond(
      bus, taskId, contextId,
      `⚠ Negotiation escalated to human review.\nGap: ₹${data.gap}  |  Report: ${data.reportPath}`
    );
  }

  // ================= HANDLE BUYER ACCEPTANCE =================
  private async handleBuyerAcceptance(
    data: AcceptanceData,
    contextId: string,
    bus: ExecutionEventBus,
    taskId: string
  ) {
    const state  = this.negotiations.get(data.negotiationId);
    const logger = this.loggers.get(data.negotiationId);

    if (!state || !logger) {
      logInternal(`Negotiation state not found: ${data.negotiationId}`);
      return;
    }

    if (state.status === "COMPLETED" || state.status === "ACCEPTED") {
      logInternal(`Ignoring duplicate acceptance — already ${state.status}`);
      return;
    }

    logger.log({
      round:        state.currentRound,
      messageType:  "ACCEPT",
      from:         "BUYER",
      offeredPrice: data.acceptedPrice,
      decision:     "ACCEPT",
      reasoning:    "Buyer accepted our offer",
    });

    state.agreedPrice   = data.acceptedPrice;
    state.profitPerUnit = data.acceptedPrice - SELLER_CONFIG.marginPrice;
    state.totalRevenue  = data.acceptedPrice * state.quantity;
    state.status        = "ACCEPTED";

    const acceptanceData: AcceptanceData = {
      type:          "ACCEPT_OFFER",
      negotiationId: state.negotiationId,
      round:         state.currentRound,
      timestamp:     new Date().toISOString(),
      acceptedPrice: data.acceptedPrice,
      from:          "SELLER",
      finalTerms: {
        pricePerUnit: data.acceptedPrice,
        quantity:     state.quantity,
        totalAmount:  state.totalRevenue,
        deliveryDate: state.deliveryDate,
      },
    };

    logger.log({
      round:        state.currentRound,
      messageType:  "ACCEPT",
      from:         "SELLER",
      offeredPrice: data.acceptedPrice,
      decision:     "ACCEPT",
      reasoning:    "bilateral acceptance rule",
    });

    await this.sendToBuyer(acceptanceData, contextId);

    const buyerStart  = state.history[0]?.buyerOffer;
    const sellerStart = state.history[0]?.sellerOffer;

    logger.printNegotiationSummary("COMPLETED", {
      roundsUsed:       state.currentRound,
      maxRounds:        state.maxRounds,
      finalPrice:       data.acceptedPrice,
      buyerStartPrice:  buyerStart,
      sellerStartPrice: sellerStart,
      totalRevenue:     state.totalRevenue,
      profitMargin:     state.profitPerUnit,
      quantity:         state.quantity,
    });

    state.status = "COMPLETED";

    this.respond(
      bus, taskId, contextId,
      `✓✓ Deal Closed!\n\nFinal Price    : ₹${data.acceptedPrice}/unit\nProfit         : ₹${state.profitPerUnit}/unit\nTotal Revenue  : ₹${state.totalRevenue?.toLocaleString()}\nWaiting for Purchase Order...`
    );
  }

  // ================= HANDLE PURCHASE ORDER =================
  private async handlePurchaseOrder(
    data: PurchaseOrderData,
    contextId: string,
    bus: ExecutionEventBus,
    taskId: string
  ) {
    const state  = this.negotiations.get(data.negotiationId);
    const logger = this.loggers.get(data.negotiationId);

    if (!state || !logger) {
      logInternal(`Negotiation state not found: ${data.negotiationId}`);
      return;
    }

    logger.printPurchaseOrder(data);

    // ── Step 1: Send the standard full invoice (existing behavior) ────────────
    const invoiceId = `INV-${Date.now()}`;
    await this.sendInvoice(state, data.poId, invoiceId, logger, contextId);
    state.status = "COMPLETED";

    // ── Step 2: Compute safe DD rate from seller's margin ─────────────────────
    const agreedPrice = state.agreedPrice!;
    const safeDDRate  = computeSafeDDRate(
      agreedPrice,
      SELLER_CONFIG.marginPrice,
      SELLER_CONFIG.dd.safetyFactor
    );

    if (safeDDRate <= 0) {
      logInternal("DD rate is 0 (no profit margin) — skipping DD offer");
      this.respond(bus, taskId, contextId, "📄 Invoice sent to buyer\nNegotiation completed successfully!");
      return;
    }

    // ── Step 3: Build DD_OFFER payload ────────────────────────────────────────
    const invoiceDate            = new Date().toISOString().split("T")[0];
    const dueDate                = addDays(invoiceDate, SELLER_CONFIG.dd.paymentTermsDays);
    const proposedSettlementDate = addDays(invoiceDate, SELLER_CONFIG.dd.proposedEarlyPayDays);

    const subtotal    = agreedPrice * state.quantity;
    const tax         = Math.round(subtotal * 0.18);
    const totalAmount = subtotal + tax;

    const discountAtProposed = computeLinearDiscount(
      totalAmount,
      safeDDRate,
      invoiceDate,
      dueDate,
      proposedSettlementDate
    );

    const ddOfferData: DDOfferData = {
      type:                    "DD_OFFER",
      invoiceId,
      negotiationId:           state.negotiationId,
      invoiceDate,
      dueDate,
      originalTotal:           totalAmount,
      maxDiscountRate:         safeDDRate,
      paymentTermsDays:        SELLER_CONFIG.dd.paymentTermsDays,
      proposedSettlementDate,
      discountAtProposedDate:  discountAtProposed,
    };

    logger.printDDOffer(ddOfferData);
    await this.sendToBuyer(ddOfferData, contextId);

    this.respond(
      bus, taskId, contextId,
      `📄 Invoice sent\n💰 DD Offer sent — max ${(safeDDRate * 100).toFixed(3)}% discount\n   Pay by ${proposedSettlementDate} → ₹${discountAtProposed.discountedAmount.toLocaleString()} (save ₹${discountAtProposed.savingAmount.toLocaleString()})\nAwaiting buyer's DD acceptance...`
    );
  }

  // ================= HANDLE DD_ACCEPT =================
  private async handleDDAccept(
    data: DDAcceptData,
    contextId: string,
    bus: ExecutionEventBus,
    taskId: string
  ) {
    const state  = this.negotiations.get(data.negotiationId);
    const logger = this.loggers.get(data.negotiationId);

    if (!state || !logger) {
      logInternal(`DD_ACCEPT: negotiation state not found: ${data.negotiationId}`);
      return;
    }

    logger.printDDAccept(data);

    // ── Re-compute discount for the buyer's chosen settlement date ────────────
    const agreedPrice = state.agreedPrice!;
    const safeDDRate  = computeSafeDDRate(
      agreedPrice,
      SELLER_CONFIG.marginPrice,
      SELLER_CONFIG.dd.safetyFactor
    );

    const invoiceDate = new Date().toISOString().split("T")[0];
    const dueDate     = addDays(invoiceDate, SELLER_CONFIG.dd.paymentTermsDays);
    const subtotal    = agreedPrice * state.quantity;
    const tax         = Math.round(subtotal * 0.18);
    const totalAmount = subtotal + tax;

    const ddResult = computeLinearDiscount(
      totalAmount,
      safeDDRate,
      invoiceDate,
      dueDate,
      data.chosenSettlementDate
    );

    // ── Submit to ACTUS ───────────────────────────────────────────────────────
    logInternal(`Submitting DD contract to ACTUS for ${data.invoiceId}...`);

    const actusResult = await this.actusClient.submitDDContract({
      contractId:           data.invoiceId,
      negotiationId:        data.negotiationId,
      invoiceDate,
      dueDate,
      settlementDate:       data.chosenSettlementDate,
      notionalAmount:       totalAmount,
      maxDiscountRate:      safeDDRate,
      hurdleRateAnnualized: SELLER_CONFIG.dd.hurdleRateAnnualized,
      sellerRevenue:        state.totalRevenue ?? totalAmount,
    });

    if (actusResult.success) {
      logInternal(`ACTUS simulation SUCCESS — contractId: ${actusResult.contractId}  scenarioId: ${actusResult.scenarioId}`);
    } else {
      logInternal(`ACTUS simulation FAILED — ${actusResult.error}`);
    }

    // ── Build and send DD_INVOICE ─────────────────────────────────────────────
    const ddInvoice = {
      type:                    "DD_INVOICE",
      invoiceId:               data.invoiceId,
      negotiationId:           data.negotiationId,
      originalTotal:           totalAmount,
      discountedTotal:         ddResult.discountedAmount,
      savingAmount:            ddResult.savingAmount,
      appliedRate:             ddResult.appliedRate,
      settlementDate:          data.chosenSettlementDate,
      dueDate,
      actusContractId:         actusResult.contractId,
      actusScenarioId:         actusResult.scenarioId,
      actusSimulationStatus:   actusResult.success ? "SUCCESS" : "FAILED",
      actusError:              actusResult.error,
    };

    logger.printDDInvoice(ddInvoice);
    await this.sendToBuyer(ddInvoice, contextId);

    state.status = "DD_COMPLETED";

    this.respond(
      bus, taskId, contextId,
      `✅ DD Invoice sent!\n\nOriginal   : ₹${totalAmount.toLocaleString()}\nDiscounted : ₹${ddResult.discountedAmount.toLocaleString()}  (${(ddResult.appliedRate * 100).toFixed(3)}% off)\nSaving     : ₹${ddResult.savingAmount.toLocaleString()}\nSettle by  : ${data.chosenSettlementDate}\nACTUS      : ${actusResult.success ? "✓ Simulation complete" : "⚠ " + actusResult.error}\n\nWorkflow complete!`
    );
  }

  // ================= HYBRID DECISION MAKING =================
  private async makeNegotiationDecision(state: SellerNegotiationState): Promise<NegotiationDecision> {
    const llmDecision       = await this.getLLMDecision(state);
    const validatedDecision = this.applySellerConstraints(llmDecision, state);

    if (!validatedDecision) {
      logInternal("LLM decision invalid — using rule-based fallback");
      return this.ruleBasedDecision(state);
    }
    return validatedDecision;
  }

  private async getLLMDecision(state: SellerNegotiationState): Promise<NegotiationDecision> {
    const context: LLMPromptContext = {
      role:           "SELLER",
      round:          state.currentRound,
      maxRounds:      state.maxRounds,
      lastOwnOffer:   state.lastSellerOffer,
      lastTheirOffer: state.lastBuyerOffer,
      history:        state.history,
      // Pass margin + minProfitMargin as the effective floor so the LLM
      // never reasons about accepting a zero-profit deal.
      constraints:    { marginPrice: state.marginPrice + state.strategyParams.minProfitMargin, quantity: state.quantity },
      targetPrice:    TARGET_PRICE,
    };
    const llmResponse = await this.llmClient.getNegotiationDecision(context);
    return { action: llmResponse.action, price: llmResponse.price, reasoning: llmResponse.reasoning };
  }

  private applySellerConstraints(
    decision: NegotiationDecision,
    state: SellerNegotiationState
  ): NegotiationDecision | null {
    // The minimum we will ever accept — cost floor + required profit buffer.
    const minAcceptable = state.marginPrice + state.strategyParams.minProfitMargin;

    if (decision.action === "ACCEPT") {
      // Use strict < minAcceptable (not < marginPrice) so ₹350 is also rejected.
      if (state.lastBuyerOffer && state.lastBuyerOffer < minAcceptable) {
        logInternal(`Cannot accept ₹${state.lastBuyerOffer} — below min acceptable ₹${minAcceptable} (margin ₹${state.marginPrice} + buffer ₹${state.strategyParams.minProfitMargin})`);
        if (state.currentRound < state.maxRounds) {
          decision.action    = "COUNTER";
          decision.price     = minAcceptable;
          decision.reasoning = `Buyer offer below minimum acceptable ₹${minAcceptable}, countering at floor`;
        } else {
          decision.action    = "REJECT";
          decision.reasoning = `Buyer offer ₹${state.lastBuyerOffer} below minimum ₹${minAcceptable} in final round`;
        }
      }
    }

    if (decision.action === "COUNTER") {
      if (!decision.price) {
        logInternal("Counter-offer missing price — falling back to rule-based");
        return null;
      }
      if (decision.price < state.marginPrice) {
        logInternal(`Counter price ₹${decision.price} floored to margin+buffer`);
        decision.price     = state.marginPrice + state.strategyParams.minProfitMargin;
        decision.reasoning += " (protected margin floor)";
      }
      if (state.lastSellerOffer && decision.price > state.lastSellerOffer) {
        decision.price = Math.max(
          state.lastSellerOffer - 5,
          state.marginPrice + state.strategyParams.minProfitMargin
        );
        decision.reasoning += " (decreased from last offer)";
      }
      decision.price = Math.round(decision.price);
    }

    return decision;
  }

  private ruleBasedDecision(state: SellerNegotiationState): NegotiationDecision {
    const buyerOffer = state.lastBuyerOffer!;

    const profitTargets: Record<number, number> = {
      1: state.marginPrice + 30,
      2: state.marginPrice + 20,
      3: state.marginPrice + 10,
    };
    const targetProfit = profitTargets[state.currentRound] ?? state.marginPrice + 5;

    if (buyerOffer >= targetProfit) {
      return { action: "ACCEPT", reasoning: `Buyer ₹${buyerOffer} meets round ${state.currentRound} profit target` };
    }
    if (state.currentRound === state.maxRounds) {
      if (buyerOffer >= state.marginPrice + state.strategyParams.minProfitMargin) {
        return { action: "ACCEPT", reasoning: "Final round — accepting above-margin offer" };
      } else {
        return { action: "REJECT", reasoning: "Final round — buyer offer below margin" };
      }
    }

    let newOffer: number;
    if (!state.lastSellerOffer) {
      newOffer = Math.max(state.marginPrice * 1.25, buyerOffer * 1.3);
    } else {
      const gap        = state.lastSellerOffer - buyerOffer;
      const concession = gap * (state.currentRound === 2 ? 0.3 : 0.4);
      newOffer = Math.max(
        state.lastSellerOffer - concession,
        state.marginPrice + state.strategyParams.minProfitMargin
      );
    }

    return {
      action:    "COUNTER",
      price:     Math.round(newOffer),
      reasoning: `Strategic counter — ₹${Math.round(newOffer - state.marginPrice)} profit margin`,
    };
  }

  // ================= SEND COUNTER OFFER =================
  private async sendCounterOffer(
    state: SellerNegotiationState,
    price: number,
    reasoning: string,
    logger: NegotiationLogger,
    contextId: string
  ) {
    const previousPrice        = state.lastSellerOffer ?? state.lastBuyerOffer!;
    const priceMovement        = price - previousPrice;
    const priceMovementPercent = (priceMovement / previousPrice) * 100;
    const gap                  = price - state.lastBuyerOffer!;

    logger.log({
      round:                state.currentRound,
      messageType:          "COUNTER_OFFER",
      from:                 "SELLER",
      offeredPrice:         price,
      previousPrice,
      priceMovement,
      priceMovementPercent,
      gap,
      decision:             "COUNTER_OFFER",
      reasoning,
    });

    state.lastSellerOffer = price;
    state.history.push({
      round:        state.currentRound,
      sellerOffer:  price,
      buyerOffer:   state.lastBuyerOffer,
      sellerAction: "COUNTER_OFFER",
      timestamp:    new Date().toISOString(),
      reasoning,
    });

    const counterData: CounterOfferData = {
      type:          "COUNTER_OFFER",
      negotiationId: state.negotiationId,
      round:         state.currentRound,
      timestamp:     new Date().toISOString(),
      pricePerUnit:  price,
      previousPrice,
      from:          "SELLER",
      reasoning,
    };

    await this.sendToBuyer(counterData, contextId);
  }

  // ================= SEND ACCEPTANCE =================
  private async sendAcceptance(
    state: SellerNegotiationState,
    logger: NegotiationLogger,
    contextId: string
  ) {
    const acceptedPrice = state.lastBuyerOffer!;
    const totalAmount   = acceptedPrice * state.quantity;
    const profit        = acceptedPrice - state.marginPrice;

    logger.log({
      round:        state.currentRound,
      messageType:  "ACCEPT",
      from:         "SELLER",
      offeredPrice: acceptedPrice,
      decision:     "ACCEPT",
      reasoning:    `Profit: ₹${profit}/unit (${((profit / state.marginPrice) * 100).toFixed(1)}%)`,
    });

    state.agreedPrice   = acceptedPrice;
    state.profitPerUnit = profit;
    state.totalRevenue  = totalAmount;
    state.status        = "ACCEPTED";

    const acceptanceData: AcceptanceData = {
      type:          "ACCEPT_OFFER",
      negotiationId: state.negotiationId,
      round:         state.currentRound,
      timestamp:     new Date().toISOString(),
      acceptedPrice,
      from:          "SELLER",
      finalTerms: {
        pricePerUnit: acceptedPrice,
        quantity:     state.quantity,
        totalAmount,
        deliveryDate: state.deliveryDate,
      },
    };

    await this.sendToBuyer(acceptanceData, contextId);
  }

  // ================= SEND INVOICE =================
  private async sendInvoice(
    state: SellerNegotiationState,
    poId: string,
    invoiceId: string,
    logger: NegotiationLogger,
    contextId: string
  ) {
    const subtotal = state.agreedPrice! * state.quantity;
    const tax      = Math.round(subtotal * 0.18);
    const total    = subtotal + tax;

    const invoiceData: InvoiceData = {
      type:          "INVOICE",
      invoiceId,
      negotiationId: state.negotiationId,
      poId,
      invoiceDate:   new Date().toISOString(),
      terms: {
        pricePerUnit: state.agreedPrice!,
        quantity:     state.quantity,
        subtotal,
        tax,
        total,
      },
      paymentTerms: `Net ${SELLER_CONFIG.dd.paymentTermsDays} days`,
      deliveryDate: state.deliveryDate,
    };

    logger.printInvoice(invoiceData);
    await this.sendToBuyer(invoiceData, contextId);
  }

  // ================= HELPERS =================
  private async sendToBuyer(data: any, contextId: string): Promise<void> {
    try {
      const buyerClient = await A2AClient.fromCardUrl(
        "http://localhost:9090/.well-known/agent-card.json"
      );

      const message: Message = {
        messageId: uuidv4(),
        kind:      "message",
        role:      "agent",
        contextId,
        parts: [
          { kind: "data", data },
          { kind: "text", text: `Negotiation ${data.type} - Round ${data.round || "N/A"}` },
        ],
      };

      const params: MessageSendParams = { message };
      const stream = buyerClient.sendMessageStream(params);

      await Promise.race([
        (async () => { for await (const _ of stream) {} })(),
        new Promise((resolve) => setTimeout(resolve, 10000)),
      ]);
    } catch (error: any) {
      if (error.code !== "UND_ERR_BODY_TIMEOUT" && error.message !== "terminated") {
        logInternal(`Send-to-buyer error: ${error.message || error}`);
      }
    }
  }

  private respond(bus: ExecutionEventBus, taskId: string, contextId: string, text: string) {
    bus.publish({
      kind:      "status-update",
      taskId,
      contextId,
      status: {
        state:     "completed",
        timestamp: new Date().toISOString(),
        message: {
          kind:      "message",
          role:      "agent",
          messageId: uuidv4(),
          parts:     [{ kind: "text", text }],
          taskId,
          contextId,
        },
      },
      final: true,
    } as TaskStatusUpdateEvent);
  }
}

// ================= SERVER SETUP =================
const cardPath   = path.resolve(__dirname, "../../../agent-cards/jupiterSellerAgent-card.json");
const sellerCard: AgentCard = JSON.parse(fs.readFileSync(cardPath, "utf8"));

async function main() {
  const executor = new SellerAgentExecutor();
  const handler  = new DefaultRequestHandler(sellerCard, new InMemoryTaskStore(), executor);

  const app = express();
  app.use(cors());
  new A2AExpressApp(handler).setupRoutes(app);

  const PORT = process.env.PORT || 8080;
  app.listen(PORT, () => {
    console.log(`\n🏪  Seller Agent  →  http://localhost:${PORT}`);
    console.log(`    Margin Price : ₹${SELLER_CONFIG.marginPrice}/unit  (protected)`);
    console.log(`    Target Price : ₹${TARGET_PRICE}/unit`);
    console.log(`    Target Profit: ${(SELLER_CONFIG.targetProfitPercentage * 100).toFixed(0)}%`);
    console.log(`    Max Rounds   : ${SELLER_CONFIG.maxRounds}`);
    console.log(`    DD Safety    : ${SELLER_CONFIG.dd.safetyFactor * 100}%  |  Payment Terms: Net ${SELLER_CONFIG.dd.paymentTermsDays}\n`);
  });
}

main().catch(console.error);
