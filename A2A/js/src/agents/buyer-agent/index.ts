// ================= BUYER AGENT WITH HYBRID LLM + RULE-BASED DECISION MAKING =================
import express from "express";
import cors from "cors";
import { v4 as uuidv4 } from "uuid";
import fs from "fs";
import path from "path";
import dotenv from "dotenv";
import { fileURLToPath } from "url";

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
import { A2AClient } from "@a2a-js/sdk/client";

import {
  BuyerNegotiationState,
  NegotiationDecision,
  OfferData,
  CounterOfferData,
  AcceptanceData,
  EscalationNoticeData,
  PurchaseOrderData,
  NegotiationData,
  DDOfferData,
  DDAcceptData,
  DDInvoiceData,
} from "../../shared/negotiation-types.js";

import { LLMNegotiationClient, LLMPromptContext } from "../../shared/llm-client.js";
import { NegotiationLogger, logInternal, suppressSDKNoise } from "../../shared/logger.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, ".env") });

// Suppress @a2a-js/sdk internal stdout noise (ResultManager logs etc.)
suppressSDKNoise();

// ================= BUYER AGENT CONFIGURATION =================
const BUYER_CONFIG = {
  maxBudget: 400,
  targetQuantity: 2000,
  maxRounds: 3,
  initialOfferRange: { min: 250, max: 320 },
  targetPrice: 330,
  strategyParams: {
    aggressiveness: 0.6,
    riskTolerance: 0.7,
  },
};

// ================= PENDING DD OFFER (awaiting buyer CLI input) =================
interface PendingDDOffer {
  offer:      DDOfferData;
  contextId:  string;
}

// ================= BUYER AGENT EXECUTOR =================
class BuyerAgentExecutor implements AgentExecutor {
  private negotiations   = new Map<string, BuyerNegotiationState>();
  private loggers        = new Map<string, NegotiationLogger>();
  private llmClient:       LLMNegotiationClient;
  // Pending DD offers waiting for buyer CLI decision.
  // Key = negotiationId. We also track the last one so bare 'dd accept' works.
  private pendingDDOffers         = new Map<string, PendingDDOffer>();
  private lastPendingDDNegId: string | undefined;

  constructor() {
    this.llmClient = new LLMNegotiationClient();
  }

  async cancelTask(taskId: string): Promise<void> {
    logInternal(`Task cancellation requested: ${taskId}`);
  }

  // ================= MAIN EXECUTION =================
  async execute(ctx: RequestContext, bus: ExecutionEventBus) {
    const taskId    = ctx.task?.id        || uuidv4();
    const contextId = ctx.task?.contextId || uuidv4();

    const textInput = ctx.userMessage.parts
      .filter((p) => p.kind === "text")
      .map((p) => (p as any).text)
      .join(" ")
      .toLowerCase();

    const dataParts = ctx.userMessage.parts.filter((p) => p.kind === "data");

    if (textInput.includes("start negotiation")) {
      const match     = textInput.match(/start negotiation\s+(\d+)/);
      const userPrice = match ? parseInt(match[1], 10) : undefined;
      await this.startNegotiation(contextId, bus, taskId, userPrice);
      return;
    }

    // ── DD commands from buyer CLI ────────────────────────────────────────────
    if (textInput.startsWith("dd ")) {
      await this.handleDDCommand(textInput.trim(), contextId, bus, taskId);
      return;
    }

    if (dataParts.length > 0) {
      const data = (dataParts[0] as any).data as NegotiationData;
      await this.handleSellerMessage(data, contextId, bus, taskId);
      return;
    }

    this.respond(bus, taskId, contextId, "🛒 Buyer Agent Ready. Send 'start negotiation' to begin.");
  }

  // ================= START NEGOTIATION =================
  private async startNegotiation(
    contextId: string,
    bus: ExecutionEventBus,
    taskId: string,
    userPrice?: number
  ) {
    const negotiationId = `NEG-${Date.now()}`;
    const logger        = new NegotiationLogger(negotiationId, "BUYER");
    this.loggers.set(negotiationId, logger);

    logger.printSessionHeader(contextId);

    const initialOffer = userPrice ?? this.generateInitialOffer();
    if (userPrice) {
      logInternal(`Using user-specified price: ₹${initialOffer}`);
    } else {
      logInternal(`Generated random initial price: ₹${initialOffer}`);
    }

    const state: BuyerNegotiationState = {
      negotiationId,
      contextId,
      status:         "INITIATED",
      targetQuantity: BUYER_CONFIG.targetQuantity,
      maxBudget:      BUYER_CONFIG.maxBudget,
      deliveryDate:   this.getDeliveryDate(),
      currentRound:   1,
      maxRounds:      BUYER_CONFIG.maxRounds,
      history:        [],
      lastBuyerOffer: initialOffer,
      strategyParams: {
        ...BUYER_CONFIG.strategyParams,
        initialOfferRange: BUYER_CONFIG.initialOfferRange,
      },
    };

    this.negotiations.set(negotiationId, state);
    logger.printRoundHeader(1, BUYER_CONFIG.maxRounds);

    const offerData: OfferData = {
      type:         "OFFER",
      negotiationId,
      round:        1,
      timestamp:    new Date().toISOString(),
      pricePerUnit: initialOffer,
      quantity:     BUYER_CONFIG.targetQuantity,
      from:         "BUYER",
      deliveryDate: state.deliveryDate,
    };

    logger.log({
      round:        1,
      messageType:  "OFFER",
      from:         "BUYER",
      offeredPrice: initialOffer,
      decision:     "OFFER",
      reasoning:    `Opening at ₹${initialOffer}, leaving negotiation room`,
    });

    state.history.push({
      round:       1,
      buyerOffer:  initialOffer,
      buyerAction: "OFFER",
      timestamp:   new Date().toISOString(),
    });

    await this.sendToSeller(offerData, contextId);

    this.respond(
      bus, taskId, contextId,
      `✓ Negotiation started\nInitial offer: ₹${initialOffer}/unit  |  Qty: ${BUYER_CONFIG.targetQuantity}\nWaiting for seller response...`
    );
  }

  // ================= HANDLE SELLER MESSAGES =================
  private async handleSellerMessage(
    data: NegotiationData,
    contextId: string,
    bus: ExecutionEventBus,
    taskId: string
  ) {
    const negotiationId = data.negotiationId || (data as any).negotiationId;
    const state  = this.negotiations.get(negotiationId);
    const logger = this.loggers.get(negotiationId);

    // ── DD messages (no state guard needed) ──────────────────────────────────
    if (data.type === "DD_OFFER") {
      await this.handleDDOffer(data as DDOfferData, state, logger, bus, taskId, contextId);
      return;
    }
    if (data.type === "DD_INVOICE") {
      await this.handleDDInvoice(data as DDInvoiceData, state, logger, bus, taskId, contextId);
      return;
    }

    if (!state || !logger) {
      logInternal(`Negotiation state not found: ${negotiationId}`);
      return;
    }

    if (data.type === "ACCEPT_OFFER") {
      await this.handleSellerAcceptance(data as AcceptanceData, state, logger, bus, taskId, contextId);
      return;
    }
    if (data.type === "COUNTER_OFFER") {
      await this.handleSellerCounterOffer(data as CounterOfferData, state, logger, bus, taskId, contextId);
      return;
    }
    if (data.type === "REJECT_OFFER") {
      logger.log({
        round:       state.currentRound,
        messageType: "REJECT",
        from:        "SELLER",
        decision:    "REJECT",
        reasoning:   (data as any).reason,
      });
      state.status = "REJECTED";
      logger.printNegotiationSummary("FAILED", {
        roundsUsed: state.currentRound,
        maxRounds:  state.maxRounds,
        quantity:   state.targetQuantity,
      });
      this.respond(bus, taskId, contextId, "✗ Negotiation failed — Seller rejected offer");
    }
  }

  // ================= HANDLE SELLER ACCEPTANCE =================
  private async handleSellerAcceptance(
    data: AcceptanceData,
    state: BuyerNegotiationState,
    logger: NegotiationLogger,
    bus: ExecutionEventBus,
    taskId: string,
    contextId: string
  ) {
    // Bilateral from seller after buyer already accepted a counter-offer:
    // PO was already sent from handleSellerCounterOffer. Just silently return.
    if (state.status === "COMPLETED" || state.status === "ACCEPTED") {
      logInternal(`Bilateral acceptance received — deal already closed at ₹${state.agreedPrice}`);
      return;
    }

    logger.log({
      round:        state.currentRound,
      messageType:  "ACCEPT",
      from:         "SELLER",
      offeredPrice: data.acceptedPrice,
      decision:     "ACCEPT",
      reasoning:    "Seller accepted our offer",
    });

    state.agreedPrice = data.acceptedPrice;
    state.totalCost   = data.acceptedPrice * state.targetQuantity;
    state.status      = "ACCEPTED";

    const acceptanceData: AcceptanceData = {
      type:          "ACCEPT_OFFER",
      negotiationId: state.negotiationId,
      round:         state.currentRound,
      timestamp:     new Date().toISOString(),
      acceptedPrice: data.acceptedPrice,
      from:          "BUYER",
      finalTerms: {
        pricePerUnit: data.acceptedPrice,
        quantity:     state.targetQuantity,
        totalAmount:  state.totalCost,
        deliveryDate: state.deliveryDate,
      },
    };

    logger.log({
      round:        state.currentRound,
      messageType:  "ACCEPT",
      from:         "BUYER",
      offeredPrice: data.acceptedPrice,
      decision:     "ACCEPT",
      reasoning:    "bilateral acceptance rule",
    });

    await this.sendToSeller(acceptanceData, contextId);
    await this.sendPurchaseOrder(state, logger, contextId);

    const buyerStart  = state.history[0]?.buyerOffer;
    const sellerStart = state.history[0]?.sellerOffer;

    logger.printNegotiationSummary("COMPLETED", {
      roundsUsed:       state.currentRound,
      maxRounds:        state.maxRounds,
      finalPrice:       data.acceptedPrice,
      buyerStartPrice:  buyerStart,
      sellerStartPrice: sellerStart,
      totalCost:        state.totalCost,
      quantity:         state.targetQuantity,
    });

    state.status = "COMPLETED";

    this.respond(
      bus, taskId, contextId,
      `✓✓ Deal Closed!\n\nFinal Price : ₹${data.acceptedPrice}/unit\nTotal       : ₹${state.totalCost?.toLocaleString()}\nPurchase Order sent to seller.`
    );
  }

  // ================= HANDLE SELLER COUNTER OFFER =================
  private async handleSellerCounterOffer(
    data: CounterOfferData,
    state: BuyerNegotiationState,
    logger: NegotiationLogger,
    bus: ExecutionEventBus,
    taskId: string,
    contextId: string
  ) {
    state.lastSellerOffer = data.pricePerUnit;

    const priceMovement        = data.pricePerUnit - data.previousPrice;
    const priceMovementPercent = (priceMovement / data.previousPrice) * 100;

    logger.log({
      round:                state.currentRound,
      messageType:          "COUNTER_OFFER",
      from:                 "SELLER",
      offeredPrice:         data.pricePerUnit,
      previousPrice:        data.previousPrice,
      priceMovement,
      priceMovementPercent,
      decision:             "COUNTER_OFFER",
      reasoning:            data.reasoning,
    });

    const currentHistory = state.history.find((h) => h.round === state.currentRound);
    if (currentHistory) {
      currentHistory.sellerOffer  = data.pricePerUnit;
      currentHistory.sellerAction = "COUNTER_OFFER";
    }

    state.currentRound += 1;

    // ── MAX ROUNDS EXCEEDED → ESCALATE ────────────────────────────────────────
    if (state.currentRound > state.maxRounds) {
      await this.escalateToHuman(state, logger, bus, taskId, contextId);
      return;
    }

    logger.printRoundHeader(state.currentRound, state.maxRounds);

    const decision = await this.makeNegotiationDecision(state);

    if (decision.action === "ACCEPT") {
      await this.sendAcceptance(state, logger, contextId);

      // ── Send PO immediately — don't wait for bilateral ────────────────────
      // When buyer accepts seller's counter, state.status is now "ACCEPTED".
      // The seller's bilateral ACCEPT_OFFER will arrive and hit the duplicate
      // guard in handleSellerAcceptance, so we must complete the flow here.
      await this.sendPurchaseOrder(state, logger, contextId);

      const buyerStart  = state.history[0]?.buyerOffer;
      const sellerStart = state.history[0]?.sellerOffer;

      logger.printNegotiationSummary("COMPLETED", {
        roundsUsed:       state.currentRound,
        maxRounds:        state.maxRounds,
        finalPrice:       data.pricePerUnit,
        buyerStartPrice:  buyerStart,
        sellerStartPrice: sellerStart,
        totalCost:        data.pricePerUnit * state.targetQuantity,
        quantity:         state.targetQuantity,
      });

      state.status = "COMPLETED";

      this.respond(
        bus, taskId, contextId,
        `✓✓ Deal Closed!\n\nFinal Price : ₹${data.pricePerUnit}/unit\nTotal       : ₹${(data.pricePerUnit * state.targetQuantity).toLocaleString()}\nPurchase Order sent to seller.`
      );
    } else if (decision.action === "COUNTER") {
      await this.sendCounterOffer(state, decision.price!, decision.reasoning, logger, contextId);
      this.respond(
        bus, taskId, contextId,
        `↑ Counter-offer sent: ₹${decision.price}/unit  (Round ${state.currentRound}/${state.maxRounds})\nWaiting for seller response...`
      );
    } else {
      state.status = "REJECTED";
      this.respond(bus, taskId, contextId, "✗ Offer rejected — exceeds budget");
    }
  }

  // ================= HANDLE DD_OFFER (store + prompt CLI) =================
  private async handleDDOffer(
    data: DDOfferData,
    state: BuyerNegotiationState | undefined,
    logger: NegotiationLogger | undefined,
    bus: ExecutionEventBus,
    taskId: string,
    contextId: string
  ) {
    // Store pending offer so CLI commands can find it.
    this.pendingDDOffers.set(data.negotiationId, { offer: data, contextId });
    this.lastPendingDDNegId = data.negotiationId;

    const maxPct  = (data.maxDiscountRate * 100).toFixed(3);
    const propPct = (data.discountAtProposedDate.appliedRate * 100).toFixed(3);

    // Print to buyer terminal (server-side log)
    console.log("");
    console.log(`  \x1b[36m\x1b[1m  💰  DD OFFER RECEIVED — AWAITING YOUR DECISION\x1b[0m`);
    console.log(`  \x1b[2m  ─────────────────────────────────────────────────────────\x1b[0m`);
    console.log(`  \x1b[2m  Invoice      : ${data.invoiceId}\x1b[0m`);
    console.log(`  \x1b[2m  Invoice date : ${data.invoiceDate}\x1b[0m`);
    console.log(`  \x1b[2m  Due date     : ${data.dueDate}  (full ₹${data.originalTotal.toLocaleString()} if paid on this date)\x1b[0m`);
    console.log(`  \x1b[1m  Max discount : ${maxPct}% (linear — more days early = higher discount)\x1b[0m`);
    console.log(`  \x1b[32m\x1b[1m  Seller suggests: pay by ${data.proposedSettlementDate}  (${data.discountAtProposedDate.daysEarly}/${data.discountAtProposedDate.totalDays} days early)\x1b[0m`);
    console.log(`  \x1b[32m\x1b[1m    → ₹${data.discountAtProposedDate.discountedAmount.toLocaleString()}  (save ₹${data.discountAtProposedDate.savingAmount.toLocaleString()} @ ${propPct}%)\x1b[0m`);
    console.log(`  \x1b[2m  ─────────────────────────────────────────────────────────\x1b[0m`);

    // Respond to CLI with the decision menu
    this.respond(
      bus, taskId, contextId,
      [
        `💰 Dynamic Discount Offer received from seller`,
        ``,
        `  Invoice      : ${data.invoiceId}`,
        `  Invoice date : ${data.invoiceDate}`,
        `  Due date     : ${data.dueDate}`,
        `  Full amount  : ₹${data.originalTotal.toLocaleString()}`,
        `  Max DD rate  : ${maxPct}% (LINEAR — more days early = higher discount)`,
        ``,
        `  Seller's proposal:`,
        `    Pay by ${data.proposedSettlementDate}  (${data.discountAtProposedDate.daysEarly} days early)`,
        `    → ₹${data.discountAtProposedDate.discountedAmount.toLocaleString()}  (save ₹${data.discountAtProposedDate.savingAmount.toLocaleString()} @ ${propPct}%)`,
        ``,
        `  You can choose any date between ${data.invoiceDate} and ${data.dueDate}.`,
        `  Earlier = more discount.`,
        ``,
        `  Commands:`,
        `    dd accept                → accept seller's date (${data.proposedSettlementDate})`,
        `    dd accept YYYY-MM-DD     → choose your own early payment date`,
        `    dd reject                → decline, pay full amount on due date`,
      ].join("\n")
    );
  }

  // ================= HANDLE DD CLI COMMANDS =================
  private async handleDDCommand(
    textInput: string,
    contextId: string,
    bus: ExecutionEventBus,
    taskId: string
  ) {
    // Resolve the pending offer
    if (!this.lastPendingDDNegId) {
      this.respond(bus, taskId, contextId, "⚠ No pending DD offer. Wait for the seller to send one after the invoice.");
      return;
    }
    const pending = this.pendingDDOffers.get(this.lastPendingDDNegId);
    if (!pending) {
      this.respond(bus, taskId, contextId, "⚠ Pending DD offer not found. It may have already been processed.");
      return;
    }

    const { offer, contextId: ddContextId } = pending;
    const state = this.negotiations.get(offer.negotiationId);

    // ── dd reject ────────────────────────────────────────────────────────────
    if (textInput === "dd reject") {
      this.pendingDDOffers.delete(offer.negotiationId);
      this.lastPendingDDNegId = undefined;
      if (state) state.status = "COMPLETED";

      console.log("");
      console.log(`  \x1b[33m\x1b[1m  ✗  DD REJECTED — will pay full amount on due date (${offer.dueDate})\x1b[0m`);
      console.log("");

      this.respond(
        bus, taskId, contextId,
        `✗ DD offer declined.\nFull payment of ₹${offer.originalTotal.toLocaleString()} due on ${offer.dueDate}.\nWorkflow complete.`
      );
      return;
    }

    // ── dd accept  OR  dd accept YYYY-MM-DD ──────────────────────────────────
    if (textInput.startsWith("dd accept")) {
      // Parse optional date
      const dateMatch = textInput.match(/(\d{4}-\d{2}-\d{2})/);
      let chosenDate: string;

      if (dateMatch) {
        chosenDate = dateMatch[1];

        // Validate: must be between invoiceDate and dueDate (inclusive)
        const chosen  = new Date(chosenDate).getTime();
        const invoice = new Date(offer.invoiceDate).getTime();
        const due     = new Date(offer.dueDate).getTime();

        if (chosen < invoice || chosen > due) {
          this.respond(
            bus, taskId, contextId,
            `⚠ Invalid date ${chosenDate}.\nMust be between ${offer.invoiceDate} and ${offer.dueDate}.\nTry again: dd accept YYYY-MM-DD`
          );
          return;
        }
      } else {
        // No date given — use seller's proposed date
        chosenDate = offer.proposedSettlementDate;
      }

      // Compute preview of discount at chosen date
      const { computeLinearDiscount } = await import("../../shared/dd-calculator.js");
      const preview = computeLinearDiscount(
        offer.originalTotal,
        offer.maxDiscountRate,
        offer.invoiceDate,
        offer.dueDate,
        chosenDate
      );

      const chosenPct = (preview.appliedRate * 100).toFixed(3);

      // Remove from pending
      this.pendingDDOffers.delete(offer.negotiationId);
      this.lastPendingDDNegId = undefined;
      if (state) state.status = "DD_COMPLETED";

      // Send DD_ACCEPT to seller
      const ddAccept: DDAcceptData = {
        type:                 "DD_ACCEPT",
        invoiceId:            offer.invoiceId,
        negotiationId:        offer.negotiationId,
        chosenSettlementDate: chosenDate,
        from:                 "BUYER",
      };

      console.log("");
      console.log(`  \x1b[32m\x1b[1m  ✓  DD ACCEPTED — settlement date: ${chosenDate}\x1b[0m`);
      console.log(`  \x1b[2m  Payable: ₹${preview.discountedAmount.toLocaleString()}  (save ₹${preview.savingAmount.toLocaleString()} @ ${chosenPct}%)\x1b[0m`);
      console.log("");

      logInternal(`DD_ACCEPT sent — invoiceId: ${offer.invoiceId}  settlementDate: ${chosenDate}`);
      await this.sendToSeller(ddAccept, ddContextId);

      this.respond(
        bus, taskId, contextId,
        [
          `✓ DD accepted — settlement date: ${chosenDate}`,
          ``,
          `  Original amount  : ₹${offer.originalTotal.toLocaleString()}`,
          `  Discount applied : ${chosenPct}%  (${preview.daysEarly} of ${preview.totalDays} days early)`,
          `  Payable          : ₹${preview.discountedAmount.toLocaleString()}`,
          `  Saving           : ₹${preview.savingAmount.toLocaleString()}`,
          ``,
          `Awaiting discounted invoice from seller...`,
        ].join("\n")
      );
      return;
    }

    // Unknown dd command
    this.respond(
      bus, taskId, contextId,
      [
        `⚠ Unknown command. Use:`,
        `  dd accept               → accept seller's proposed date`,
        `  dd accept YYYY-MM-DD    → choose your own date`,
        `  dd reject               → decline discount`,
      ].join("\n")
    );
  }

  // ================= HANDLE DD_INVOICE =================
  private async handleDDInvoice(
    data: DDInvoiceData,
    state: BuyerNegotiationState | undefined,
    logger: NegotiationLogger | undefined,
    bus: ExecutionEventBus,
    taskId: string,
    contextId: string
  ) {
    const pct         = (data.appliedRate * 100).toFixed(3);
    const actusStatus = data.actusSimulationStatus === "SUCCESS" ? "✓" : "⚠";

    console.log("");
    console.log(`  \x1b[35m\x1b[1m  📄  DD INVOICE RECEIVED — FINAL\x1b[0m`);
    console.log(`  \x1b[2m  ─────────────────────────────────────────────────────────\x1b[0m`);
    console.log(`  \x1b[2m  Invoice ID   : ${data.invoiceId}\x1b[0m`);
    console.log(`  \x1b[2m  Original     : ₹${data.originalTotal.toLocaleString()}\x1b[0m`);
    console.log(`  \x1b[1m  Applied Rate →  ${pct}%\x1b[0m`);
    console.log(`  \x1b[32m\x1b[1m  PAYABLE      →  ₹${data.discountedTotal.toLocaleString()}  (saved ₹${data.savingAmount.toLocaleString()})\x1b[0m`);
    console.log(`  \x1b[1m  Settle by   →  ${data.settlementDate}\x1b[0m`);
    console.log(`  \x1b[2m  ACTUS ID     : ${data.actusContractId}\x1b[0m`);
    console.log(`  \x1b[2m  ACTUS Status : ${actusStatus} ${data.actusSimulationStatus}${data.actusError ? " — " + data.actusError : ""}\x1b[0m`);
    console.log(`  \x1b[2m  ─────────────────────────────────────────────────────────\x1b[0m`);
    console.log("");
    console.log(`  \x1b[32m\x1b[1m  ✅  END-TO-END WORKFLOW COMPLETE\x1b[0m`);
    console.log(`  \x1b[2m  Negotiation → Invoice → Dynamic Discounting → ACTUS\x1b[0m`);
    console.log("");

    if (state) state.status = "DD_COMPLETED";

    this.respond(
      bus, taskId, contextId,
      `✅ DD Invoice received!\n\nOriginal   : ₹${data.originalTotal.toLocaleString()}\nDiscounted : ₹${data.discountedTotal.toLocaleString()}  (${pct}% off)\nSaving     : ₹${data.savingAmount.toLocaleString()}\nSettle by  : ${data.settlementDate}\nACTUS      : ${actusStatus} ${data.actusSimulationStatus}\n\n🎉 End-to-end workflow complete!\nNegotiation → Invoice → Dynamic Discounting → ACTUS`
    );
  }

  // ================= ESCALATE TO HUMAN =================
  private async escalateToHuman(
    state: BuyerNegotiationState,
    logger: NegotiationLogger,
    bus: ExecutionEventBus,
    taskId: string,
    contextId: string
  ) {
    state.status = "ESCALATED";

    const buyerFinalOffer  = state.lastBuyerOffer!;
    const sellerFinalOffer = state.lastSellerOffer!;
    const gap              = sellerFinalOffer - buyerFinalOffer;

    // Save the report and get back its path
    const reportPath = logger.saveEscalationReport({
      buyerFinalOffer,
      sellerFinalOffer,
      gap,
      rounds:       state.maxRounds,
      maxRounds:    state.maxRounds,
      quantity:     state.targetQuantity,
      deliveryDate: state.deliveryDate,
      logs:         logger.getLogs(),
    });

    // Print escalation notice on the buyer terminal
    logger.printEscalationNotice(buyerFinalOffer, sellerFinalOffer, gap, reportPath);

    // Notify seller agent so it exits cleanly
    const escalationNotice: EscalationNoticeData = {
      type:             "ESCALATION_NOTICE",
      negotiationId:    state.negotiationId,
      round:            state.maxRounds,
      timestamp:        new Date().toISOString(),
      from:             "BUYER",
      buyerFinalOffer,
      sellerFinalOffer,
      gap,
      reportPath,
    };

    await this.sendToSeller(escalationNotice, contextId);

    this.respond(
      bus, taskId, contextId,
      `⚠ Negotiation escalated to human review.\nGap of ₹${gap} remains after ${state.maxRounds} round(s).\nReport saved → ${reportPath}`
    );
  }

  // ================= HYBRID DECISION MAKING =================
  private async makeNegotiationDecision(state: BuyerNegotiationState): Promise<NegotiationDecision> {
    const llmDecision       = await this.getLLMDecision(state);
    const validatedDecision = this.applyBuyerConstraints(llmDecision, state);

    if (!validatedDecision) {
      logInternal("LLM decision invalid — using rule-based fallback");
      return this.ruleBasedDecision(state);
    }
    return validatedDecision;
  }

  private async getLLMDecision(state: BuyerNegotiationState): Promise<NegotiationDecision> {
    const context: LLMPromptContext = {
      role:           "BUYER",
      round:          state.currentRound,
      maxRounds:      state.maxRounds,
      lastOwnOffer:   state.lastBuyerOffer,
      lastTheirOffer: state.lastSellerOffer,
      history:        state.history,
      constraints:    { maxBudget: state.maxBudget, quantity: state.targetQuantity },
      targetPrice:    BUYER_CONFIG.targetPrice,
    };
    const llmResponse = await this.llmClient.getNegotiationDecision(context);
    return { action: llmResponse.action, price: llmResponse.price, reasoning: llmResponse.reasoning };
  }

  private applyBuyerConstraints(
    decision: NegotiationDecision,
    state: BuyerNegotiationState
  ): NegotiationDecision | null {
    if (decision.action === "ACCEPT") {
      if (state.lastSellerOffer && state.lastSellerOffer > state.maxBudget) {
        logInternal(`Cannot accept ₹${state.lastSellerOffer} — exceeds budget ₹${state.maxBudget}`);
        if (state.currentRound < state.maxRounds) {
          decision.action    = "COUNTER";
          decision.price     = Math.min(state.maxBudget, state.lastSellerOffer! - 10);
          decision.reasoning = "Seller price exceeds budget, making counter-offer";
        } else {
          decision.action    = "REJECT";
          decision.reasoning = "Price exceeds budget in final round";
        }
      }
    }

    if (decision.action === "COUNTER") {
      if (!decision.price) {
        logInternal("Counter-offer missing price — falling back to rule-based");
        return null;
      }
      if (decision.price > state.maxBudget) {
        logInternal(`Counter price ₹${decision.price} capped to budget ₹${state.maxBudget}`);
        decision.price     = state.maxBudget;
        decision.reasoning += " (capped at budget)";
      }
      if (state.lastBuyerOffer && decision.price < state.lastBuyerOffer) {
        decision.price     = state.lastBuyerOffer + 5;
        decision.reasoning += " (increased from last offer)";
      }
      decision.price = Math.round(decision.price);
    }

    return decision;
  }

  private ruleBasedDecision(state: BuyerNegotiationState): NegotiationDecision {
    const sellerOffer    = state.lastSellerOffer!;
    const lastBuyerOffer = state.lastBuyerOffer!;

    const acceptanceThresholds: Record<number, number> = { 1: 340, 2: 360, 3: 380 };
    const threshold = acceptanceThresholds[state.currentRound] ?? 380;

    if (sellerOffer <= threshold && sellerOffer <= state.maxBudget) {
      return { action: "ACCEPT", reasoning: `Seller ₹${sellerOffer} meets round ${state.currentRound} threshold` };
    }
    if (state.currentRound === state.maxRounds && sellerOffer <= state.maxBudget + 10) {
      return { action: "ACCEPT", reasoning: "Final round — accepting near-budget offer" };
    }

    const gap            = sellerOffer - lastBuyerOffer;
    const concessionRate = state.currentRound === 3 ? 0.6 : 0.4;
    const newOffer       = Math.min(Math.round(lastBuyerOffer + gap * concessionRate), state.maxBudget);

    return {
      action:    "COUNTER",
      price:     newOffer,
      reasoning: `Closing ${(concessionRate * 100).toFixed(0)}% of gap`,
    };
  }

  // ================= SEND COUNTER OFFER =================
  private async sendCounterOffer(
    state: BuyerNegotiationState,
    price: number,
    reasoning: string,
    logger: NegotiationLogger,
    contextId: string
  ) {
    const priceMovement        = price - state.lastBuyerOffer!;
    const priceMovementPercent = (priceMovement / state.lastBuyerOffer!) * 100;
    const gap                  = state.lastSellerOffer! - price;
    const gapClosed            = gap > 0 ? (priceMovement / (state.lastSellerOffer! - state.lastBuyerOffer!)) * 100 : 0;

    logger.log({
      round:                state.currentRound,
      messageType:          "COUNTER_OFFER",
      from:                 "BUYER",
      offeredPrice:         price,
      previousPrice:        state.lastBuyerOffer,
      priceMovement,
      priceMovementPercent,
      gap,
      gapClosed,
      decision:             "COUNTER_OFFER",
      reasoning,
    });

    state.lastBuyerOffer = price;
    state.history.push({
      round:       state.currentRound,
      buyerOffer:  price,
      buyerAction: "COUNTER_OFFER",
      timestamp:   new Date().toISOString(),
      reasoning,
    });

    const counterData: CounterOfferData = {
      type:          "COUNTER_OFFER",
      negotiationId: state.negotiationId,
      round:         state.currentRound,
      timestamp:     new Date().toISOString(),
      pricePerUnit:  price,
      previousPrice: state.lastBuyerOffer!,
      from:          "BUYER",
      reasoning,
    };

    await this.sendToSeller(counterData, contextId);
  }

  // ================= SEND ACCEPTANCE =================
  private async sendAcceptance(
    state: BuyerNegotiationState,
    logger: NegotiationLogger,
    contextId: string
  ) {
    const acceptedPrice = state.lastSellerOffer!;
    const totalAmount   = acceptedPrice * state.targetQuantity;

    logger.log({
      round:        state.currentRound,
      messageType:  "ACCEPT",
      from:         "BUYER",
      offeredPrice: acceptedPrice,
      decision:     "ACCEPT",
      reasoning:    "Accepting seller's offer based on strategic analysis",
    });

    state.agreedPrice = acceptedPrice;
    state.totalCost   = totalAmount;
    state.status      = "ACCEPTED";

    const acceptanceData: AcceptanceData = {
      type:          "ACCEPT_OFFER",
      negotiationId: state.negotiationId,
      round:         state.currentRound,
      timestamp:     new Date().toISOString(),
      acceptedPrice,
      from:          "BUYER",
      finalTerms: {
        pricePerUnit: acceptedPrice,
        quantity:     state.targetQuantity,
        totalAmount,
        deliveryDate: state.deliveryDate,
      },
    };

    await this.sendToSeller(acceptanceData, contextId);
  }

  // ================= SEND PURCHASE ORDER =================
  private async sendPurchaseOrder(
    state: BuyerNegotiationState,
    logger: NegotiationLogger,
    contextId: string
  ) {
    const poData: PurchaseOrderData = {
      type:          "PURCHASE_ORDER",
      poId:          `PO-${Date.now()}`,
      negotiationId: state.negotiationId,
      orderDate:     new Date().toISOString(),
      terms: {
        pricePerUnit: state.agreedPrice!,
        quantity:     state.targetQuantity,
        total:        state.totalCost!,
      },
      deliveryDate: state.deliveryDate,
    };

    logger.printPurchaseOrder(poData);
    await this.sendToSeller(poData, contextId);
  }

  // ================= HELPERS =================
  private generateInitialOffer(): number {
    const { min, max } = BUYER_CONFIG.initialOfferRange;
    return Math.round(Math.random() * (max - min) + min);
  }

  private getDeliveryDate(): string {
    const date = new Date();
    date.setDate(date.getDate() + 60);
    return date.toISOString().split("T")[0];
  }

  private async sendToSeller(data: any, contextId: string): Promise<void> {
    try {
      const sellerClient = await A2AClient.fromCardUrl(
        "http://localhost:8080/.well-known/agent-card.json"
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
      const stream = sellerClient.sendMessageStream(params);

      await Promise.race([
        (async () => { for await (const _ of stream) {} })(),
        new Promise((resolve) => setTimeout(resolve, 10000)),
      ]);
    } catch (error: any) {
      if (error.code !== "UND_ERR_BODY_TIMEOUT" && error.message !== "terminated") {
        logInternal(`Send-to-seller error: ${error.message || error}`);
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
const buyerCard: AgentCard = JSON.parse(
  fs.readFileSync(
    path.resolve(__dirname, "../../../agent-cards/tommyBuyerAgent-card.json"),
    "utf8"
  )
);

const app = express();
app.use(cors());

const executor = new BuyerAgentExecutor();
const handler  = new DefaultRequestHandler(buyerCard, new InMemoryTaskStore(), executor);
new A2AExpressApp(handler).setupRoutes(app);

const PORT = process.env.PORT || 9090;
app.listen(PORT, () => {
  console.log(`\n🛒  Buyer Agent  →  http://localhost:${PORT}`);
  console.log(`    Max Budget  : ₹${BUYER_CONFIG.maxBudget}/unit`);
  console.log(`    Target Price: ₹${BUYER_CONFIG.targetPrice}/unit`);
  console.log(`    Quantity    : ${BUYER_CONFIG.targetQuantity} units`);
  console.log(`    Max Rounds  : ${BUYER_CONFIG.maxRounds}`);
  console.log(`    DD Decision : manual (buyer types dd accept / dd accept YYYY-MM-DD / dd reject)\n`);
});
