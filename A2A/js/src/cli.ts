#!/usr/bin/env node
// ================= NEGOTIATION CLIENT CLI =================

import readline from "node:readline";
import crypto   from "node:crypto";

import {
  MessageSendParams,
  TaskStatusUpdateEvent,
  TaskArtifactUpdateEvent,
  Message,
  Task,
  AgentCard,
  Part,
  FilePart,
  DataPart,
} from "@a2a-js/sdk";
import { A2AClient } from "@a2a-js/sdk/client";

// ── ANSI helpers ──────────────────────────────────────────────────────────────
const C = {
  reset:   "\x1b[0m",
  bold:    "\x1b[1m",
  dim:     "\x1b[2m",
  red:     "\x1b[31m",
  green:   "\x1b[32m",
  yellow:  "\x1b[33m",
  blue:    "\x1b[34m",
  magenta: "\x1b[35m",
  cyan:    "\x1b[36m",
};
const dim    = (s: string) => `${C.dim}${s}${C.reset}`;
const bold   = (s: string) => `${C.bold}${s}${C.reset}`;
const green  = (s: string) => `${C.green}${s}${C.reset}`;
const yellow = (s: string) => `${C.yellow}${s}${C.reset}`;
const cyan   = (s: string) => `${C.cyan}${s}${C.reset}`;
const red    = (s: string) => `${C.red}${s}${C.reset}`;

// ── State ─────────────────────────────────────────────────────────────────────
let currentTaskId:    string | undefined;
let currentContextId: string | undefined;

const serverUrl = process.argv[2] || "http://localhost:41241";
const client    = new A2AClient(serverUrl);
let   agentName = "Agent";

// ── Readline ──────────────────────────────────────────────────────────────────
const rl = readline.createInterface({
  input:  process.stdin,
  output: process.stdout,
  prompt: cyan("You: "),
});

// ── Print only the meaningful text from an agent response ─────────────────────
function printAgentResponse(text: string) {
  if (!text.trim()) return;
  const ts = new Date().toLocaleTimeString();
  console.log(`\n${bold(agentName)} ${dim(`[${ts}]`)}`);
  // Each line of the text message, indented
  for (const line of text.split("\n")) {
    console.log(`  ${line}`);
  }
}

// ── Extract all text parts from an A2A message ────────────────────────────────
function extractText(parts: Part[]): string {
  return parts
    .filter((p) => p.kind === "text")
    .map((p) => (p as any).text as string)
    .join("\n")
    .trim();
}

// ── Process one stream event — show only what matters ────────────────────────
function handleEvent(event: any) {
  if (event.kind === "status-update") {
    const e    = event as TaskStatusUpdateEvent;
    const text = e.status.message ? extractText(e.status.message.parts) : "";

    // Update IDs silently
    if (!currentTaskId)    currentTaskId    = e.taskId;
    if (!currentContextId) currentContextId = e.contextId;

    // Only print if there is actual content to show
    if (text) printAgentResponse(text);

  } else if (event.kind === "artifact-update") {
    const e    = event as TaskArtifactUpdateEvent;
    const text = extractText(e.artifact.parts);
    if (text) printAgentResponse(text);

  } else if (event.kind === "message") {
    const e    = event as Message;
    const text = extractText(e.parts);
    if (e.taskId    && e.taskId    !== currentTaskId)    currentTaskId    = e.taskId;
    if (e.contextId && e.contextId !== currentContextId) currentContextId = e.contextId;
    if (text) printAgentResponse(text);

  } else if (event.kind === "task") {
    const e = event as Task;
    if (e.id        !== currentTaskId)    currentTaskId    = e.id;
    if (e.contextId !== currentContextId) currentContextId = e.contextId;
    const text = e.status.message ? extractText(e.status.message.parts) : "";
    if (text) printAgentResponse(text);
  }
  // All other event kinds (A2A protocol internals) are silently ignored
}

// ── Agent card ────────────────────────────────────────────────────────────────
async function fetchAgentCard() {
  try {
    const card: AgentCard = await client.getAgentCard();
    agentName = card.name || "Agent";
    console.log(green(`✓ Connected to: ${bold(agentName)}`));
    if (card.description) console.log(dim(`  ${card.description}`));
  } catch {
    console.log(yellow(`⚠  Could not reach agent at ${serverUrl} — is it running?`));
    throw new Error("Agent unreachable");
  }
}

// ── Main loop ─────────────────────────────────────────────────────────────────
async function main() {
  console.log(bold("\n═══════════════════════════════════════"));
  console.log(bold("   NEGOTIATION CLIENT"));
  console.log(bold("═══════════════════════════════════════"));
  console.log(dim(`  Agent URL : ${serverUrl}`));
  console.log("");

  await fetchAgentCard();

  console.log(dim("\n  Commands:"));
  console.log(dim("    start negotiation <price>  — begin a new negotiation (price optional)"));
  console.log(dim("    dd accept                  — accept seller's proposed early payment date"));
  console.log(dim("    dd accept YYYY-MM-DD        — choose your own early payment date"));
  console.log(dim("    dd reject                  — decline discount, pay full on due date"));
  console.log(dim("    /new                       — reset session"));
  console.log(dim("    /exit                      — quit"));
  console.log(dim(""));
  console.log(dim("  DD Workflow:"));
  console.log(dim("    1. Run 'start negotiation' and watch the negotiation complete in Terminals 1 & 2."));
  console.log(dim("    2. When you see the DD Offer in Terminal 2 (Buyer), come back here."));
  console.log(dim("    3. Type 'dd accept' or 'dd accept YYYY-MM-DD' to trigger the discounted invoice.\n"));

  rl.setPrompt(cyan("You: "));
  rl.prompt();

  rl.on("line", async (line) => {
    const input = line.trim();
    if (!input) { rl.prompt(); return; }

    if (input.toLowerCase() === "/new") {
      currentTaskId    = undefined;
      currentContextId = undefined;
      console.log(dim("  Session reset.\n"));
      rl.prompt();
      return;
    }

    if (input.toLowerCase() === "/exit") {
      rl.close();
      return;
    }

    // ── Build and send message ────────────────────────────────────────────────

    // DD commands must start a NEW task — the negotiation task is already
    // completed and removed from InMemoryTaskStore. Clearing taskId makes
    // the SDK create a fresh task. We keep contextId to stay in the same
    // conversation so the buyer agent can find the pending DD offer.
    if (input.toLowerCase().startsWith("dd ")) {
      currentTaskId = undefined;
    }

    const messagePayload: Message = {
      messageId: crypto.randomUUID(),
      kind:      "message",
      role:      "user",
      parts:     [{ kind: "text", text: input }],
    };
    if (currentTaskId)    messagePayload.taskId    = currentTaskId;
    if (currentContextId) messagePayload.contextId = currentContextId;

    const params: MessageSendParams = { message: messagePayload };

    try {
      const stream = client.sendMessageStream(params);
      for await (const event of stream) {
        handleEvent(event);
      }
    } catch (error: any) {
      console.error(red(`\n  Error: ${error.message || error}`));
    } finally {
      rl.prompt();
    }
  });

  rl.on("close", () => {
    console.log(yellow("\nGoodbye!\n"));
    process.exit(0);
  });
}

main().catch((err) => {
  console.error(red("Fatal error:"), err);
  process.exit(1);
});
