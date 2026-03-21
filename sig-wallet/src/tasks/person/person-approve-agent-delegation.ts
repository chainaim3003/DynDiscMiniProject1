/**
 * Person/OOR Holder Approves Agent Delegation
 * 
 * This script approves a delegation request from an agent.
 * The OOR holder anchors the delegation in their KEL with a seal.
 * 
 * Pattern: Same as geda-delegate-approve.ts
 * 
 * Usage:
 *   tsx person-approve-agent-delegation.ts <env> <oorHolderPasscode> <oorHolderName> <agentDelegateInfoPath>
 * 
 * Example:
 *   tsx person-approve-agent-delegation.ts docker myOORPass123 Jupiter_Chief_Sales_Officer /task-data/jupiterSellerAgent-delegate-info.json
 * 
 * Effect:
 *   - Anchors delegation seal in OOR Holder's KEL
 *   - Seal contains agent AID
 */

import {approveDelegation, getOrCreateClient} from "../../client/identifiers.js";
import fs from "fs";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const oorHolderPasscode = args[1];
const oorHolderName = args[2];       // e.g., 'Jupiter_Chief_Sales_Officer'
const agentDelegateInfoPath = args[3];

// Validate arguments
if (!env || !oorHolderPasscode || !oorHolderName || !agentDelegateInfoPath) {
    console.error('Usage: tsx person-approve-agent-delegation.ts <env> <oorHolderPasscode> <oorHolderName> <agentDelegateInfoPath>');
    process.exit(1);
}

console.log(`Approving delegation from ${oorHolderName} to agent`);

// Read agent delegation info - use synchronous read
if (!fs.existsSync(agentDelegateInfoPath)) {
    throw new Error(`Agent delegate info file not found: ${agentDelegateInfoPath}`);
}
const agentInfo = JSON.parse(fs.readFileSync(agentDelegateInfoPath, 'utf-8'));

console.log(`Agent AID: ${agentInfo.aid}`);

// OOR Holder approves delegation
const oorHolderClient = await getOrCreateClient(oorHolderPasscode, env);
const approved = await approveDelegation(oorHolderClient, oorHolderName, agentInfo.aid);

console.log(`OOR Holder ${oorHolderName} approved delegation of agent ${agentInfo.aid}: ${approved}`);
