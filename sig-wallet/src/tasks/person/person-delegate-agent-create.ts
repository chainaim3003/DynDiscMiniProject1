/**
 * Person/OOR Holder Initiates Agent Delegation
 * 
 * This script creates a delegation request from an agent to an OOR holder.
 * The agent client initiates the delegation, and the OOR holder must approve it.
 * 
 * Pattern: Same as qvi-aid-delegate-create.ts
 * 
 * Usage:
 *   tsx person-delegate-agent-create.ts <env> <agentPasscode> <dataDir> <oorHolderName> <agentName>
 * 
 * Example:
 *   tsx person-delegate-agent-create.ts docker myAgentPass123 /task-data Jupiter_Chief_Sales_Officer jupiterSellerAgent
 * 
 * Creates:
 *   /task-data/jupiterSellerAgent-delegate-info.json
 *   {
 *     "aid": "EAgent...",
 *     "icpOpName": "delegation.EAgent..."
 *   }
 */

import fs from 'fs';
import {createDelegate, getOrCreateClient} from "../../client/identifiers.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const agentPasscode = args[1];
const dataDir = args[2];
const oorHolderName = args[3];  // e.g., 'Jupiter_Chief_Sales_Officer'
const agentName = args[4];      // e.g., 'jupiterSellerAgent'

// Validate arguments
if (!env || !agentPasscode || !dataDir || !oorHolderName || !agentName) {
    console.error('Usage: tsx person-delegate-agent-create.ts <env> <agentPasscode> <dataDir> <oorHolderName> <agentName>');
    process.exit(1);
}

// Read OOR Holder info - use synchronous read
const oorHolderInfoPath = `${dataDir}/${oorHolderName}-info.json`;
if (!fs.existsSync(oorHolderInfoPath)) {
    throw new Error(`OOR Holder info file not found: ${oorHolderInfoPath}`);
}
const oorHolderInfo = JSON.parse(fs.readFileSync(oorHolderInfoPath, 'utf-8'));

console.log(`Creating agent ${agentName} as delegate of ${oorHolderName}`);
console.log(`OOR Holder AID: ${oorHolderInfo.aid}`);
console.log(`OOR Holder OOBI: ${oorHolderInfo.oobi}`);

// Agent client creates delegation request to OOR Holder
const agentClient = await getOrCreateClient(agentPasscode, env);
const clientInfo: any = await createDelegate(
    agentClient,
    agentName,              // delegate name
    oorHolderInfo.aid,      // delegator prefix (OOR Holder AID)
    oorHolderName,          // delegator alias name
    oorHolderInfo.oobi      // delegator OOBI
);

// Save agent delegation info - use synchronous write
const agentDelegateInfoPath = `${dataDir}/${agentName}-delegate-info.json`;
fs.writeFileSync(agentDelegateInfoPath, JSON.stringify(clientInfo, null, 2));

// Verify file was written
if (!fs.existsSync(agentDelegateInfoPath)) {
    throw new Error(`Failed to write ${agentDelegateInfoPath}`);
}

console.log(`Agent delegation info written to ${agentDelegateInfoPath}`);
console.log(`Agent AID: ${clientInfo.aid}`);
console.log(`ICP Operation: ${clientInfo.icpOpName}`);
console.log(`Delegation request sent to ${oorHolderName}`);
