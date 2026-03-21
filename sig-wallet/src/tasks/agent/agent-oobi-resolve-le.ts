/**
 * Agent Resolves LE OOBI
 * 
 * This script resolves the LE OOBI for an agent.
 * 
 * Usage:
 *   tsx agent-oobi-resolve-le.ts <env> <agentPasscode> <agentName> <leOobi>
 * 
 * Example:
 *   tsx agent-oobi-resolve-le.ts docker myAgentPass123 jupiterSellerAgent http://keria:3902/oobi/ELEAid...
 */

import {getOrCreateClient} from "../../client/identifiers.js";
import {resolveOobi} from "../../client/oobis.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const agentPasscode = args[1];
const agentName = args[2];
const leOobi = args[3];

if (!env || !agentPasscode || !agentName || !leOobi) {
    console.error('Usage: tsx agent-oobi-resolve-le.ts <env> <agentPasscode> <agentName> <leOobi>');
    process.exit(1);
}

console.log(`Agent ${agentName} resolving LE OOBI: ${leOobi}`);
const agentClient = await getOrCreateClient(agentPasscode, env);
await resolveOobi(agentClient, leOobi, 'le');
console.log(`OOBI Resolved: ${leOobi}`);
