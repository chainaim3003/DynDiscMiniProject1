/**
 * Agent Resolves QVI OOBI
 * 
 * This script resolves the QVI OOBI for an agent.
 * 
 * Usage:
 *   tsx agent-oobi-resolve-qvi.ts <env> <agentPasscode> <agentName> <qviOobi>
 * 
 * Example:
 *   tsx agent-oobi-resolve-qvi.ts docker myAgentPass123 jupiterSellerAgent http://keria:3902/oobi/EQVIAid...
 */

import {getOrCreateClient} from "../../client/identifiers.js";
import {resolveOobi} from "../../client/oobis.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const agentPasscode = args[1];
const agentName = args[2];
const qviOobi = args[3];

if (!env || !agentPasscode || !agentName || !qviOobi) {
    console.error('Usage: tsx agent-oobi-resolve-qvi.ts <env> <agentPasscode> <agentName> <qviOobi>');
    process.exit(1);
}

console.log(`Agent ${agentName} resolving QVI OOBI: ${qviOobi}`);
const agentClient = await getOrCreateClient(agentPasscode, env);
await resolveOobi(agentClient, qviOobi, 'qvi');
console.log(`OOBI Resolved: ${qviOobi}`);
