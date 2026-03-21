/**
 * Agent Resolves Verifier (Sally) OOBI
 * 
 * This script resolves the Sally verifier OOBI for an agent.
 * 
 * Usage:
 *   tsx agent-oobi-resolve-verifier.ts <env> <agentPasscode> <agentName>
 * 
 * Example:
 *   tsx agent-oobi-resolve-verifier.ts docker myAgentPass123 jupiterSellerAgent
 */

import {getOrCreateClient} from "../../client/identifiers.js";
import {resolveOobi} from "../../client/oobis.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const agentPasscode = args[1];
const agentName = args[2];

if (!env || !agentPasscode || !agentName) {
    console.error('Usage: tsx agent-oobi-resolve-verifier.ts <env> <agentPasscode> <agentName>');
    process.exit(1);
}

const verifierOobi = 'http://verifier:9723/oobi';
console.log(`Agent ${agentName} resolving Verifier (Sally) OOBI: ${verifierOobi}`);
const agentClient = await getOrCreateClient(agentPasscode, env);
await resolveOobi(agentClient, verifierOobi, 'verifier');
console.log(`OOBI Resolved: ${verifierOobi}`);
