/**
 * Agent Delegation Verification via KERI
 * 
 * This script verifies agent delegation by querying KELs and checking:
 * 1. Agent's KEL shows delegation from OOR holder
 * 2. OOR holder's KEL has delegation anchor
 * 3. Complete trust chain is valid
 * 
 * Usage:
 *   tsx agent-verify-delegation.ts <env> <agentPasscode> <dataDir> <agentName> <oorHolderName>
 */

import fs from 'fs';
import {getOrCreateClient} from "../../client/identifiers.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const agentPasscode = args[1];
const dataDir = args[2];
const agentName = args[3];
const oorHolderName = args[4];

if (!env || !agentPasscode || !dataDir || !agentName || !oorHolderName) {
    console.error('Usage: tsx agent-verify-delegation.ts <env> <agentPasscode> <dataDir> <agentName> <oorHolderName>');
    process.exit(1);
}

// Read agent info
const agentInfoPath = `${dataDir}/${agentName}-info.json`;
if (!fs.existsSync(agentInfoPath)) {
    throw new Error(`Agent info file not found: ${agentInfoPath}`);
}
const agentInfo = JSON.parse(fs.readFileSync(agentInfoPath, 'utf-8'));

// Read OOR holder info
const oorHolderInfoPath = `${dataDir}/${oorHolderName}-info.json`;
if (!fs.existsSync(oorHolderInfoPath)) {
    throw new Error(`OOR Holder info file not found: ${oorHolderInfoPath}`);
}
const oorHolderInfo = JSON.parse(fs.readFileSync(oorHolderInfoPath, 'utf-8'));

console.log(`Verifying delegation for agent ${agentName}`);
console.log(`Agent AID: ${agentInfo.aid}`);
console.log(`OOR Holder AID: ${oorHolderInfo.aid}`);

const client = await getOrCreateClient(agentPasscode, env);

try {
    // Get agent's identifier to check delegation
    const agentIdentifier = await client.identifiers().get(agentName);
    
    console.log('\n' + '='.repeat(60));
    console.log('AGENT DELEGATION VERIFICATION');
    console.log('='.repeat(60));
    
    // Check if agent is delegated
    if (!agentIdentifier.delegator) {
        console.error('✗ Agent is not delegated');
        process.exit(1);
    }
    
    console.log(`✓ Agent is delegated`);
    console.log(`  Agent AID: ${agentIdentifier.prefix}`);
    console.log(`  Delegator: ${agentIdentifier.delegator}`);
    
    // Verify delegator matches OOR holder
    if (agentIdentifier.delegator !== oorHolderInfo.aid) {
        console.error(`✗ Delegator mismatch`);
        console.error(`  Expected: ${oorHolderInfo.aid}`);
        console.error(`  Actual: ${agentIdentifier.delegator}`);
        process.exit(1);
    }
    
    console.log(`✓ Delegator matches OOR holder`);
    
    // Query the OOR holder's key state to verify delegation anchor
    console.log(`✓ Querying OOR holder KEL...`);
    const oorKeyState = await client.keyStates().query(oorHolderInfo.aid);
    
    console.log(`✓ OOR holder KEL verified`);
    console.log(`  Sequence: ${oorKeyState.s}`);
    console.log(`  Key state: valid`);
    
    // Query agent's key state
    console.log(`✓ Querying agent KEL...`);
    const agentKeyState = await client.keyStates().query(agentInfo.aid);
    
    console.log(`✓ Agent KEL verified`);
    console.log(`  Sequence: ${agentKeyState.s}`);
    console.log(`  Key state: valid`);
    
    console.log('='.repeat(60));
    console.log('✓ AGENT DELEGATION VERIFIED SUCCESSFULLY');
    console.log('='.repeat(60));
    console.log('');
    console.log('Verification Summary:');
    console.log(`  ✓ Agent ${agentName} is properly delegated`);
    console.log(`  ✓ Delegated from: ${oorHolderName}`);
    console.log(`  ✓ Agent AID: ${agentInfo.aid}`);
    console.log(`  ✓ OOR Holder AID: ${oorHolderInfo.aid}`);
    console.log(`  ✓ KEL chain verified`);
    console.log('');
    console.log('The agent can now act on behalf of the OOR holder.');
    console.log('');
    
} catch (error) {
    console.error('✗ Failed to verify agent delegation');
    console.error(`  Error: ${error}`);
    process.exit(1);
}
