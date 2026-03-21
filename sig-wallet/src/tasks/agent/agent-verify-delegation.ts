/**
 * Agent Verifies Delegation via Sally
 * 
 * This script calls Sally verifier to verify agent delegation from OOR holder.
 * 
 * Usage:
 *   tsx agent-verify-delegation.ts <dataDir> <agentName> <oorHolderName>
 * 
 * Example:
 *   tsx agent-verify-delegation.ts /task-data jupiterSellerAgent Jupiter_Chief_Sales_Officer
 */

import fs from 'fs';

const args = process.argv.slice(2);
const dataDir = args[0];
const agentName = args[1];
const oorHolderName = args[2];

if (!dataDir || !agentName || !oorHolderName) {
    console.error('Usage: tsx agent-verify-delegation.ts <dataDir> <agentName> <oorHolderName>');
    process.exit(1);
}

const agentInfoPath = `${dataDir}/${agentName}-info.json`;
if (!fs.existsSync(agentInfoPath)) {
    throw new Error(`Agent info file not found: ${agentInfoPath}`);
}
const agentInfo = JSON.parse(fs.readFileSync(agentInfoPath, 'utf-8'));

const oorHolderInfoPath = `${dataDir}/${oorHolderName}-info.json`;
if (!fs.existsSync(oorHolderInfoPath)) {
    throw new Error(`OOR Holder info file not found: ${oorHolderInfoPath}`);
}
const oorHolderInfo = JSON.parse(fs.readFileSync(oorHolderInfoPath, 'utf-8'));

console.log(`Verifying delegation for agent ${agentName}`);
console.log(`Agent AID: ${agentInfo.aid}`);
console.log(`OOR Holder AID: ${oorHolderInfo.aid}`);

const sallyUrl = 'http://vlei-verification:9723/verify/agent-delegation';
console.log(`Calling Sally verifier at ${sallyUrl}`);

try {
    const response = await fetch(sallyUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            aid: oorHolderInfo.aid,
            agent_aid: agentInfo.aid,
            verify_kel: false
        })
    });

    const result = await response.json();
    console.log('\n' + '='.repeat(60));
    console.log('SALLY VERIFICATION RESULT');
    console.log('='.repeat(60));
    console.log(JSON.stringify(result, null, 2));
    console.log('='.repeat(60) + '\n');

    if (result.valid) {
        console.log('✓ Agent delegation verified successfully');
        console.log(`  Agent: ${agentName} (${agentInfo.aid})`);
        console.log(`  Delegated from: ${oorHolderName} (${oorHolderInfo.aid})`);
        if (result.verification.le_lei) console.log(`  LE LEI: ${result.verification.le_lei}`);
        if (result.verification.qvi_aid) console.log(`  QVI AID: ${result.verification.qvi_aid}`);
        if (result.verification.geda_aid) console.log(`  GEDA AID: ${result.verification.geda_aid}`);
    } else {
        console.error('✗ Verification failed');
        console.error(`  Error: ${result.error}`);
        process.exit(1);
    }
} catch (error) {
    console.error('✗ Failed to call Sally verifier');
    console.error(`  Error: ${error}`);
    process.exit(1);
}
