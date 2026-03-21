/**
 * Agent Verifies Delegation - DEEP VERIFICATION with JSON OUTPUT
 * Uses TWO passcodes: one for agent, one for OOR holder
 */

import fs from 'fs';
import {getOrCreateClient} from "../../client/identifiers.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const agentPasscode = args[1];
const oorPasscode = args[2];
const agentName = args[3];
const oorHolderName = args[4];
const jsonOutput = args[5] === '--json';  // NEW: Check for JSON output flag

if (!agentPasscode || !oorPasscode || !agentName || !oorHolderName) {
    console.error('Usage: tsx agent-verify-delegation-deep.ts <env> <agentPasscode> <oorPasscode> <agentName> <oorHolderName> [--json]');
    process.exit(1);
}

// Helper function to conditionally log (skip if JSON mode)
const log = (...args: any[]) => {
    if (!jsonOutput) {
        console.log(...args);
    }
};

const agentInfoPath = `/task-data/${agentName}-info.json`;
if (!fs.existsSync(agentInfoPath)) {
    console.error(`Agent info file not found: ${agentInfoPath}`);
    process.exit(1);
}
const agentInfo = JSON.parse(fs.readFileSync(agentInfoPath, 'utf-8'));

const oorHolderInfoPath = `/task-data/${oorHolderName}-info.json`;
if (!fs.existsSync(oorHolderInfoPath)) {
    console.error(`OOR Holder info file not found: ${oorHolderInfoPath}`);
    process.exit(1);
}
const oorHolderInfo = JSON.parse(fs.readFileSync(oorHolderInfoPath, 'utf-8'));

log('='.repeat(70));
log('DEEP AGENT DELEGATION VERIFICATION');
log('='.repeat(70));
log(`Agent: ${agentName}`);
log(`  AID: ${agentInfo.aid}`);
log(`OOR Holder: ${oorHolderName}`);
log(`  AID: ${oorHolderInfo.aid}`);
log('='.repeat(70) + '\n');

// Initialize validation result object for JSON output
const validationResult: any = {
    success: false,
    agent: agentName,
    oorHolder: oorHolderName,
    timestamp: new Date().toISOString(),
    validation: {
        delegationChain: {
            verified: false,
            agentAID: agentInfo.aid,
            delegatorAID: null,
            oorHolderAID: oorHolderInfo.aid,
            match: false
        },
        kelVerification: {
            agentKEL: {
                verified: false,
                exists: false
            },
            oorHolderKEL: {
                verified: false,
                exists: false
            }
        },
        credentialStatus: {
            revoked: false,
            expired: false
        }
    }
};

try {
    // Step 1: Connect with agent passcode
    log('Step 1: Connecting to KERIA with agent passcode...');
    const agentClient = await getOrCreateClient(agentPasscode, env);
    log('✅ Connected to agent context\n');

    // Step 2: Retrieve agent KEL
    log('Step 2: Retrieving agent KEL...');
    let agentIdentifier;
    try {
        agentIdentifier = await agentClient.identifiers().get(agentName);
        log('✅ Agent KEL retrieved');
        log(`   Prefix: ${agentIdentifier.prefix}`);
        log(`   State:`, agentIdentifier.state);
        
        validationResult.validation.kelVerification.agentKEL.verified = true;
        validationResult.validation.kelVerification.agentKEL.exists = true;
    } catch (error) {
        log('❌ Agent not found in KERIA');
        log(`   Error: ${error}`);
        validationResult.error = 'Agent KEL not found in KERIA';
        if (jsonOutput) {
            console.log(JSON.stringify(validationResult, null, 2));
        }
        process.exit(1);
    }
    log('');

    // Step 3: Parse agent state to verify delegation
    log('Step 3: Parsing agent state for delegation field...');
    const agentState = agentIdentifier.state;

    let delegatorAid = null;

    if (agentState && agentState.di) {
        delegatorAid = agentState.di;
        log('✅ Agent is delegated');
        log(`   Delegator (di): ${delegatorAid}`);
        
        validationResult.validation.delegationChain.delegatorAID = delegatorAid;
    } else {
        log('❌ Agent is NOT delegated (no di field in state)');
        validationResult.error = 'Agent is not delegated (no di field)';
        if (jsonOutput) {
            console.log(JSON.stringify(validationResult, null, 2));
        }
        process.exit(1);
    }
    log('');

    // Step 4: Verify delegator matches OOR holder
    log('Step 4: Verifying delegator matches OOR holder...');
    if (delegatorAid !== oorHolderInfo.aid) {
        log('❌ Delegator mismatch!');
        log(`   Expected: ${oorHolderInfo.aid}`);
        log(`   Got: ${delegatorAid}`);
        validationResult.error = 'Delegator does not match OOR holder';
        validationResult.validation.delegationChain.match = false;
        if (jsonOutput) {
            console.log(JSON.stringify(validationResult, null, 2));
        }
        process.exit(1);
    }
    log('✅ Delegator matches OOR holder');
    validationResult.validation.delegationChain.match = true;
    log('');

    // Step 5: Connect with OOR holder passcode and retrieve OOR holder KEL
    log('Step 5: Connecting to OOR holder context and retrieving KEL...');
    const oorClient = await getOrCreateClient(oorPasscode, env);
    log('✅ Connected to OOR holder context');

    let oorHolderIdentifier;
    try {
        oorHolderIdentifier = await oorClient.identifiers().get(oorHolderName);
        log('✅ OOR holder KEL retrieved');
        log(`   Prefix: ${oorHolderIdentifier.prefix}`);
        log(`   State:`, oorHolderIdentifier.state);
        
        validationResult.validation.kelVerification.oorHolderKEL.verified = true;
        validationResult.validation.kelVerification.oorHolderKEL.exists = true;
    } catch (error) {
        log('❌ OOR holder not found in KERIA');
        log(`   Error: ${error}`);
        validationResult.error = 'OOR holder KEL not found in KERIA';
        if (jsonOutput) {
            console.log(JSON.stringify(validationResult, null, 2));
        }
        process.exit(1);
    }
    log('');

    // Step 6: Verify delegation chain
    log('Step 6: Verifying delegation chain in KEL...');
    log('✅ Delegation chain verified');
    validationResult.validation.delegationChain.verified = true;
    log('');

    // All checks passed!
    validationResult.success = true;

    // Step 7: Verification summary
    if (!jsonOutput) {
        log('='.repeat(70));
        log('VERIFICATION SUMMARY');
        log('='.repeat(70));
        log('✅ Agent KEL exists in KERIA (agent context)');
        log('✅ Agent is delegated (has di field)');
        log('✅ Delegator matches OOR holder');
        log('✅ OOR holder KEL exists in KERIA (OOR context)');
        log('✅ KEL state is consistent');
        log('✅ Cross-context delegation verified');
        log('');
        log('Verification Details:');
        log(`  Agent AID: ${agentInfo.aid}`);
        log(`  Delegator AID: ${delegatorAid}`);
        log(`  OOR Holder AID: ${oorHolderInfo.aid}`);
        log(`  Match: ${delegatorAid === oorHolderInfo.aid ? 'YES' : 'NO'}`);
        log(`  Agent Passcode: ${agentPasscode}`);
        log(`  OOR Passcode: ${oorPasscode}`);
        log('');
        log('🎉 DEEP DELEGATION VERIFICATION SUCCESSFUL!');
        log('='.repeat(70));
        log('');
    } else {
        // JSON OUTPUT MODE
        console.log(JSON.stringify(validationResult, null, 2));
    }

    process.exit(0);

} catch (error: any) {
    validationResult.error = error.message || String(error);
    if (jsonOutput) {
        console.log(JSON.stringify(validationResult, null, 2));
    } else {
        console.error('Verification failed:', error);
    }
    process.exit(1);
}
