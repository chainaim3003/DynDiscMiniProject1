/**
 * subagent-delegate-finish.ts (FIXED - Handles Missing State Fields)
 */

import fs from 'fs';
import path from 'path';
import { getOrCreateClient } from '../../client/identifiers.js';
import { resolveOobi } from '../../client/oobis.js';

const ENV = process.argv[2] || 'local';
const SUB_AGENT_ALIAS = process.argv[3];
const PARENT_AGENT_ALIAS = process.argv[4];
const TASK_DATA_DIR = process.argv[5] || './task-data';

console.log('═════════════════════════════════════════════════════════════');
console.log('SUB-AGENT DELEGATION FINISH (Step 3: Complete Delegation)');
console.log('═════════════════════════════════════════════════════════════');
console.log(`Environment: ${ENV}`);
console.log(`Sub-Agent: ${SUB_AGENT_ALIAS}`);
console.log(`Parent Agent: ${PARENT_AGENT_ALIAS}`);
console.log('═════════════════════════════════════════════════════════════');
console.log('');

if (!SUB_AGENT_ALIAS || !PARENT_AGENT_ALIAS) {
    console.error('ERROR: Missing required arguments');
    process.exit(1);
}

async function finishSubDelegation() {
    try {
        console.log('[1/7] Loading configuration...');
        
        // Load sub-agent's BRAN
        const subBranPath = path.join(TASK_DATA_DIR, `${SUB_AGENT_ALIAS}-bran.txt`);
        if (!fs.existsSync(subBranPath)) {
            throw new Error(`Sub-agent BRAN not found: ${subBranPath}`);
        }
        const subBran = fs.readFileSync(subBranPath, 'utf-8').trim();
        console.log(`✓ Sub-agent BRAN loaded`);
        
        // Load delegation info
        const delegateInfoPath = path.join(TASK_DATA_DIR, `${SUB_AGENT_ALIAS}-delegate-info.json`);
        if (!fs.existsSync(delegateInfoPath)) {
            throw new Error(`Delegation info not found: ${delegateInfoPath}`);
        }
        const delegateInfo = JSON.parse(fs.readFileSync(delegateInfoPath, 'utf-8'));
        const subAid = delegateInfo.aid;
        const parentOobi = delegateInfo.delegatorOobi;
        
        console.log(`✓ Sub-agent AID: ${subAid}`);
        console.log('');
        
        console.log('[2/7] Connecting as sub-agent...');
        
        // Use helper function
        const subClient = await getOrCreateClient(subBran, ENV as 'docker' | 'testnet');
        
        console.log(`✓ Connected as sub-agent`);
        console.log(`  Controller: ${subClient.controller.pre}`);
        console.log(`  Agent: ${subClient.agent!.pre}`);
        console.log('');
        
        console.log('[3/7] Resolving parent agent OOBI...');
        console.log(`  Parent OOBI: ${parentOobi}`);
        
        await resolveOobi(subClient, parentOobi, PARENT_AGENT_ALIAS);
        console.log(`✓ Parent agent OOBI resolved`);
        console.log('');
        
        console.log('[4/7] Querying parent key state...');
        const parentAid = delegateInfo.delegator;
        await subClient.keyStates().query(parentAid, '1');
        console.log(`✓ Parent key state queried`);
        console.log('');
        
        console.log('[5/7] Waiting for delegation to propagate...');
        await new Promise(resolve => setTimeout(resolve, 5000));
        console.log('✓ Wait complete');
        console.log('');
        
        console.log('[6/7] Adding endpoint role...');
        
        const roleResult = await subClient.identifiers().addEndRole(
            SUB_AGENT_ALIAS,
            'agent',
            subClient.agent!.pre
        );
        await roleResult.op();
        console.log(`✓ Endpoint role added`);
        console.log('');
        
        console.log('[7/7] Getting sub-agent details...');
        
        // Get OOBI
        const subOobi = await subClient.oobis().get(SUB_AGENT_ALIAS, 'agent');
        const subOobiUrl = subOobi.oobis[0];
        console.log(`✓ Sub-agent OOBI: ${subOobiUrl}`);
        
        // Get identifier to get state
        const identifier = await subClient.identifiers().get(SUB_AGENT_ALIAS);
        
        // Save final sub-agent info
        const subInfo = {
            aid: subAid,
            alias: SUB_AGENT_ALIAS,
            oobi: subOobiUrl,
            delegator: delegateInfo.delegator,
            delegatorAlias: PARENT_AGENT_ALIAS,
            delegatorType: 'agent',
            isSubDelegation: true,
            hasUniqueBran: true,
            publicKey: identifier.state?.k?.[0] || 'N/A',
            state: identifier.state,
            createdAt: new Date().toISOString()
        };
        
        const subInfoPath = path.join(TASK_DATA_DIR, `${SUB_AGENT_ALIAS}-info.json`);
        fs.writeFileSync(subInfoPath, JSON.stringify(subInfo, null, 2));
        console.log(`✓ Sub-agent info saved: ${subInfoPath}`);
        console.log('');
        
        console.log('═════════════════════════════════════════════════════════════');
        console.log('✅ SUB-AGENT DELEGATION COMPLETE');
        console.log('═════════════════════════════════════════════════════════════');
        console.log('');
        console.log('Trust Chain:');
        console.log(`  OOR Holder → ${PARENT_AGENT_ALIAS} → ${SUB_AGENT_ALIAS}`);
        console.log('');
        console.log('Sub-Agent Details:');
        console.log(`  AID: ${subAid}`);
        console.log(`  OOBI: ${subOobiUrl}`);
        console.log('');
        
    } catch (error) {
        console.error('');
        console.error('═════════════════════════════════════════════════════════════');
        console.error('❌ SUB-AGENT DELEGATION FINISH FAILED');
        console.error('═════════════════════════════════════════════════════════════');
        console.error('');
        console.error('Error:', error);
        console.error('');
        if ((error as Error).stack) {
            console.error('Stack:', (error as Error).stack);
        }
        process.exit(1);
    }
}

finishSubDelegation();