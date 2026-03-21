/**
 * subagent-delegate-approve.ts (CORRECTED - Uses Helper Functions)
 * 
 * Purpose: Parent agent approves sub-agent delegation
 * 
 * Pattern: Uses the SAME helper functions as the working code
 */

import fs from 'fs';
import path from 'path';
import { getOrCreateClient, approveDelegation } from '../../client/identifiers.js';

const ENV = process.argv[2] || 'local';
const PARENT_AGENT_ALIAS = process.argv[3];
const SUB_AGENT_ALIAS = process.argv[4];
const TASK_DATA_DIR = process.argv[5] || './task-data';

console.log('═════════════════════════════════════════════════════════════');
console.log('SUB-AGENT DELEGATION APPROVE (Step 2: Parent Approves)');
console.log('═════════════════════════════════════════════════════════════');
console.log(`Environment: ${ENV}`);
console.log(`Parent Agent: ${PARENT_AGENT_ALIAS}`);
console.log(`Sub-Agent: ${SUB_AGENT_ALIAS}`);
console.log('═════════════════════════════════════════════════════════════');
console.log('');

if (!PARENT_AGENT_ALIAS || !SUB_AGENT_ALIAS) {
    console.error('ERROR: Missing required arguments');
    process.exit(1);
}

async function approveSubDelegation() {
    try {
        console.log('[1/4] Loading parent agent BRAN...');
        
        // Load PARENT agent's BRAN from agent-brans.json
        const agentBransPath = path.join(TASK_DATA_DIR, 'agent-brans.json');
        if (!fs.existsSync(agentBransPath)) {
            throw new Error(`agent-brans.json not found: ${agentBransPath}`);
        }
        
        const agentBrans = JSON.parse(fs.readFileSync(agentBransPath, 'utf-8'));
        const parentBranObj = agentBrans.agents.find((a: any) => a.alias === PARENT_AGENT_ALIAS);
        
        if (!parentBranObj) {
            throw new Error(`Parent agent ${PARENT_AGENT_ALIAS} not found in agent-brans.json`);
        }
        
        const parentBran = parentBranObj.bran;
        console.log(`✓ Parent agent BRAN loaded: ${parentBran.substring(0, 20)}...`);
        console.log('');
        
        console.log('[2/4] Connecting as parent agent...');
        
        // Use the SAME helper function - this will connect to the existing agent
        const parentClient = await getOrCreateClient(parentBran, ENV as 'docker' | 'testnet');
        
        console.log(`✓ Connected as parent agent`);
        console.log(`  Controller: ${parentClient.controller.pre}`);
        console.log(`  Agent: ${parentClient.agent!.pre}`);
        console.log('');
        
        console.log('[3/4] Loading sub-agent delegation request...');
        
        const delegateInfoPath = path.join(TASK_DATA_DIR, `${SUB_AGENT_ALIAS}-delegate-info.json`);
        if (!fs.existsSync(delegateInfoPath)) {
            throw new Error(`Sub-agent delegation info not found: ${delegateInfoPath}`);
        }
        const delegateInfo = JSON.parse(fs.readFileSync(delegateInfoPath, 'utf-8'));
        
        const subAid = delegateInfo.aid;
        
        console.log(`✓ Sub-agent AID: ${subAid}`);
        console.log('');
        
        console.log('[4/4] Approving delegation...');
        
        // Use the SAME helper function as the working code
        const approved = await approveDelegation(
            parentClient,
            PARENT_AGENT_ALIAS,
            subAid
        );
        
        if (approved) {
            console.log(`✓ Delegation approved`);
        } else {
            throw new Error('Delegation approval failed');
        }
        
        console.log('');
        console.log('═════════════════════════════════════════════════════════════');
        console.log('✅ PARENT AGENT APPROVAL COMPLETE');
        console.log('═════════════════════════════════════════════════════════════');
        console.log('');
        console.log('Next: Sub-agent must complete delegation');
        console.log('');
        
    } catch (error) {
        console.error('');
        console.error('═════════════════════════════════════════════════════════════');
        console.error('❌ PARENT AGENT APPROVAL FAILED');
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

approveSubDelegation();