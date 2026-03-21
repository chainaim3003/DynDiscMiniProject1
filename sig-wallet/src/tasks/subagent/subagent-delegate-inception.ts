/**
 * subagent-delegate-inception.ts (FIXED - Handles Existing Agents)
 * 
 * Purpose: Create sub-agent AID delegated from parent agent
 * 
 * Pattern: Uses the SAME helper functions as person-delegate-agent-create.ts
 * FIXED: Checks if agent already exists and retrieves it instead of failing
 */

import { SignifyClient, Tier, ready } from 'signify-ts';
import fs from 'fs';
import path from 'path';
import { getOrCreateClient, createDelegate } from '../../client/identifiers.js';

const ENV = process.argv[2] || 'local';
const SUB_AGENT_ALIAS = process.argv[3];
const PARENT_AGENT_ALIAS = process.argv[4];
const TASK_DATA_DIR = process.argv[5] || './task-data';

console.log('═════════════════════════════════════════════════════════════');
console.log('SUB-AGENT INCEPTION (Step 1: Create and Request Delegation)');
console.log('═════════════════════════════════════════════════════════════');
console.log(`Environment: ${ENV}`);
console.log(`Sub-Agent: ${SUB_AGENT_ALIAS}`);
console.log(`Parent Agent: ${PARENT_AGENT_ALIAS}`);
console.log(`Task Data: ${TASK_DATA_DIR}`);
console.log('═════════════════════════════════════════════════════════════');
console.log('');

if (!SUB_AGENT_ALIAS || !PARENT_AGENT_ALIAS) {
    console.error('ERROR: Missing required arguments');
    process.exit(1);
}

async function createSubAgentInception() {
    try {
        console.log('[1/5] Loading configuration...');
        
        // Load SUB-AGENT's unique BRAN (will be used as passcode)
        const subBranPath = path.join(TASK_DATA_DIR, `${SUB_AGENT_ALIAS}-bran.txt`);
        if (!fs.existsSync(subBranPath)) {
            throw new Error(`Sub-agent BRAN not found: ${subBranPath}`);
        }
        const subBran = fs.readFileSync(subBranPath, 'utf-8').trim();
        console.log(`✓ Sub-agent BRAN loaded: ${subBran.substring(0, 20)}...`);
        
        // Load parent agent info
        const parentInfoPath = path.join(TASK_DATA_DIR, `${PARENT_AGENT_ALIAS}-info.json`);
        if (!fs.existsSync(parentInfoPath)) {
            throw new Error(`Parent agent info not found: ${parentInfoPath}`);
        }
        const parentInfo = JSON.parse(fs.readFileSync(parentInfoPath, 'utf-8'));
        const parentAid = parentInfo.aid;
        const parentOobi = parentInfo.oobi;
        console.log(`✓ Parent agent AID: ${parentAid}`);
        console.log(`✓ Parent agent OOBI: ${parentOobi}`);
        console.log('');
        
        console.log('[2/5] Creating sub-agent client...');
        
        // Use the SAME helper function as person-delegate-agent-create.ts
        const subClient = await getOrCreateClient(subBran, ENV as 'docker' | 'testnet');
        
        console.log(`✓ Sub-agent client created`);
        console.log(`  Controller: ${subClient.controller.pre}`);
        console.log(`  Agent: ${subClient.agent?.pre || 'Will be created'}`);
        console.log('');
        
        console.log('[3/5] Checking if agent already exists...');
        
        let clientInfo: any;
        let alreadyExists = false;
        
        try {
            // Try to get existing agent
            const existingAgent = await subClient.identifiers().get(SUB_AGENT_ALIAS);
            
            if (existingAgent) {
                console.log(`✓ Agent already exists in KERIA`);
                console.log(`  AID: ${existingAgent.prefix}`);
                
                alreadyExists = true;
                
                // Get operation name from the AID (format: delegation.{AID})
                const icpOpName = `delegation.${existingAgent.prefix}`;
                
                clientInfo = {
                    aid: existingAgent.prefix,
                    icpOpName: icpOpName
                };
            }
        } catch (e: any) {
            console.log(`  Agent does not exist yet, will create new one`);
        }
        
        console.log('');
        
        if (!alreadyExists) {
            console.log('[4/5] Creating delegated AID...');
            console.log(`  Delegator: ${parentAid}`);
            
            // Use the SAME helper function as person-delegate-agent-create.ts
            clientInfo = await createDelegate(
                subClient,
                SUB_AGENT_ALIAS,    // delegate name
                parentAid,          // delegator prefix (parent agent AID)
                PARENT_AGENT_ALIAS, // delegator alias name
                parentOobi          // delegator OOBI
            );
            
            console.log(`✓ Sub-agent AID created: ${clientInfo.aid}`);
            console.log(`✓ Operation: ${clientInfo.icpOpName}`);
        } else {
            console.log('[4/5] Using existing agent');
            console.log(`✓ Sub-agent AID: ${clientInfo.aid}`);
        }
        
        console.log('');
        
        console.log('[5/5] Saving delegation info...');
        
        // Add additional metadata
        const delegateInfo = {
            ...clientInfo,
            delegatorAlias: PARENT_AGENT_ALIAS,
            delegatorType: 'agent',
            isSubDelegation: true,
            subAgentAlias: SUB_AGENT_ALIAS,
            hasUniqueBran: true,
            delegator: parentAid,
            delegatorOobi: parentOobi,
            alreadyExisted: alreadyExists,
            requestedAt: new Date().toISOString()
        };
        
        const delegateInfoPath = path.join(TASK_DATA_DIR, `${SUB_AGENT_ALIAS}-delegate-info.json`);
        fs.writeFileSync(delegateInfoPath, JSON.stringify(delegateInfo, null, 2));
        
        if (!fs.existsSync(delegateInfoPath)) {
            throw new Error(`Failed to write ${delegateInfoPath}`);
        }
        
        console.log(`✓ Saved: ${SUB_AGENT_ALIAS}-delegate-info.json`);
        console.log('');
        
        console.log('═════════════════════════════════════════════════════════════');
        console.log('✅ SUB-AGENT INCEPTION COMPLETE');
        console.log('═════════════════════════════════════════════════════════════');
        console.log('');
        console.log(`Sub-Agent AID: ${clientInfo.aid}`);
        console.log(`ICP Operation: ${clientInfo.icpOpName}`);
        if (alreadyExists) {
            console.log(`Status: Using existing agent from KERIA`);
        } else {
            console.log(`Delegation request sent to ${PARENT_AGENT_ALIAS}`);
        }
        console.log('');
        console.log('Trust Chain:');
        console.log(`  OOR Holder → ${PARENT_AGENT_ALIAS} → ${SUB_AGENT_ALIAS}`);
        console.log('');
        console.log('Next: Parent agent must approve delegation');
        console.log('');
        
    } catch (error) {
        console.error('');
        console.error('═════════════════════════════════════════════════════════════');
        console.error('❌ SUB-AGENT INCEPTION FAILED');
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

createSubAgentInception();