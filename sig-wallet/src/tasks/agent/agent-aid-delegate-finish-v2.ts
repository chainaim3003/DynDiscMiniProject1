/**
 * Agent Delegation Finish Script v2 - WITH OOBI RESOLUTION FIX
 * 
 * This script completes the agent delegation process by:
 * 1. CRITICAL FIX: Resolving the OOR holder's OOBI first
 * 2. Querying the OOR holder's key state (to get the delegation anchor)
 * 3. Waiting for the agent inception operation to complete
 * 4. Adding endpoint role for the agent
 * 5. Getting the agent's OOBI
 * 
 * The key insight from vLEI training (101_47_Delegated_AIDs.md):
 * - Delegation is a COOPERATIVE process
 * - The delegate (agent) needs to query the delegator's (OOR holder) KEL
 *   to find the anchoring interaction event that approves the delegation
 * - To query another AID's key state, you MUST first resolve their OOBI
 * 
 * From KERIA/Signify docs (102_05_KERIA_Signify.md):
 * - Each Signify client connects to KERIA with its own session
 * - OOBIs must be resolved in each session to establish contact
 * 
 * Usage:
 *   tsx agent-aid-delegate-finish-v2.ts <env> <passcode> <agentName> <oorHolderInfoPath> <agentInceptionInfoPath> <outputPath>
 */

import fs from 'fs';
import {getOrCreateClient} from "../../client/identifiers.js";
import {resolveOobi} from "../../client/oobis.js";
import {SignifyClient} from "signify-ts";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1];
const agentAidName = args[2];
const oorHolderInfoPath = args[3];
const agentInceptionInfoPath = args[4];
const agentOutputPath = args[5];

// ============================================================================
// ENHANCED HELPER FUNCTIONS
// ============================================================================

/**
 * Wait for operation with configurable timeout and detailed logging
 */
async function waitOperationWithTimeout(
    client: SignifyClient,
    op: any,
    timeoutMs: number = 180000,
    operationName: string = "operation"
): Promise<any> {
    console.log(`  Waiting for ${operationName} (timeout: ${timeoutMs/1000}s)...`);
    try {
        const result = await client
            .operations()
            .wait(op, { signal: AbortSignal.timeout(timeoutMs) });
        console.log(`  ✓ ${operationName} completed successfully`);
        return result;
    } catch (error: any) {
        if (error.name === 'TimeoutError' || error.name === 'AbortError') {
            console.error(`  ✗ ${operationName} timed out after ${timeoutMs/1000}s`);
            console.error(`  Operation name: ${op.name}`);
            console.error(`  Operation done: ${op.done}`);
            throw new Error(`${operationName} timed out - witness receipts may not be propagating`);
        }
        throw error;
    }
}

/**
 * CRITICAL FIX: Resolve OOBI with retries and detailed logging
 * 
 * According to KERIA/Signify documentation:
 * - OOBIs must be resolved to establish contact with another AID
 * - Without OOBI resolution, key state queries will timeout
 */
async function resolveOobiWithRetries(
    client: SignifyClient,
    oobi: string,
    alias: string,
    maxRetries: number = 3,
    retryDelayMs: number = 2000
): Promise<void> {
    console.log(`\nResolving OOBI for ${alias}...`);
    console.log(`  OOBI: ${oobi}`);
    console.log(`  Max retries: ${maxRetries}`);
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            console.log(`  Attempt ${attempt}/${maxRetries}...`);
            const op = await client.oobis().resolve(oobi, alias);
            
            // Wait for OOBI resolution with 30 second timeout per attempt
            await waitOperationWithTimeout(
                client,
                op,
                30000,
                `OOBI resolution (attempt ${attempt})`
            );
            
            console.log(`✓ OOBI resolved for ${alias}`);
            return;
            
        } catch (error: any) {
            console.log(`  Attempt ${attempt}/${maxRetries} failed: ${error.message}`);
            
            if (attempt < maxRetries) {
                console.log(`  Waiting ${retryDelayMs}ms before retry...`);
                await new Promise(resolve => setTimeout(resolve, retryDelayMs));
            } else {
                throw new Error(
                    `Failed to resolve OOBI after ${maxRetries} attempts. ` +
                    `Check: (1) OOR holder service is running, ` +
                    `(2) KERIA has network access to witnesses`
                );
            }
        }
    }
}

/**
 * Query key state with retries
 * 
 * According to vLEI training (101_47_Delegated_AIDs.md):
 * - The delegate needs to query the delegator's KEL to find the anchor
 * - This anchor contains the seal proving delegation approval
 */
async function queryKeyStateWithRetries(
    client: SignifyClient,
    prefix: string,
    sn: string = '1',
    maxRetries: number = 5,
    retryDelayMs: number = 3000
): Promise<any> {
    console.log(`\nQuerying key state for ${prefix} at sequence ${sn}...`);
    console.log(`  Max retries: ${maxRetries}, Delay: ${retryDelayMs}ms`);
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            console.log(`  Attempt ${attempt}/${maxRetries}...`);
            
            const op: any = await client.keyStates().query(prefix, sn);
            console.log(`  Query operation: ${op.name}`);
            console.log(`  Operation done: ${op.done}`);
            
            // 60 seconds per attempt
            const result = await waitOperationWithTimeout(
                client,
                op,
                60000,
                `Key state query (attempt ${attempt}/${maxRetries})`
            );
            
            console.log(`✓ Key state query successful on attempt ${attempt}`);
            return result;
            
        } catch (error: any) {
            console.log(`  Attempt ${attempt}/${maxRetries} failed: ${error.message}`);
            
            if (attempt < maxRetries) {
                console.log(`  Waiting ${retryDelayMs}ms before retry...`);
                await new Promise(resolve => setTimeout(resolve, retryDelayMs));
            } else {
                throw new Error(
                    `Failed to query key state after ${maxRetries} attempts. ` +
                    `The interaction event (delegation anchor) may not have been ` +
                    `witnessed yet. Check witness logs and network connectivity.`
                );
            }
        }
    }
}

/**
 * Verify identifier exists in KERIA
 */
async function verifyIdentifierExists(
    client: SignifyClient,
    name: string,
    expectedPrefix: string,
    maxRetries: number = 15,
    retryDelayMs: number = 2000
): Promise<boolean> {
    console.log(`\nVerifying identifier ${name} exists in KERIA...`);
    console.log(`  Expected prefix: ${expectedPrefix}`);
    
    for (let i = 0; i < maxRetries; i++) {
        try {
            const aid = await client.identifiers().get(name);
            if (aid && aid.prefix === expectedPrefix) {
                console.log(`✓ Identifier verified (attempt ${i + 1}/${maxRetries})`);
                return true;
            }
        } catch (error: any) {
            console.log(`  Attempt ${i + 1}/${maxRetries}: Not found yet...`);
        }
        
        if (i < maxRetries - 1) {
            await new Promise(resolve => setTimeout(resolve, retryDelayMs));
        }
    }
    
    return false;
}

// ============================================================================
// MAIN DELEGATION FINISH FUNCTION
// ============================================================================

async function finishAgentDelegation(
    agentClient: SignifyClient,
    oorHolderPre: string,
    oorHolderOobi: string,
    oorHolderName: string,
    agentName: string,
    agentIcpOpName: string,
): Promise<any> {
    console.log(`\n${'═'.repeat(70)}`);
    console.log(`FINISHING AGENT DELEGATION (v2 - WITH OOBI FIX)`);
    console.log(`${'═'.repeat(70)}`);
    console.log(`Agent name: ${agentName}`);
    console.log(`OOR Holder name: ${oorHolderName}`);
    console.log(`OOR Holder prefix: ${oorHolderPre}`);
    console.log(`OOR Holder OOBI: ${oorHolderOobi}`);
    console.log(`Inception operation: ${agentIcpOpName}`);
    console.log(`${'═'.repeat(70)}\n`);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 0: CRITICAL FIX - Resolve OOR Holder's OOBI
    // ═══════════════════════════════════════════════════════════════════════
    console.log(`[0/5] RESOLVING OOR HOLDER'S OOBI (CRITICAL)`);
    console.log(`This step is REQUIRED before querying key state.`);
    console.log(`Without OOBI resolution, the agent doesn't know how to reach`);
    console.log(`the OOR holder to verify the delegation anchor.\n`);
    
    try {
        await resolveOobiWithRetries(
            agentClient,
            oorHolderOobi,
            oorHolderName,
            3,
            2000
        );
        console.log(`✓ Step 0 complete: OOR Holder OOBI resolved\n`);
    } catch (error: any) {
        console.error(`\n✗ CRITICAL ERROR in Step 0: ${error.message}`);
        console.error(`Without OOBI resolution, delegation cannot complete.`);
        throw error;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 1: Query OOR Holder key state to discover delegation anchor
    // ═══════════════════════════════════════════════════════════════════════
    console.log(`[1/5] Querying OOR Holder key state to find delegation anchor...`);
    console.log(`This retrieves the interaction event (s='1') where the OOR holder`);
    console.log(`anchored the delegation approval seal.\n`);
    
    try {
        await queryKeyStateWithRetries(agentClient, oorHolderPre, '1', 5, 3000);
        console.log(`✓ Step 1 complete: OOR Holder key state retrieved\n`);
    } catch (error: any) {
        console.error(`\n✗ CRITICAL ERROR in Step 1: ${error.message}\n`);
        throw error;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 2: Wait for agent inception operation to complete
    // ═══════════════════════════════════════════════════════════════════════
    console.log(`[2/5] Waiting for agent inception operation to complete...`);
    console.log(`This waits for KERIA to finish processing the delegation.\n`);
    
    try {
        const agentOp: any = await agentClient.operations().get(agentIcpOpName);
        console.log(`Inception operation status:`);
        console.log(`  Name: ${agentOp.name}`);
        console.log(`  Done: ${agentOp.done}`);
        
        await waitOperationWithTimeout(
            agentClient,
            agentOp,
            180000,  // 3 minutes
            "Agent inception operation"
        );
        console.log(`✓ Step 2 complete: Inception operation finished\n`);
    } catch (error: any) {
        console.error(`\n✗ CRITICAL ERROR in Step 2: ${error.message}\n`);
        throw error;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 3: Extract and verify agent AID
    // ═══════════════════════════════════════════════════════════════════════
    console.log(`[3/5] Extracting and verifying agent AID...`);
    const agentPre = agentIcpOpName.split('.')[1];
    console.log(`  Extracted prefix: ${agentPre}\n`);

    const kelExists = await verifyIdentifierExists(
        agentClient,
        agentName,
        agentPre,
        15,
        2000
    );

    if (!kelExists) {
        throw new Error(
            `CRITICAL: Agent KEL was not created in KERIA after 15 attempts. ` +
            `The delegation may have failed. Check KERIA logs.`
        );
    }
    console.log(`✓ Step 3 complete: Agent KEL verified in KERIA\n`);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 4: Add endpoint role
    // ═══════════════════════════════════════════════════════════════════════
    console.log(`[4/5] Adding endpoint role for agent...`);
    console.log(`This makes the agent discoverable via OOBIs.\n`);
    
    try {
        const endRoleOp = await agentClient.identifiers()
            .addEndRole(agentName, 'agent', agentClient!.agent!.pre);
        await waitOperationWithTimeout(
            agentClient,
            await endRoleOp.op(),
            60000,
            "Add endpoint role"
        );
        console.log(`✓ Step 4 complete: Endpoint role added\n`);
    } catch (error: any) {
        console.error(`\n✗ CRITICAL ERROR in Step 4: ${error.message}\n`);
        throw error;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 5: Get OOBI and final verification
    // ═══════════════════════════════════════════════════════════════════════
    console.log(`[5/5] Getting OOBI and performing final verification...`);
    
    try {
        const oobiResp = await agentClient.oobis().get(agentName, 'agent');
        const oobi = oobiResp.oobis[0];
        console.log(`  OOBI retrieved: ${oobi}`);

        const finalAid = await agentClient.identifiers().get(agentName);
        console.log(`  Final agent state:`);
        console.log(`    Prefix: ${finalAid.prefix}`);
        console.log(`    Name: ${agentName}`);
        
        console.log(`\n${'═'.repeat(70)}`);
        console.log(`✓✓✓ AGENT DELEGATION SUCCESSFULLY COMPLETED ✓✓✓`);
        console.log(`${'═'.repeat(70)}\n`);

        return {
            aid: finalAid.prefix,
            oobi,
            state: finalAid.state
        };
    } catch (error: any) {
        console.error(`\n✗ CRITICAL ERROR in Step 5: ${error.message}\n`);
        throw error;
    }
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

console.log(`\n${'═'.repeat(70)}`);
console.log(`AGENT DELEGATION FINISH SCRIPT (v2 - WITH OOBI FIX)`);
console.log(`${'═'.repeat(70)}`);
console.log(`Based on vLEI training documentation:`);
console.log(`  - 101_47_Delegated_AIDs.md: Cooperative delegation process`);
console.log(`  - 102_05_KERIA_Signify.md: OOBI resolution requirements`);
console.log(`${'═'.repeat(70)}`);
console.log(`Environment: ${env}`);
console.log(`Agent name: ${agentAidName}`);
console.log(`OOR holder info: ${oorHolderInfoPath}`);
console.log(`Agent inception info: ${agentInceptionInfoPath}`);
console.log(`Output path: ${agentOutputPath}`);
console.log(`${'═'.repeat(70)}\n`);

try {
    // Initialize agent client
    console.log(`Initializing agent client...`);
    const agentClient = await getOrCreateClient(passcode, env);
    console.log(`✓ Agent client initialized`);
    console.log(`  Controller: ${agentClient.controller.pre}`);
    console.log(`  Agent: ${agentClient.agent?.pre}\n`);

    // Read OOR Holder info
    console.log(`Reading OOR Holder info from ${oorHolderInfoPath}...`);
    if (!fs.existsSync(oorHolderInfoPath)) {
        throw new Error(`OOR Holder info file not found: ${oorHolderInfoPath}`);
    }
    const oorHolderInfo = JSON.parse(fs.readFileSync(oorHolderInfoPath, 'utf-8'));
    console.log(`✓ OOR Holder info loaded`);
    console.log(`  AID: ${oorHolderInfo.aid}`);
    console.log(`  OOBI: ${oorHolderInfo.oobi}\n`);

    // Read agent inception info
    console.log(`Reading agent inception info from ${agentInceptionInfoPath}...`);
    if (!fs.existsSync(agentInceptionInfoPath)) {
        throw new Error(`Agent inception info file not found: ${agentInceptionInfoPath}`);
    }
    const agentIcpInfo = JSON.parse(fs.readFileSync(agentInceptionInfoPath, 'utf-8'));
    console.log(`✓ Agent inception info loaded`);
    console.log(`  AID: ${agentIcpInfo.aid}`);
    console.log(`  Operation: ${agentIcpInfo.icpOpName}\n`);

    // Extract OOR holder name from info path
    // Format: /task-data/Jupiter_Chief_Sales_Officer-info.json
    const oorHolderName = oorHolderInfoPath.split('/').pop()?.replace('-info.json', '') || 'oor-holder';

    // Finish delegation with enhanced diagnostics
    const agentDelegationInfo: any = await finishAgentDelegation(
        agentClient, 
        oorHolderInfo.aid,
        oorHolderInfo.oobi,
        oorHolderName,
        agentAidName, 
        agentIcpInfo.icpOpName
    );

    // Write to file
    console.log(`Writing agent info to ${agentOutputPath}...`);
    fs.writeFileSync(agentOutputPath, JSON.stringify(agentDelegationInfo, null, 2));

    if (!fs.existsSync(agentOutputPath)) {
        throw new Error(`Failed to write ${agentOutputPath}`);
    }

    console.log(`✓ Agent delegation data written successfully\n`);
    console.log(`${'═'.repeat(70)}`);
    console.log(`SUCCESS: Agent ${agentAidName} fully delegated and operational`);
    console.log(`${'═'.repeat(70)}\n`);
    
    process.exit(0);
    
} catch (error: any) {
    console.error(`\n${'═'.repeat(70)}`);
    console.error(`CRITICAL ERROR: Agent delegation failed`);
    console.error(`${'═'.repeat(70)}`);
    console.error(`Error type: ${error.name}`);
    console.error(`Error message: ${error.message}`);
    if (error.stack) {
        console.error(`\nStack trace:`);
        console.error(error.stack);
    }
    console.error(`${'═'.repeat(70)}\n`);
    
    console.error(`TROUBLESHOOTING:`);
    console.error(`1. Check Docker services: docker compose ps`);
    console.error(`2. Check witness logs: docker compose logs witness`);
    console.error(`3. Check KERIA logs: docker compose logs keria`);
    console.error(`4. Verify OOR holder has witnesses configured`);
    console.error(`5. Ensure OOR holder's OOBI is reachable`);
    console.error(`6. Check network connectivity between services\n`);
    
    process.exit(1);
}
