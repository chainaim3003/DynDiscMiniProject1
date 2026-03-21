import fs from 'fs';
import {getOrCreateClient} from "../../client/identifiers.js";
import {waitOperation} from "../../client/operations.js";
import {SignifyClient} from "signify-ts";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1];
const agentAidName = args[2];
const oorHolderInfoPath = args[3];
const agentInceptionInfoPath = args[4];
const agentOutputPath = args[5];

/**
 * Enhanced wait operation with custom timeout and better error messages
 */
async function waitOperationWithTimeout<T = any>(
    client: SignifyClient,
    op: any,
    timeoutMs: number = 180000,  // Increased to 3 minutes
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
            throw new Error(`${operationName} timed out - witness receipts may not be propagating properly`);
        }
        throw error;
    }
}

/**
 * CRITICAL FIX: Resolve OOBI with retries
 * 
 * According to vLEI training (102_05_KERIA_Signify.md):
 * - Each Signify client session requires OOBI resolution to establish contact
 * - Without OOBI resolution, key state queries will timeout
 */
async function resolveOobiWithRetries(
    client: SignifyClient,
    oobi: string,
    alias: string,
    maxRetries: number = 3,
    retryDelayMs: number = 2000
): Promise<void> {
    console.log(`Resolving OOBI for ${alias}...`);
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
 * Query key state with retries and diagnostic information
 */
async function queryKeyStateWithRetries(
    client: SignifyClient,
    prefix: string,
    maxRetries: number = 5,
    retryDelayMs: number = 3000
): Promise<any> {
    console.log(`Querying key state for ${prefix}...`);
    console.log(`  Max retries: ${maxRetries}, Delay between retries: ${retryDelayMs}ms`);
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            console.log(`  Attempt ${attempt}/${maxRetries}...`);
            
            // Query the key state
            const op: any = await client.keyStates().query(prefix, '1');
            console.log(`  Query operation created: ${op.name}`);
            console.log(`  Operation done: ${op.done}`);
            
            // Wait for the operation with increased timeout (60s per attempt)
            const result = await waitOperationWithTimeout(
                client,
                op,
                60000,  // 60 seconds per attempt
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
                console.error(`✗ All ${maxRetries} attempts failed`);
                throw new Error(
                    `Failed to query key state after ${maxRetries} attempts. ` +
                    `This usually means witness receipts are not being received. ` +
                    `Check: (1) Witnesses are running, (2) OOR holder has witnesses configured, ` +
                    `(3) Network connectivity between services`
                );
            }
        }
    }
}

/**
 * Verify identifier exists with enhanced diagnostics
 */
async function verifyIdentifierExists(
    client: SignifyClient,
    name: string,
    expectedPrefix: string,
    maxRetries: number = 15,
    retryDelayMs: number = 2000
): Promise<boolean> {
    console.log(`Verifying identifier ${name} exists in KERIA...`);
    console.log(`  Expected prefix: ${expectedPrefix}`);
    
    for (let i = 0; i < maxRetries; i++) {
        try {
            const aid = await client.identifiers().get(name);
            if (aid && aid.prefix === expectedPrefix) {
                console.log(`✓ Identifier verified (attempt ${i + 1}/${maxRetries})`);
                console.log(`  Prefix: ${aid.prefix}`);
                console.log(`  State: ${JSON.stringify(aid.state)}`);
                return true;
            }
        } catch (error: any) {
            console.log(`  Attempt ${i + 1}/${maxRetries}: Not found yet...`);
        }
        
        if (i < maxRetries - 1) {
            console.log(`  Waiting ${retryDelayMs}ms before next check...`);
            await new Promise(resolve => setTimeout(resolve, retryDelayMs));
        }
    }
    
    return false;
}

/**
 * Main delegation finish function with comprehensive diagnostics
 * FIXED: Now includes OOBI resolution as Step 0
 */
async function finishAgentDelegation(
    agentClient: SignifyClient,
    oorHolderPre: string,
    oorHolderOobi: string,
    oorHolderName: string,
    agentName: string,
    agentIcpOpName: string,
): Promise<any> {
    console.log(`\n${'='.repeat(70)}`);
    console.log(`FINISHING AGENT DELEGATION (WITH OOBI FIX)`);
    console.log(`${'='.repeat(70)}`);
    console.log(`Agent name: ${agentName}`);
    console.log(`OOR Holder name: ${oorHolderName}`);
    console.log(`OOR Holder prefix: ${oorHolderPre}`);
    console.log(`OOR Holder OOBI: ${oorHolderOobi}`);
    console.log(`Inception operation: ${agentIcpOpName}`);
    console.log(`${'='.repeat(70)}\n`);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 0: CRITICAL FIX - Resolve OOR Holder's OOBI first
    // Without this, the agent client can't reach the OOR holder to query
    // their key state and find the delegation anchor
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

    // Step 1: Query OOR Holder key state to discover delegation anchor
    console.log(`[1/5] Querying OOR Holder key state to find delegation anchor...`);
    console.log(`This step retrieves the interaction event where the OOR holder`);
    console.log(`anchored the delegation approval seal.`);
    
    try {
        await queryKeyStateWithRetries(agentClient, oorHolderPre, 5, 3000);
        console.log(`✓ Step 1 complete: OOR Holder key state retrieved\n`);
    } catch (error: any) {
        console.error(`\n✗ CRITICAL ERROR in Step 1: ${error.message}\n`);
        throw error;
    }

    // Step 2: Wait for delegation to propagate (operation query often 404s because it completes instantly)
    console.log(`[2/5] Waiting for delegation to propagate through the network...`);
    console.log(`Note: Delegation operations complete very quickly with witnesses (toad=1).`);
    console.log(`The operation may already be cleared from the queue, which is normal.\n`);
    
    try {
        // Try to get the operation, but don't fail if it 404s
        try {
            const agentOp: any = await agentClient.operations().get(agentIcpOpName);
            console.log(`  Operation still in queue: ${agentOp.name}`);
            console.log(`  Operation done: ${agentOp.done}`);
            
            if (!agentOp.done) {
                console.log(`  Waiting for operation to complete...`);
                await waitOperationWithTimeout(
                    agentClient,
                    agentOp,
                    60000,  // 1 minute
                    "Agent inception operation"
                );
            }
        } catch (opError: any) {
            if (opError.message.includes('404')) {
                console.log(`  Operation already cleared from queue (404) - this is normal`);
                console.log(`  Proceeding with verification...`);
            } else {
                throw opError;
            }
        }
        
        // Wait for propagation regardless of operation status
        console.log(`  Waiting 5 seconds for KEL propagation...`);
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        console.log(`✓ Step 2 complete: Delegation propagation wait finished\n`);
    } catch (error: any) {
        console.error(`\n✗ CRITICAL ERROR in Step 2: ${error.message}\n`);
        throw error;
    }

    // Step 3: Verify agent identifier exists in KERIA
    console.log(`[3/5] Extracting and verifying agent AID...`);
    
    const agentPre = agentIcpOpName.split('.')[1];
    console.log(`  Extracted prefix from operation: ${agentPre}`);

    const kelExists = await verifyIdentifierExists(
        agentClient,
        agentName,
        agentPre,
        15,
        2000
    );

    if (!kelExists) {
        throw new Error(
            `Agent KEL was not created in KERIA after 15 attempts (30 seconds). ` +
            `Delegation may have failed or witness receipts not propagating.`
        );
    }
    console.log(`✓ Step 3 complete: Agent KEL verified in KERIA\n`);

    // Step 4: Add endpoint role for agent
    console.log(`[4/5] Adding endpoint role for agent...`);
    console.log(`This makes the agent discoverable via OOBIs.`);
    
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

    // Step 5: Get OOBI and perform final verification
    console.log(`[5/5] Getting OOBI and performing final verification...`);
    
    try {
        const oobiResp = await agentClient.oobis().get(agentName, 'agent');
        const oobi = oobiResp.oobis[0];
        console.log(`  OOBI retrieved: ${oobi}`);

        const finalAid = await agentClient.identifiers().get(agentName);
        console.log(`  Final agent state:`);
        console.log(`    Prefix: ${finalAid.prefix}`);
        console.log(`    Name: ${agentName}`);
        console.log(`    State: ${JSON.stringify(finalAid.state, null, 2)}`);
        
        console.log(`✓ Step 5 complete: Agent fully configured\n`);
        console.log(`${'='.repeat(70)}`);
        console.log(`✓✓✓ AGENT DELEGATION SUCCESSFULLY COMPLETED ✓✓✓`);
        console.log(`${'='.repeat(70)}\n`);

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

console.log(`\n${'='.repeat(70)}`);
console.log(`AGENT DELEGATION FINISH SCRIPT (WITH OOBI FIX)`);
console.log(`${'='.repeat(70)}`);
console.log(`Environment: ${env}`);
console.log(`Agent name: ${agentAidName}`);
console.log(`OOR holder info: ${oorHolderInfoPath}`);
console.log(`Agent inception info: ${agentInceptionInfoPath}`);
console.log(`Output path: ${agentOutputPath}`);
console.log(`${'='.repeat(70)}\n`);

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

    // Finish delegation with OOBI resolution fix
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

    // Verify file was written
    if (!fs.existsSync(agentOutputPath)) {
        throw new Error(`Failed to write ${agentOutputPath}`);
    }

    console.log(`✓ Agent delegation data written successfully\n`);
    console.log(`${'='.repeat(70)}`);
    console.log(`SUCCESS: Agent ${agentAidName} fully delegated and operational`);
    console.log(`${'='.repeat(70)}\n`);
    
    process.exit(0);
    
} catch (error: any) {
    console.error(`\n${'='.repeat(70)}`);
    console.error(`CRITICAL ERROR: Agent delegation failed`);
    console.error(`${'='.repeat(70)}`);
    console.error(`Error type: ${error.name}`);
    console.error(`Error message: ${error.message}`);
    if (error.stack) {
        console.error(`\nStack trace:`);
        console.error(error.stack);
    }
    console.error(`${'='.repeat(70)}\n`);
    
    // Provide troubleshooting guidance
    console.error(`TROUBLESHOOTING STEPS:`);
    console.error(`1. Check that all Docker services are running: docker compose ps`);
    console.error(`2. Check witness logs: docker compose logs witness`);
    console.error(`3. Check KERIA logs: docker compose logs keria`);
    console.error(`4. Verify OOR holder has witnesses configured properly`);
    console.error(`5. Ensure witness receipts are being received`);
    console.error(`6. Check network connectivity between services\n`);
    
    process.exit(1);
}
