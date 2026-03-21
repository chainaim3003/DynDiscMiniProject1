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
 */
async function finishAgentDelegation(
    agentClient: SignifyClient,
    oorHolderPre: string,
    agentName: string,
    agentIcpOpName: string,
): Promise<any> {
    console.log(`\n${'='.repeat(70)}`);
    console.log(`FINISHING AGENT DELEGATION`);
    console.log(`${'='.repeat(70)}`);
    console.log(`Agent name: ${agentName}`);
    console.log(`OOR Holder prefix: ${oorHolderPre}`);
    console.log(`Inception operation: ${agentIcpOpName}`);
    console.log(`${'='.repeat(70)}\n`);

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

    // Step 2: Wait for agent inception operation to complete
    console.log(`[2/5] Waiting for agent inception operation to complete...`);
    console.log(`This step waits for KERIA to finish processing the delegation.`);
    
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

    // Step 3: Extract and verify agent AID
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
            `CRITICAL: Agent KEL was not created in KERIA after 15 attempts (30 seconds). ` +
            `The delegation may have failed. Check KERIA logs for errors.`
        );
    }
    console.log(`✓ Step 3 complete: Agent KEL verified in KERIA\n`);

    // Step 4: Add endpoint role
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
console.log(`AGENT DELEGATION FINISH SCRIPT (ENHANCED)`);
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
    console.log(`  AID: ${oorHolderInfo.aid}\n`);

    // Read agent inception info
    console.log(`Reading agent inception info from ${agentInceptionInfoPath}...`);
    if (!fs.existsSync(agentInceptionInfoPath)) {
        throw new Error(`Agent inception info file not found: ${agentInceptionInfoPath}`);
    }
    const agentIcpInfo = JSON.parse(fs.readFileSync(agentInceptionInfoPath, 'utf-8'));
    console.log(`✓ Agent inception info loaded`);
    console.log(`  AID: ${agentIcpInfo.aid}`);
    console.log(`  Operation: ${agentIcpInfo.icpOpName}\n`);

    // Finish delegation with enhanced diagnostics
    const agentDelegationInfo: any = await finishAgentDelegation(
        agentClient, 
        oorHolderInfo.aid, 
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
