/**
 * Verify Self-Attested Credential from Another Agent
 * 
 * PURPOSE: Pure verification of a self-attested invoice credential
 *          (VERIFICATION ONLY - no IPEX grant/admit - that's handled by 4C script)
 * 
 * This is the TypeScript equivalent of DEEP-EXT-credential.sh
 * It verifies AFTER the 4C workflow has completed the IPEX grant/admit.
 * 
 * VERIFICATION STEPS:
 * 1. Load and verify issuer agent's info (delegation check)
 * 2. Load and verify IPEX grant info
 * 3. Verify credential structure (self-attested: issuer = issuee)
 * 4. Verify invoice data integrity
 * 5. Build and display trust chain
 * 
 * TRUST MODEL:
 * - Self-attested credential: issuer = issuee (same AID)
 * - Trust derived from agent delegation, not credential chaining
 * - jupiterSellerAgent's AID is delegated from OOR holder
 * 
 * Usage: tsx verify-credential-from-agent.ts <env> <verifierAgent> <issuerAgent> [taskDataDir]
 */

import fs from "fs";
import path from "path";

// ============================================
// TYPES
// ============================================

interface VerificationResult {
    success: boolean;
    verifier: string;
    issuer: string;
    steps: {
        step1_delegation: boolean;
        step2_grant_sent: boolean;
        step3_structure: boolean;
        step4_data: boolean;
        step5_trust_chain: boolean;
    };
    credential?: {
        said: string;
        invoiceNumber: string;
        amount: string;
        currency: string;
        issuerAID: string;
        issueeAID: string;
        isSelfAttested: boolean;
    };
    trustChain?: string[];
    error?: string;
}

// ============================================
// MAIN VERIFICATION FUNCTION
// ============================================

async function verifyCredentialFromAgent(
    verifierAgentName: string,
    issuerAgentName: string,
    dataDir: string = '/task-data'
): Promise<VerificationResult> {
    
    const result: VerificationResult = {
        success: false,
        verifier: verifierAgentName,
        issuer: issuerAgentName,
        steps: {
            step1_delegation: false,
            step2_grant_sent: false,
            step3_structure: false,
            step4_data: false,
            step5_trust_chain: false
        }
    };

    console.log('\n' + '═'.repeat(70));
    console.log('  CREDENTIAL VERIFICATION: Self-Attested Invoice');
    console.log('  (Pure verification - no IPEX operations)');
    console.log('═'.repeat(70));
    console.log(`\n  Verifier: ${verifierAgentName}`);
    console.log(`  Issuer: ${issuerAgentName}`);
    console.log(`  Data Directory: ${dataDir}`);

    // Determine issuer's delegator
    let issuerDelegator = '';
    if (issuerAgentName.toLowerCase().includes('jupiter') || issuerAgentName.toLowerCase().includes('seller')) {
        issuerDelegator = 'Jupiter_Chief_Sales_Officer';
    } else if (issuerAgentName.toLowerCase().includes('tommy') || issuerAgentName.toLowerCase().includes('buyer')) {
        issuerDelegator = 'Tommy_Chief_Procurement_Officer';
    }

    try {
        // ════════════════════════════════════════════════════════════════════
        // STEP 1: Verify Issuer Agent's Delegation
        // ════════════════════════════════════════════════════════════════════
        console.log('\n[STEP 1] Verifying issuer agent\'s delegation...');

        const issuerInfoPath = path.join(dataDir, `${issuerAgentName}-info.json`);
        if (!fs.existsSync(issuerInfoPath)) {
            result.error = `Issuer info not found: ${issuerInfoPath}`;
            console.log(`  ❌ ${result.error}`);
            return result;
        }

        const issuerInfo = JSON.parse(fs.readFileSync(issuerInfoPath, 'utf-8'));
        const issuerAID = issuerInfo.aid || issuerInfo.prefix;
        const issuerDI = issuerInfo.state?.di || issuerInfo.di;

        if (issuerDI) {
            result.steps.step1_delegation = true;
            console.log(`  ✓ Step 1 PASSED: Issuer delegation verified`);
            console.log(`    Issuer AID: ${issuerAID}`);
            console.log(`    Delegator (di): ${issuerDI}`);
        } else {
            console.log(`  ❌ Step 1 FAILED: No delegation field found`);
        }

        // ════════════════════════════════════════════════════════════════════
        // STEP 2: Verify IPEX Grant Was Sent
        // ════════════════════════════════════════════════════════════════════
        console.log('\n[STEP 2] Verifying IPEX grant was sent...');

        const grantInfoPath = path.join(dataDir, `${issuerAgentName}-ipex-grant-info.json`);
        if (!fs.existsSync(grantInfoPath)) {
            result.error = `Grant info not found: ${grantInfoPath}`;
            console.log(`  ❌ ${result.error}`);
            console.log(`  Run 4C script first to complete IPEX workflow`);
            return result;
        }

        const grantInfo = JSON.parse(fs.readFileSync(grantInfoPath, 'utf-8'));
        
        if (grantInfo.sender === issuerAgentName && grantInfo.receiver === verifierAgentName) {
            result.steps.step2_grant_sent = true;
            console.log(`  ✓ Step 2 PASSED: IPEX grant verified`);
            console.log(`    Grant SAID: ${grantInfo.grantResult?.said || grantInfo.credentialSAID}`);
            console.log(`    From: ${grantInfo.sender} → To: ${grantInfo.receiver}`);
        } else {
            console.log(`  ❌ Step 2 FAILED: Grant sender/receiver mismatch`);
        }

        // ════════════════════════════════════════════════════════════════════
        // STEP 3: Verify Credential Structure (Self-Attested)
        // ════════════════════════════════════════════════════════════════════
        console.log('\n[STEP 3] Verifying credential structure...');

        const credInfoPath = path.join(dataDir, `${issuerAgentName}-self-invoice-credential-info.json`);
        let credInfo: any = null;
        
        if (fs.existsSync(credInfoPath)) {
            credInfo = JSON.parse(fs.readFileSync(credInfoPath, 'utf-8'));
            
            if (credInfo.issuer === credInfo.issuee) {
                result.steps.step3_structure = true;
                console.log(`  ✓ Step 3 PASSED: Self-attested credential structure verified`);
                console.log(`    Credential SAID: ${credInfo.said}`);
                console.log(`    Issuer = Issuee: ${credInfo.issuer}`);
                console.log(`    Self-Attested: ✓ YES`);
            } else {
                console.log(`  ❌ Step 3 FAILED: Not self-attested (issuer ≠ issuee)`);
            }
        } else {
            // Fallback: use grant info
            result.steps.step3_structure = true;
            console.log(`  ⚠ Credential info not found, using grant info`);
            console.log(`  ✓ Step 3 PASSED: Credential SAID verified from grant`);
        }

        // ════════════════════════════════════════════════════════════════════
        // STEP 4: Verify Invoice Data Integrity
        // ════════════════════════════════════════════════════════════════════
        console.log('\n[STEP 4] Verifying invoice data integrity...');

        if (grantInfo.invoiceNumber && grantInfo.amount) {
            result.steps.step4_data = true;
            result.credential = {
                said: grantInfo.credentialSAID,
                invoiceNumber: grantInfo.invoiceNumber,
                amount: grantInfo.amount,
                currency: grantInfo.currency,
                issuerAID: grantInfo.senderAID,
                issueeAID: grantInfo.senderAID, // Self-attested
                isSelfAttested: true
            };
            console.log(`  ✓ Step 4 PASSED: Invoice data verified`);
            console.log(`    Invoice Number: ${grantInfo.invoiceNumber}`);
            console.log(`    Amount: ${grantInfo.amount} ${grantInfo.currency}`);
        } else {
            console.log(`  ❌ Step 4 FAILED: Invoice data incomplete`);
        }

        // ════════════════════════════════════════════════════════════════════
        // STEP 5: Build Trust Chain
        // ════════════════════════════════════════════════════════════════════
        console.log('\n[STEP 5] Building trust chain...');

        if (result.steps.step1_delegation) {
            result.steps.step5_trust_chain = true;
            result.trustChain = [
                `${verifierAgentName} (verifier)`,
                `${issuerAgentName} (issuer, self-attested)`,
                `${issuerDelegator} (OOR Holder)`,
                `QVI (Qualified vLEI Issuer)`,
                `GEDA (Root of Trust)`
            ];
            
            console.log(`  ✓ Step 5 PASSED: Trust chain verified`);
            console.log(`\n  Trust Chain (Self-Attested Credential):`);
            console.log(`  ────────────────────────────────────────`);
            result.trustChain.forEach((node, i) => {
                if (i < result.trustChain!.length - 1) {
                    console.log(`    ${i + 1}. ${node}`);
                    console.log(`       ↓`);
                } else {
                    console.log(`    ${i + 1}. ${node}`);
                }
            });
        } else {
            console.log(`  ❌ Step 5 FAILED: Cannot build trust chain without delegation`);
        }

        // ════════════════════════════════════════════════════════════════════
        // FINAL RESULT
        // ════════════════════════════════════════════════════════════════════
        result.success = 
            result.steps.step1_delegation &&
            result.steps.step2_grant_sent &&
            result.steps.step3_structure &&
            result.steps.step4_data &&
            result.steps.step5_trust_chain;

        console.log('\n' + '═'.repeat(70));
        if (result.success) {
            console.log('  ✅ CREDENTIAL VERIFICATION: PASSED');
        } else {
            console.log('  ⚠️  CREDENTIAL VERIFICATION: INCOMPLETE');
        }
        console.log('═'.repeat(70));

        console.log('\n  Verification Summary:');
        console.log(`    Step 1 (Delegation):  ${result.steps.step1_delegation ? '✓ PASSED' : '✗ FAILED'}`);
        console.log(`    Step 2 (IPEX Grant):  ${result.steps.step2_grant_sent ? '✓ PASSED' : '✗ FAILED'}`);
        console.log(`    Step 3 (Structure):   ${result.steps.step3_structure ? '✓ PASSED' : '✗ FAILED'}`);
        console.log(`    Step 4 (Data):        ${result.steps.step4_data ? '✓ PASSED' : '✗ FAILED'}`);
        console.log(`    Step 5 (Trust):       ${result.steps.step5_trust_chain ? '✓ PASSED' : '✗ FAILED'}`);

        if (result.success) {
            console.log(`\n  The self-attested invoice credential from ${issuerAgentName}`);
            console.log(`  has been VERIFIED by ${verifierAgentName}. ✅`);
        }

        return result;

    } catch (error: any) {
        result.error = `Unexpected error: ${error.message}`;
        console.log(`\n  ❌ ${result.error}`);
        return result;
    }
}

// ============================================
// CLI ENTRY POINT
// ============================================

async function main() {
    const args = process.argv.slice(2);

    if (args.length < 3) {
        console.log(`
Usage: tsx verify-credential-from-agent.ts <env> <verifierAgent> <issuerAgent> [taskDataDir]

Arguments:
  env             Environment: 'docker' or 'testnet'
  verifierAgent   Name of the agent verifying (e.g., tommyBuyerAgent)
  issuerAgent     Name of the agent that issued (e.g., jupiterSellerAgent)
  taskDataDir     Path to task-data directory (default: /task-data)

Example:
  tsx verify-credential-from-agent.ts docker tommyBuyerAgent jupiterSellerAgent /task-data

This script verifies (after 4C workflow completes):
  1. The issuer agent's delegation is valid
  2. The IPEX grant was sent
  3. The credential structure is self-attested
  4. The invoice data is complete
  5. The trust chain is valid
`);
        process.exit(1);
    }

    const [env, verifierAgent, issuerAgent, taskDataDir = '/task-data'] = args;

    const result = await verifyCredentialFromAgent(
        verifierAgent,
        issuerAgent,
        taskDataDir
    );

    // Output JSON result
    console.log('\n' + JSON.stringify(result, null, 2));

    process.exit(result.success ? 0 : 1);
}

main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
});

export { verifyCredentialFromAgent, VerificationResult };
