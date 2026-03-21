import fs from "fs";
import { SignifyClient, Serder } from "signify-ts";
import { getOrCreateClient } from "../../client/identifiers.js";
import { findMatchingCredentials, grantCredential } from "../../client/credentials.js";
import { resolveOobi } from "../../client/oobis.js";
import { createTimestamp } from "../../time.js";

/**
 * Send IPEX grant for self-attested invoice credential
 * 
 * FIXED VERSION: Ensures proper OOBI resolution before grant
 * 
 * Key fix: Resolve receiver's OOBI AND witness OOBIs to enable
 * proper message routing through KERIA's messaging layer.
 * 
 * Usage: tsx invoice-ipex-grant.ts <env> <passcode> <senderAgent> <receiverAgent> [taskDataDir]
 */

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1];
const senderAgentName = args[2];
const receiverAgentName = args[3];
const taskDataDir = args[4] || '/task-data';

console.log(`========================================`);
console.log(`IPEX GRANT: Invoice Credential (FIXED)`);
console.log(`========================================`);
console.log(``);
console.log(`Sender: ${senderAgentName}`);
console.log(`Receiver: ${receiverAgentName}`);
console.log(`Task data dir: ${taskDataDir}`);
console.log(``);

// Determine the actual passcode to use for sender
let senderPasscode = passcode;

// If passcode is empty or not provided, try to read from agent's bran file
if (!passcode || passcode.trim() === '') {
    const branFilePath = `${taskDataDir}/${senderAgentName}-bran.txt`;
    console.log(`No passcode provided, checking for BRAN file: ${branFilePath}`);
    
    if (fs.existsSync(branFilePath)) {
        senderPasscode = fs.readFileSync(branFilePath, 'utf-8').trim();
        console.log(`✓ Found sender's unique BRAN: ${senderPasscode.substring(0, 20)}...`);
    } else {
        console.error(`✗ ERROR: No BRAN file found at ${branFilePath}`);
        console.error(`The agent must have been created with a unique BRAN first.`);
        process.exit(1);
    }
}

try {
    // Get sender's client
    console.log(`[1/7] Connecting to sender KERIA agent...`);
    const senderClient = await getOrCreateClient(senderPasscode, env);
    const senderAID = await senderClient.identifiers().get(senderAgentName);
    console.log(`✓ Connected as sender`);
    console.log(`  Sender AID: ${senderAID.prefix}`);
    console.log(``);
    
    // Read receiver info from file
    console.log(`[2/7] Getting receiver information...`);
    const receiverInfoPath = `${taskDataDir}/${receiverAgentName}-info.json`;
    
    if (!fs.existsSync(receiverInfoPath)) {
        throw new Error(`Receiver info file not found: ${receiverInfoPath}`);
    }
    
    const receiverInfo = JSON.parse(fs.readFileSync(receiverInfoPath, 'utf-8'));
    const receiverPrefix = receiverInfo.aid;
    const receiverOobi = receiverInfo.oobi;
    
    console.log(`✓ Receiver info loaded`);
    console.log(`  Receiver AID: ${receiverPrefix}`);
    console.log(`  Receiver OOBI: ${receiverOobi}`);
    console.log(``);
    
    // FIXED: Resolve receiver's OOBI BEFORE doing anything else
    console.log(`[3/7] Resolving receiver's OOBI (CRITICAL for message routing)...`);
    try {
        await resolveOobi(senderClient, receiverOobi, receiverAgentName);
        console.log(`✓ Receiver OOBI resolved`);
    } catch (e: any) {
        console.log(`  Note: ${e.message} (may already be resolved)`);
    }
    
    // FIXED: Query receiver's key state to ensure we have their current keys
    console.log(`[4/7] Querying receiver's key state...`);
    try {
        const receiverState = await senderClient.keyStates().query(receiverPrefix);
        console.log(`✓ Receiver key state verified`);
        console.log(`  Sequence: ${receiverState?.s || 'N/A'}`);
    } catch (e: any) {
        console.log(`  Note: Key state query: ${e.message}`);
    }
    console.log(``);
    
    // Get the self-attested invoice credential
    console.log(`[5/7] Retrieving self-attested invoice credential...`);
    
    // Find the invoice credential
    const credentials = await senderClient.credentials().list({ filter: {} });
    console.log(`  Found ${credentials.length} credential(s) in sender's KERIA`);
    
    // Find the most recent invoice credential
    let invoiceCred: any = null;
    for (const cred of credentials) {
        console.log(`  - Credential: ${cred.sad?.d}, Schema: ${cred.sad?.s}`);
        // Check if it's an invoice credential (by schema or content)
        if (cred.sad?.a?.invoiceNumber || cred.sad?.a?.totalAmount) {
            invoiceCred = cred;
            break;
        }
    }
    
    // Also check for any credential that was self-issued
    if (!invoiceCred) {
        const credInfoPath = `${taskDataDir}/${senderAgentName}-self-invoice-credential-info.json`;
        if (fs.existsSync(credInfoPath)) {
            const credInfo = JSON.parse(fs.readFileSync(credInfoPath, 'utf-8'));
            const matchingCreds = await findMatchingCredentials(senderClient, { '-d': credInfo.said });
            if (matchingCreds.length > 0) {
                invoiceCred = matchingCreds[0];
            }
        }
    }
    
    if (!invoiceCred) {
        throw new Error(`No invoice credential found for ${senderAgentName}. Please issue a credential first.`);
    }
    
    console.log(`✓ Invoice credential found: ${invoiceCred.sad.d}`);
    console.log(`  Invoice Number: ${invoiceCred.sad.a?.invoiceNumber || 'N/A'}`);
    console.log(`  Amount: ${invoiceCred.sad.a?.totalAmount || 'N/A'} ${invoiceCred.sad.a?.currency || ''}`);
    console.log(``);
    
    // Verify this is a self-attested credential
    console.log(`[6/7] Verifying self-attestation...`);
    const issuer = invoiceCred.sad.i;
    const holder = invoiceCred.sad.a?.i;
    
    if (issuer !== holder) {
        console.log(`⚠ Warning: This credential may not be self-attested`);
        console.log(`  Issuer: ${issuer}`);
        console.log(`  Holder: ${holder}`);
    } else {
        console.log(`✓ Confirmed self-attested credential`);
        console.log(`  Issuer = Holder: ${issuer}`);
    }
    console.log(``);
    
    // Send IPEX grant
    console.log(`[7/7] Sending IPEX grant to ${receiverAgentName}...`);
    
    const grantTime = createTimestamp();
    const [grant, gsigs, gend] = await senderClient.ipex().grant({
        senderName: senderAgentName,
        acdc: new Serder(invoiceCred.sad),
        anc: new Serder(invoiceCred.anc),
        iss: new Serder(invoiceCred.iss),
        ancAttachment: invoiceCred.ancatc,
        recipient: receiverPrefix,
        datetime: grantTime,
    });

    const grantOp = await senderClient.ipex().submitGrant(
        senderAgentName, 
        grant, 
        gsigs, 
        gend, 
        [receiverPrefix]
    );
    
    // Wait for operation to complete
    const completedOp = await senderClient.operations().wait(grantOp, {
        signal: AbortSignal.timeout(30000)
    });
    
    // FIXED: Get grant SAID from the operation result, not from grant object
    // The operation result contains: { name: 'exchange.SAID', metadata: { said: 'SAID' }, response: { said: 'SAID' } }
    const grantSaid = completedOp?.response?.said || completedOp?.metadata?.said || grantOp?.metadata?.said || 'unknown';
    
    console.log(`✓ IPEX grant sent successfully`);
    console.log(`  Grant SAID: ${grantSaid}`);
    console.log(`  Operation: ${completedOp?.name || grantOp?.name}`);
    console.log(``);
    
    // Save grant info with the SAID
    const grantInfo = {
        sender: senderAgentName,
        senderAID: senderAID.prefix,
        receiver: receiverAgentName,
        receiverAID: receiverPrefix,
        credentialSAID: invoiceCred.sad.d,
        invoiceNumber: invoiceCred.sad.a?.invoiceNumber,
        amount: invoiceCred.sad.a?.totalAmount,
        currency: invoiceCred.sad.a?.currency,
        timestamp: new Date().toISOString(),
        grantResult: {
            said: grantSaid,
            operation: completedOp?.name || grantOp?.name
        },
        // Include full credential data for cross-client verification
        credential: {
            sad: invoiceCred.sad,
            anc: invoiceCred.anc,
            iss: invoiceCred.iss,
            ancatc: invoiceCred.ancatc
        }
    };
    
    const grantInfoPath = `${taskDataDir}/${senderAgentName}-ipex-grant-info.json`;
    fs.writeFileSync(grantInfoPath, JSON.stringify(grantInfo, null, 2));
    
    console.log(`Grant Information saved to ${grantInfoPath}`);
    console.log(``);
    
    console.log(`========================================`);
    console.log(`✅ IPEX GRANT COMPLETED`);
    console.log(`========================================`);
    console.log(``);
    console.log(`Summary:`);
    console.log(`  ✓ Invoice credential: ${invoiceCred.sad.d}`);
    console.log(`  ✓ Grant SAID: ${grantSaid}`);
    console.log(`  ✓ Granted from: ${senderAgentName}`);
    console.log(`  ✓ Granted to: ${receiverAgentName}`);
    console.log(`  ✓ ${receiverAgentName} can now admit the credential`);
    console.log(``);
    
} catch (error: any) {
    console.error(`❌ IPEX grant failed: ${error.message}`);
    console.error(error);
    process.exit(1);
}
