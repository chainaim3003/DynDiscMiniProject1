import { getOrCreateClient } from "../../client/identifiers.js";
import { getAID } from "../../client/identifiers.js";

/**
 * Verify self-attestation (Issuer = Issuee)
 * 
 * Usage: tsx verify-self-attestation.ts <env> <passcode> <agentName> <credentialType>
 */

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1];
const agentName = args[2];
const credentialType = args[3];

console.log(`========================================`);
console.log(`VERIFY SELF-ATTESTATION`);
console.log(`========================================`);
console.log(``);
console.log(`Agent: ${agentName}`);
console.log(`Credential Type: ${credentialType}`);
console.log(``);

try {
    // Get client
    console.log(`[1/3] Connecting to KERIA agent...`);
    const client = await getOrCreateClient(passcode, env);
    const agentAID = await getAID(client, agentName);
    console.log(`✓ Connected`);
    console.log(`  Agent AID: ${agentAID.prefix}`);
    console.log(``);
    
    // Retrieve credential
    console.log(`[2/3] Retrieving credential...`);
    const credentials = await client.credentials().list(agentAID.name);
    const cred = credentials.find((c: any) => {
        const schema = c.sad?.s || '';
        return schema.toLowerCase().includes(credentialType.toLowerCase());
    });
    
    if (!cred) {
        throw new Error(`No ${credentialType} credential found`);
    }
    
    console.log(`✓ Credential retrieved: ${cred.sad.d}`);
    console.log(``);
    
    // Verify self-attestation
    console.log(`[3/3] Verifying self-attestation...`);
    
    const issuer = cred.sad.i;
    const issuee = cred.sad.a?.i || cred.sad.a?.id;
    
    console.log(`  Issuer: ${issuer}`);
    console.log(`  Issuee: ${issuee}`);
    console.log(`  Agent AID: ${agentAID.prefix}`);
    console.log(``);
    
    // Check 1: Issuer must equal Issuee
    if (issuer !== issuee) {
        throw new Error(
            `NOT self-attested: Issuer (${issuer}) ≠ Issuee (${issuee})`
        );
    }
    console.log(`✓ Check 1: Issuer = Issuee`);
    
    // Check 2: Issuer must equal Agent AID
    if (issuer !== agentAID.prefix) {
        throw new Error(
            `Issuer (${issuer}) does not match agent AID (${agentAID.prefix})`
        );
    }
    console.log(`✓ Check 2: Issuer = Agent AID`);
    
    // Check 3: No edges to OOR credentials (self-attested means no chain)
    if (cred.sad.e && cred.sad.e.oor) {
        console.log(`⚠ Warning: Self-attested credential has OOR edge`);
        console.log(`  This is unusual for pure self-attestation`);
    } else {
        console.log(`✓ Check 3: No OOR edges (pure self-attestation)`);
    }
    console.log(``);
    
    // Display credential details
    if (credentialType === 'invoice' && cred.sad.a) {
        console.log(`Invoice Details:`);
        console.log(`  Number: ${cred.sad.a.invoiceNumber || 'N/A'}`);
        console.log(`  Amount: ${cred.sad.a.totalAmount || 'N/A'} ${cred.sad.a.currency || ''}`);
        console.log(`  Self-attested by: ${agentName}`);
        console.log(``);
    }
    
    console.log(`========================================`);
    console.log(`✅ SELF-ATTESTATION VERIFIED`);
    console.log(`========================================`);
    console.log(``);
    console.log(`Verification Summary:`);
    console.log(`  ✓ Issuer = Issuee: ${issuer}`);
    console.log(`  ✓ Matches agent AID: ${agentAID.prefix}`);
    console.log(`  ✓ Self-attestation confirmed`);
    console.log(``);
    
} catch (error: any) {
    console.error(`❌ Self-attestation verification failed: ${error.message}`);
    process.exit(1);
}
