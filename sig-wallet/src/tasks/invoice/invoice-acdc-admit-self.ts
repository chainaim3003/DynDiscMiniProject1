import { getOrCreateClient } from "../../client/identifiers.js";
import { getAID } from "../../client/identifiers.js";

/**
 * Admit self-attested invoice credential
 * This completes the self-attestation process where the agent admits its own credential
 * 
 * Usage: tsx invoice-acdc-admit-self.ts <env> <passcode> <agentName>
 */

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1];
const agentName = args[2];

console.log(`========================================`);
console.log(`ADMIT SELF-ATTESTED INVOICE`);
console.log(`========================================`);
console.log(``);
console.log(`Agent: ${agentName}`);
console.log(``);

try {
    // Get client
    console.log(`[1/4] Connecting to KERIA agent...`);
    const client = await getOrCreateClient(passcode, env);
    const agentAID = await getAID(client, agentName);
    console.log(`✓ Connected`);
    console.log(`  Agent AID: ${agentAID.prefix}`);
    console.log(``);
    
    // Get pending credentials (notifications)
    console.log(`[2/4] Checking for pending self-attested credentials...`);
    const notifications = await client.notifications().list();
    
    const pendingGrants = notifications.notes.filter((n: any) => 
        n.a?.r === '/ipex/grant' || n.a?.r === 'ipex/grant'
    );
    
    if (pendingGrants.length === 0) {
        console.log(`⚠ Warning: No pending grant notifications found`);
        console.log(`  This might mean the credential was already admitted`);
        console.log(`  or the grant hasn't been sent yet.`);
        console.log(``);
        
        // Check if credential already exists
        const credentials = await client.credentials().list(agentAID.name);
        const invoiceCred = credentials.find((c: any) => {
            const schema = c.sad?.s || '';
            return schema.toLowerCase().includes('invoice');
        });
        
        if (invoiceCred) {
            console.log(`✓ Self-attested invoice credential already exists: ${invoiceCred.sad.d}`);
            console.log(`  The credential was likely already admitted.`);
            process.exit(0);
        } else {
            throw new Error(`No pending grant found and no invoice credential exists`);
        }
    }
    
    console.log(`✓ Found ${pendingGrants.length} pending grant(s)`);
    console.log(``);
    
    // Admit the first invoice grant (should be the self-attested one)
    let admittedCount = 0;
    for (const grant of pendingGrants) {
        console.log(`[3/4] Processing grant notification...`);
        console.log(`  Notification SAID: ${grant.i}`);
        
        // Get the credential SAID from the grant
        const credSAID = grant.a?.d;
        if (!credSAID) {
            console.log(`  ⚠ Skipping grant without credential SAID`);
            continue;
        }
        
        console.log(`  Credential SAID: ${credSAID}`);
        console.log(``);
        
        // Admit the credential
        console.log(`[4/4] Admitting self-attested credential...`);
        
        try {
            // Admit the grant
            await client.ipex().admit(
                agentAID.name,
                '',  // message SAID (will be generated)
                grant.i,  // grant SAID from notification
                new Date().toISOString()
            );
            
            console.log(`✓ Self-attested credential admitted successfully`);
            console.log(`  Credential SAID: ${credSAID}`);
            admittedCount++;
            
            // Mark notification as read
            await client.notifications().mark(grant.i);
            console.log(`✓ Notification marked as read`);
            console.log(``);
            
            // Only admit the first one (should be the invoice)
            break;
            
        } catch (admitError: any) {
            console.log(`⚠ Failed to admit credential: ${admitError.message}`);
            console.log(`  This might be expected if already admitted`);
            console.log(``);
            continue;
        }
    }
    
    if (admittedCount === 0) {
        throw new Error(`Failed to admit any credentials`);
    }
    
    // Verify the credential is now in storage
    console.log(`Verifying credential storage...`);
    const credentials = await client.credentials().list(agentAID.name);
    const invoiceCred = credentials.find((c: any) => {
        const schema = c.sad?.s || '';
        return schema.toLowerCase().includes('invoice');
    });
    
    if (invoiceCred) {
        console.log(`✓ Invoice credential confirmed in KERIA storage`);
        console.log(`  SAID: ${invoiceCred.sad.d}`);
        console.log(`  Issuer: ${invoiceCred.sad.i}`);
        console.log(`  Schema: ${invoiceCred.sad.s}`);
        
        // Verify self-attestation
        const issuer = invoiceCred.sad.i;
        const holder = invoiceCred.sad.a?.i || agentAID.prefix;
        if (issuer === holder) {
            console.log(`  ✓ Confirmed self-attested (issuer = holder)`);
        } else {
            console.log(`  ⚠ Warning: Not self-attested (issuer ≠ holder)`);
        }
        console.log(``);
    } else {
        throw new Error(`Invoice credential not found in KERIA after admit`);
    }
    
    console.log(`========================================`);
    console.log(`✅ SELF-ATTESTED ADMISSION COMPLETED`);
    console.log(`========================================`);
    console.log(``);
    console.log(`Summary:`);
    console.log(`  ✓ Self-attested invoice admitted`);
    console.log(`  ✓ Credential stored in ${agentName}'s KERIA`);
    console.log(`  ✓ Ready to be granted to other agents via IPEX`);
    console.log(``);
    
} catch (error: any) {
    console.error(`❌ Admit failed: ${error.message}`);
    console.error(error);
    process.exit(1);
}
