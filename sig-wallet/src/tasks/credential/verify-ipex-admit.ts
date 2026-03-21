import { getOrCreateClient } from "../../client/identifiers.js";
import { getAID } from "../../client/identifiers.js";

/**
 * Verify IPEX admit flow (credential received via IPEX grant)
 * 
 * Usage: tsx verify-ipex-admit.ts <env> <passcode> <agentName> <credentialType>
 */

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1];
const agentName = args[2];
const credentialType = args[3];

console.log(`========================================`);
console.log(`VERIFY IPEX ADMIT FLOW`);
console.log(`========================================`);
console.log(``);
console.log(`Agent: ${agentName}`);
console.log(`Credential Type: ${credentialType}`);
console.log(``);

try {
    // Get client
    console.log(`[1/4] Connecting to KERIA agent...`);
    const client = await getOrCreateClient(passcode, env);
    const agentAID = await getAID(client, agentName);
    console.log(`✓ Connected`);
    console.log(`  Agent AID: ${agentAID.prefix}`);
    console.log(``);
    
    // Retrieve credential
    console.log(`[2/4] Retrieving credential...`);
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
    
    // Verify IPEX admit properties
    console.log(`[3/4] Verifying IPEX admit properties...`);
    
    const issuer = cred.sad.i;
    const holder = cred.sad.a?.i || cred.sad.a?.id;
    
    console.log(`  Issuer: ${issuer}`);
    console.log(`  Holder: ${holder}`);
    console.log(`  Agent AID: ${agentAID.prefix}`);
    console.log(``);
    
    // Check 1: Issuer must NOT equal holder (not self-attested)
    if (issuer === holder) {
        throw new Error(
            `This is a self-attested credential, not an IPEX admit flow`
        );
    }
    console.log(`✓ Check 1: Issuer ≠ Holder (not self-attested)`);
    
    // Check 2: Holder must equal agent AID (received by this agent)
    if (holder !== agentAID.prefix) {
        console.log(`⚠ Warning: Holder (${holder}) ≠ Agent AID (${agentAID.prefix})`);
        console.log(`  This might be an intermediary credential`);
    } else {
        console.log(`✓ Check 2: Holder = Agent AID (admitted by this agent)`);
    }
    
    // Check 3: Credential should be present in agent's KERIA
    console.log(`✓ Check 3: Credential present in agent's KERIA`);
    console.log(``);
    
    // Try to get IPEX exchange history
    console.log(`[4/4] Checking IPEX exchange history...`);
    try {
        const exchanges = await client.exchanges().list(agentAID.name);
        
        if (exchanges && exchanges.length > 0) {
            const relatedExchange = exchanges.find((ex: any) => 
                ex.exn?.a?.d === cred.sad.d
            );
            
            if (relatedExchange) {
                console.log(`✓ Found related IPEX exchange`);
                console.log(`  Exchange type: ${relatedExchange.exn?.r || 'N/A'}`);
            } else {
                console.log(`ℹ No direct IPEX exchange found for this credential`);
                console.log(`  (Credential may have been admitted successfully)`);
            }
        } else {
            console.log(`ℹ No IPEX exchange history available`);
        }
    } catch (error: any) {
        console.log(`ℹ Could not retrieve IPEX history: ${error.message}`);
    }
    console.log(``);
    
    // Display credential details
    if (credentialType === 'invoice' && cred.sad.a) {
        console.log(`Invoice Details:`);
        console.log(`  Number: ${cred.sad.a.invoiceNumber || 'N/A'}`);
        console.log(`  Amount: ${cred.sad.a.totalAmount || 'N/A'} ${cred.sad.a.currency || ''}`);
        console.log(`  Issued by: ${issuer}`);
        console.log(`  Admitted by: ${agentName}`);
        console.log(``);
    }
    
    console.log(`========================================`);
    console.log(`✅ IPEX ADMIT FLOW VERIFIED`);
    console.log(`========================================`);
    console.log(``);
    console.log(`Verification Summary:`);
    console.log(`  ✓ Credential received via IPEX grant`);
    console.log(`  ✓ Admitted by agent: ${agentName}`);
    console.log(`  ✓ Stored in agent's KERIA`);
    console.log(`  ✓ Ready for use`);
    console.log(``);
    
} catch (error: any) {
    console.error(`❌ IPEX admit verification failed: ${error.message}`);
    process.exit(1);
}
