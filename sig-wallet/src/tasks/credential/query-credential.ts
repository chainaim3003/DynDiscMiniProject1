import { getOrCreateClient } from "../../client/identifiers.js";
import { getAID } from "../../client/identifiers.js";

/**
 * Query credential from KERIA agent
 * 
 * Usage: tsx query-credential.ts <env> <passcode> <agentName> <credentialType>
 * 
 * Arguments:
 *   env: 'docker' | 'testnet'
 *   passcode: Agent's passcode
 *   agentName: Name of the agent (e.g., 'jupiterSalesAgent')
 *   credentialType: Type of credential to query (e.g., 'invoice')
 */

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1];
const agentName = args[2];
const credentialType = args[3];

console.log(`========================================`);
console.log(`QUERY CREDENTIAL FROM KERIA`);
console.log(`========================================`);
console.log(``);
console.log(`Agent: ${agentName}`);
console.log(`Credential Type: ${credentialType}`);
console.log(`Environment: ${env}`);
console.log(``);

try {
    // Get client
    console.log(`[1/3] Connecting to KERIA agent...`);
    const client = await getOrCreateClient(passcode, env);
    console.log(`✓ Connected to KERIA`);
    console.log(``);
    
    // Get agent AID
    console.log(`[2/3] Retrieving agent AID...`);
    const agentAID = await getAID(client, agentName);
    console.log(`✓ Agent AID retrieved: ${agentAID.prefix}`);
    console.log(``);
    
    // Query credentials
    console.log(`[3/3] Querying ${credentialType} credentials...`);
    const credentials = await client.credentials().list(agentAID.name);
    
    if (!credentials || credentials.length === 0) {
        throw new Error(`No credentials found for agent ${agentName}`);
    }
    
    // Filter for the requested credential type
    const filteredCreds = credentials.filter((cred: any) => {
        const schema = cred.sad?.s || '';
        return schema.toLowerCase().includes(credentialType.toLowerCase());
    });
    
    if (filteredCreds.length === 0) {
        throw new Error(`No ${credentialType} credentials found for agent ${agentName}`);
    }
    
    console.log(`✓ Found ${filteredCreds.length} ${credentialType} credential(s)`);
    console.log(``);
    
    // Display credential info
    for (let i = 0; i < filteredCreds.length; i++) {
        const cred = filteredCreds[i];
        console.log(`Credential ${i + 1}:`);
        console.log(`  SAID: ${cred.sad.d}`);
        console.log(`  Schema: ${cred.sad.s}`);
        console.log(`  Issuer: ${cred.sad.i}`);
        
        // Display key attributes based on credential type
        if (credentialType === 'invoice' && cred.sad.a) {
            console.log(`  Invoice Number: ${cred.sad.a.invoiceNumber || 'N/A'}`);
            console.log(`  Amount: ${cred.sad.a.totalAmount || 'N/A'} ${cred.sad.a.currency || ''}`);
            console.log(`  Holder: ${cred.sad.a.i || 'N/A'}`);
        }
        console.log(``);
    }
    
    console.log(`========================================`);
    console.log(`✅ CREDENTIAL QUERY SUCCESSFUL`);
    console.log(`========================================`);
    console.log(``);
    
} catch (error: any) {
    console.error(`❌ Query failed: ${error.message}`);
    process.exit(1);
}
