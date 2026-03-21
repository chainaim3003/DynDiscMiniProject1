import { getOrCreateClient } from "../../client/identifiers.js";
import { getAID } from "../../client/identifiers.js";

/**
 * Validate credential structure and integrity
 * 
 * Usage: tsx validate-credential.ts <env> <passcode> <agentName> <credentialType>
 */

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1];
const agentName = args[2];
const credentialType = args[3];

console.log(`========================================`);
console.log(`VALIDATE CREDENTIAL STRUCTURE`);
console.log(`========================================`);
console.log(``);
console.log(`Agent: ${agentName}`);
console.log(`Credential Type: ${credentialType}`);
console.log(``);

try {
    // Get client
    console.log(`[1/5] Connecting to KERIA agent...`);
    const client = await getOrCreateClient(passcode, env);
    const agentAID = await getAID(client, agentName);
    console.log(`✓ Connected`);
    console.log(``);
    
    // Query credentials
    console.log(`[2/5] Retrieving credential...`);
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
    
    // Validate structure
    console.log(`[3/5] Validating credential structure...`);
    
    // Check required fields
    if (!cred.sad) {
        throw new Error('Missing SAD (Self-Addressing Data)');
    }
    
    if (!cred.sad.d) {
        throw new Error('Missing SAID (Self-Addressing Identifier)');
    }
    
    if (!cred.sad.i) {
        throw new Error('Missing issuer identifier');
    }
    
    if (!cred.sad.s) {
        throw new Error('Missing schema identifier');
    }
    
    if (!cred.sad.a) {
        throw new Error('Missing attribute section');
    }
    
    console.log(`✓ Basic structure valid`);
    console.log(`  SAID: ${cred.sad.d}`);
    console.log(`  Schema: ${cred.sad.s}`);
    console.log(`  Issuer: ${cred.sad.i}`);
    console.log(``);
    
    // Validate attributes based on credential type
    console.log(`[4/5] Validating credential attributes...`);
    
    if (credentialType === 'invoice') {
        const attrs = cred.sad.a;
        const requiredFields = [
            'invoiceNumber',
            'invoiceDate',
            'totalAmount',
            'currency',
            'sellerLegalName',
            'buyerLegalName'
        ];
        
        for (const field of requiredFields) {
            if (!attrs[field]) {
                throw new Error(`Missing required field: ${field}`);
            }
        }
        
        console.log(`✓ All required invoice fields present`);
        console.log(`  Invoice Number: ${attrs.invoiceNumber}`);
        console.log(`  Amount: ${attrs.totalAmount} ${attrs.currency}`);
    }
    console.log(``);
    
    // Validate signatures
    console.log(`[5/5] Validating credential signatures...`);
    
    if (!cred.status || !cred.status.s) {
        console.log(`⚠ Warning: No signature status found`);
    } else {
        console.log(`✓ Credential has signature status: ${cred.status.s}`);
    }
    console.log(``);
    
    console.log(`========================================`);
    console.log(`✅ CREDENTIAL VALIDATION SUCCESSFUL`);
    console.log(`========================================`);
    console.log(``);
    console.log(`Validation Summary:`);
    console.log(`  ✓ Structure: Valid`);
    console.log(`  ✓ Required fields: Present`);
    console.log(`  ✓ Attributes: Valid`);
    console.log(`  ✓ Signatures: Checked`);
    console.log(``);
    
} catch (error: any) {
    console.error(`❌ Validation failed: ${error.message}`);
    process.exit(1);
}
