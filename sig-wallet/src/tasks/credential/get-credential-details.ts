import { getOrCreateClient } from "../../client/identifiers.js";
import { getAID } from "../../client/identifiers.js";

/**
 * Get credential details and output as JSON
 * 
 * Usage: tsx get-credential-details.ts <env> <passcode> <agentName> <credentialType>
 */

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1];
const agentName = args[2];
const credentialType = args[3];

try {
    // Get client
    const client = await getOrCreateClient(passcode, env);
    const agentAID = await getAID(client, agentName);
    
    // Retrieve credential
    const credentials = await client.credentials().list(agentAID.name);
    const cred = credentials.find((c: any) => {
        const schema = c.sad?.s || '';
        return schema.toLowerCase().includes(credentialType.toLowerCase());
    });
    
    if (!cred) {
        throw new Error(`No ${credentialType} credential found`);
    }
    
    // Build credential details object
    const details: any = {
        said: cred.sad.d,
        schema: cred.sad.s,
        issuer: cred.sad.i,
        status: cred.status || {},
        attributes: {}
    };
    
    // Extract attributes based on credential type
    if (credentialType === 'invoice' && cred.sad.a) {
        const attrs = cred.sad.a;
        details.attributes = {
            invoiceNumber: attrs.invoiceNumber,
            invoiceDate: attrs.invoiceDate,
            dueDate: attrs.dueDate,
            totalAmount: attrs.totalAmount,
            currency: attrs.currency,
            sellerLegalName: attrs.sellerLegalName,
            sellerLEI: attrs.sellerLEI,
            buyerLegalName: attrs.buyerLegalName,
            buyerLEI: attrs.buyerLEI,
            paymentMethod: attrs.paymentMethod,
            paymentChainID: attrs.paymentChainID,
            paymentWalletAddress: attrs.paymentWalletAddress,
            description: attrs.description
        };
        
        // Add holder/issuee
        details.issuee = attrs.i || attrs.id;
        
        // Determine if self-attested
        details.selfAttested = (details.issuer === details.issuee);
    }
    
    // Add edges if present
    if (cred.sad.e) {
        details.edges = cred.sad.e;
    }
    
    // Output as JSON
    console.log(JSON.stringify(details, null, 2));
    
} catch (error: any) {
    const errorOutput = {
        error: error.message,
        agent: agentName,
        credentialType: credentialType
    };
    console.error(JSON.stringify(errorOutput, null, 2));
    process.exit(1);
}
