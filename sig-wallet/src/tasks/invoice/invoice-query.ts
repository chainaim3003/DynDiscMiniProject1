import fs from "fs";
import { getOrCreateClient } from "../../client/identifiers.js";
import { getAID } from "../../client/identifiers.js";

/**
 * Query invoice credentials from KERIA agent
 * 
 * Usage: tsx invoice-query.ts <env> <passcode> <agentName>
 */

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1];
const agentName = args[2];

console.log(`========================================`);
console.log(`QUERY INVOICE CREDENTIALS`);
console.log(`========================================`);
console.log(``);
console.log(`Agent: ${agentName}`);
console.log(``);

try {
    // Get client
    console.log(`[1/3] Connecting to KERIA agent...`);
    const client = await getOrCreateClient(passcode, env);
    const agentAID = await getAID(client, agentName);
    console.log(`✓ Connected`);
    console.log(`  Agent AID: ${agentAID.prefix}`);
    console.log(``);
    
    // Query all credentials
    console.log(`[2/3] Querying credentials from KERIA...`);
    const credentials = await client.credentials().list(agentAID.name);
    
    console.log(`✓ Found ${credentials.length} total credential(s)`);
    console.log(``);
    
    // Filter invoice credentials
    console.log(`[3/3] Filtering invoice credentials...`);
    const invoiceCredentials = credentials.filter((c: any) => {
        const schema = c.sad?.s || '';
        return schema.toLowerCase().includes('invoice');
    });
    
    console.log(`✓ Found ${invoiceCredentials.length} invoice credential(s)`);
    console.log(``);
    
    if (invoiceCredentials.length === 0) {
        console.log(`⚠ No invoice credentials found for ${agentName}`);
        console.log(`  The agent may not have any invoice credentials yet`);
        console.log(``);
        
        // Save empty results
        const queryResults = {
            agent: agentName,
            agentAID: agentAID.prefix,
            invoices: [],
            totalInvoices: 0,
            timestamp: new Date().toISOString()
        };
        
        const outputPath = `./task-data/${agentName}-invoice-query-results.json`;
        await fs.promises.writeFile(outputPath, JSON.stringify(queryResults, null, 2));
        
        process.exit(0);
    }
    
    // Process and display each invoice
    const invoiceResults = [];
    
    for (let i = 0; i < invoiceCredentials.length; i++) {
        const cred = invoiceCredentials[i];
        const said = cred.sad?.d || 'N/A';
        const schema = cred.sad?.s || 'N/A';
        const issuer = cred.sad?.i || 'N/A';
        const attributes = cred.sad?.a || {};
        
        console.log(`Invoice Credential #${i + 1}:`);
        console.log(`  SAID: ${said}`);
        console.log(`  Schema: ${schema}`);
        console.log(`  Issuer: ${issuer}`);
        
        // Extract invoice details
        const invoiceNumber = attributes.invoiceNumber || 'N/A';
        const totalAmount = attributes.totalAmount || 'N/A';
        const currency = attributes.currency || 'N/A';
        const dueDate = attributes.dueDate || 'N/A';
        const sellerLEI = attributes.sellerLEI || 'N/A';
        const buyerLEI = attributes.buyerLEI || 'N/A';
        const paymentChainID = attributes.paymentChainID || 'N/A';
        const paymentWalletAddress = attributes.paymentWalletAddress || 'N/A';
        
        // Check if self-attested
        const holderInAttributes = attributes.i;
        const selfAttested = issuer === holderInAttributes || issuer === agentAID.prefix;
        
        console.log(`  Invoice Number: ${invoiceNumber}`);
        console.log(`  Amount: ${totalAmount} ${currency}`);
        console.log(`  Due Date: ${dueDate}`);
        console.log(`  Seller LEI: ${sellerLEI}`);
        console.log(`  Buyer LEI: ${buyerLEI}`);
        console.log(`  Payment Chain: ${paymentChainID}`);
        console.log(`  Payment Wallet: ${paymentWalletAddress}`);
        console.log(`  Self-Attested: ${selfAttested ? 'YES' : 'NO'}`);
        
        // Check for edges (credential chain)
        const edges = cred.sad?.e || {};
        const hasEdge = Object.keys(edges).length > 0;
        console.log(`  Has Edge (Chained): ${hasEdge ? 'YES' : 'NO'}`);
        
        if (hasEdge) {
            console.log(`  Edge Details: ${JSON.stringify(edges, null, 2)}`);
        }
        
        console.log(``);
        
        // Add to results
        invoiceResults.push({
            said,
            schema,
            issuer,
            invoiceNumber,
            totalAmount,
            currency,
            dueDate,
            sellerLEI,
            buyerLEI,
            paymentChainID,
            paymentWalletAddress,
            selfAttested,
            hasEdge,
            edges: hasEdge ? edges : null
        });
    }
    
    // Save query results
    const queryResults = {
        agent: agentName,
        agentAID: agentAID.prefix,
        invoices: invoiceResults,
        totalInvoices: invoiceResults.length,
        timestamp: new Date().toISOString()
    };
    
    const outputPath = `./task-data/${agentName}-invoice-query-results.json`;
    await fs.promises.writeFile(outputPath, JSON.stringify(queryResults, null, 2));
    console.log(`✓ Query results saved to ${outputPath}`);
    console.log(``);
    
    console.log(`========================================`);
    console.log(`✅ QUERY COMPLETED`);
    console.log(`========================================`);
    console.log(``);
    console.log(`Summary:`);
    console.log(`  ✓ Agent: ${agentName}`);
    console.log(`  ✓ Total invoices: ${invoiceResults.length}`);
    console.log(`  ✓ Self-attested: ${invoiceResults.filter(i => i.selfAttested).length}`);
    console.log(`  ✓ Chained: ${invoiceResults.filter(i => i.hasEdge).length}`);
    console.log(``);
    
} catch (error: any) {
    console.error(`❌ Query failed: ${error.message}`);
    console.error(error);
    process.exit(1);
}
