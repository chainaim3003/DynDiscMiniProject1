import fs from "fs";
import { getOrCreateClient, getAID } from "../../client/identifiers.js";

/**
 * Query all credentials from KERIA agent
 * 
 * Usage: tsx agent-query-credentials.ts <env> <passcode> <agentName>
 */

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1];
const agentName = args[2];

console.log(`========================================`);
console.log(`QUERY AGENT CREDENTIALS`);
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
    console.log(`[2/3] Querying all credentials from KERIA...`);
    const credentials = await client.credentials().list(agentAID.name);
    
    console.log(`✓ Found ${credentials.length} total credential(s)`);
    console.log(``);
    
    if (credentials.length === 0) {
        console.log(`⚠ No credentials found for ${agentName}`);
        console.log(`  The agent may not have any credentials yet`);
        console.log(``);
        
        // Save empty results
        const queryResults = {
            agent: agentName,
            agentAID: agentAID.prefix,
            credentials: [],
            totalCredentials: 0,
            timestamp: new Date().toISOString()
        };
        
        const outputPath = `/task-data/${agentName}-credential-query-results.json`;
        await fs.promises.writeFile(outputPath, JSON.stringify(queryResults, null, 2));
        
        process.exit(0);
    }
    
    // Process and display each credential
    console.log(`[3/3] Processing credentials...`);
    const credentialResults = [];
    
    for (let i = 0; i < credentials.length; i++) {
        const cred = credentials[i];
        const sad = cred.sad || {};
        const said = sad.d || 'N/A';
        const schema = sad.s || 'N/A';
        const issuer = sad.i || 'N/A';
        const attributes = sad.a || {};
        
        console.log(`Credential #${i + 1}:`);
        console.log(`  SAID: ${said}`);
        console.log(`  Schema: ${schema}`);
        console.log(`  Issuer: ${issuer}`);
        
        // Determine credential type from schema
        let credType = 'Unknown';
        if (schema.toLowerCase().includes('invoice')) {
            credType = 'Invoice';
        } else if (schema.toLowerCase().includes('oor')) {
            credType = 'OOR (Official Organizational Role)';
        } else if (schema.toLowerCase().includes('le')) {
            credType = 'LE (Legal Entity)';
        } else if (schema.toLowerCase().includes('qvi')) {
            credType = 'QVI (Qualified vLEI Issuer)';
        }
        
        console.log(`  Type: ${credType}`);
        
        // Check if self-attested
        const holderInAttributes = attributes.i;
        const selfAttested = issuer === holderInAttributes || issuer === agentAID.prefix;
        console.log(`  Self-Attested: ${selfAttested ? 'YES' : 'NO'}`);
        
        // Check for edges (credential chain)
        const edges = sad.e || {};
        const hasEdge = Object.keys(edges).length > 0;
        console.log(`  Has Edge (Chained): ${hasEdge ? 'YES' : 'NO'}`);
        
        // Display some key attributes based on type
        if (credType === 'Invoice') {
            const invoiceNumber = attributes.invoiceNumber || 'N/A';
            const totalAmount = attributes.totalAmount || 'N/A';
            const currency = attributes.currency || 'N/A';
            console.log(`  Invoice Number: ${invoiceNumber}`);
            console.log(`  Amount: ${totalAmount} ${currency}`);
        } else if (credType === 'OOR (Official Organizational Role)') {
            const personName = attributes.personLegalName || 'N/A';
            const officialRole = attributes.officialRole || 'N/A';
            const lei = attributes.LEI || 'N/A';
            console.log(`  Person: ${personName}`);
            console.log(`  Role: ${officialRole}`);
            console.log(`  LEI: ${lei}`);
        } else if (credType === 'LE (Legal Entity)') {
            const lei = attributes.LEI || 'N/A';
            console.log(`  LEI: ${lei}`);
        }
        
        console.log(``);
        
        // Add to results
        credentialResults.push({
            said,
            schema,
            issuer,
            type: credType,
            selfAttested,
            hasEdge,
            edges: hasEdge ? edges : null,
            attributes
        });
    }
    
    // Save query results
    const queryResults = {
        agent: agentName,
        agentAID: agentAID.prefix,
        credentials: credentialResults,
        totalCredentials: credentialResults.length,
        selfAttestedCount: credentialResults.filter(c => c.selfAttested).length,
        chainedCount: credentialResults.filter(c => c.hasEdge).length,
        timestamp: new Date().toISOString()
    };
    
    const outputPath = `/task-data/${agentName}-credential-query-results.json`;
    await fs.promises.writeFile(outputPath, JSON.stringify(queryResults, null, 2));
    console.log(`✓ Query results saved to ${outputPath}`);
    console.log(``);
    
    console.log(`========================================`);
    console.log(`✅ QUERY COMPLETED`);
    console.log(`========================================`);
    console.log(``);
    console.log(`Summary:`);
    console.log(`  ✓ Agent: ${agentName}`);
    console.log(`  ✓ Total credentials: ${credentialResults.length}`);
    console.log(`  ✓ Self-attested: ${queryResults.selfAttestedCount}`);
    console.log(`  ✓ Chained: ${queryResults.chainedCount}`);
    console.log(``);
    
} catch (error: any) {
    console.error(`❌ Query failed: ${error.message}`);
    console.error(error);
    process.exit(1);
}
