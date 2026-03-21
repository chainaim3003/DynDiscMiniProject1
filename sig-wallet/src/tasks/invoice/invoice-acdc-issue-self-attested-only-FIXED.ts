/**
 * invoice-acdc-issue-self-attested-only-FIXED.ts
 * 
 * Issues a self-attested invoice credential where issuer = issuee.
 * 
 * FIXED VERSION: Uses alternative schema resolution method when primary fails
 * 
 * Key Features:
 * - Retry logic with exponential backoff
 * - Fallback to pre-resolved schema from QVI client
 * - Proper schema OOBI resolution before credential issuance
 * - Handles DNS failures (EAI_AGAIN) and connection refused (ECONNREFUSED)
 * - Direct credential issuance using signify-ts API
 */

import fs from "fs";
import { getOrCreateClient } from "../../client/identifiers.js";
import { waitOperation } from "../../client/operations.js";
import { resolveOobi } from "../../client/oobis.js";
import { createTimestamp } from "../../time.js";

// Invoice Schema SAID - must match the $id in self-attested-invoice.json
const INVOICE_SCHEMA_SAID = "EIKpV6ZqOn2Rg-DY86bIKDixNlgdvUQoSpijhVqs_EPu";

// Shared client passcode (used to access schemas already resolved by QVI/LE)
const SHARED_PASSCODE = "AEqhkGD_S_SRiO-oeJA4x";

// Retry helper function with exponential backoff
async function retryWithBackoff<T>(
    operation: () => Promise<T>,
    operationName: string,
    maxRetries: number = 5,
    initialDelayMs: number = 3000
): Promise<T> {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return await operation();
        } catch (error: any) {
            const isRetryable = 
                error.cause?.code === 'ECONNREFUSED' ||
                error.cause?.code === 'EAI_AGAIN' ||
                error.message?.includes('fetch failed') ||
                error.message?.includes('ETIMEDOUT') ||
                error.message?.includes('404');
            
            console.log(`  [Retry] ${operationName} failed on attempt #${attempt} of ${maxRetries}`);
            console.log(`    Error: ${error.message}`);
            
            if (attempt === maxRetries || !isRetryable) {
                console.error(`  [Retry] Max retries (${maxRetries}) reached for ${operationName}.`);
                throw error;
            }
            
            const delay = initialDelayMs * Math.pow(1.5, attempt - 1);
            console.log(`  [Retry] Waiting ${delay}ms before next attempt...`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    throw new Error('Unexpected: exited retry loop');
}

// Process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const issuerAidName = args[1];
const passcode = args[2];
const registryName = args[3];
const invoiceSchemaSaidArg = args[4];
const invoiceDataJson = args[5];
const outputPath = args[6];
const taskDataDir = args[7] || '/task-data';

console.log(`Issuing self-attested invoice credential...`);
console.log(`  Issuer (self-attested): ${issuerAidName}`);
console.log(`  Issuee: ${issuerAidName} (SAME as issuer)`);
console.log(`  Registry: ${registryName}`);
console.log(`  Task data dir: ${taskDataDir}`);

// Determine the actual passcode to use
let actualPasscode = passcode;

if (!passcode || passcode.trim() === '') {
    const branFilePath = `${taskDataDir}/${issuerAidName}-bran.txt`;
    console.log(`  No passcode provided, checking for BRAN file: ${branFilePath}`);
    
    if (fs.existsSync(branFilePath)) {
        actualPasscode = fs.readFileSync(branFilePath, 'utf-8').trim();
        console.log(`  ✓ Found agent's unique BRAN: ${actualPasscode.substring(0, 20)}...`);
    } else {
        console.error(`  ✗ ERROR: No BRAN file found at ${branFilePath}`);
        process.exit(1);
    }
}

// Get client with the agent's unique passcode
console.log(`\nConnecting to KERIA...`);
const client = await retryWithBackoff(
    () => getOrCreateClient(actualPasscode, env),
    'KERIA connection',
    5,
    3000
);
console.log(`  Client controller: ${client.controller.pre}`);
console.log(`  Client agent: ${client.agent?.pre}`);

// Get issuer AID
let issuerAid;
try {
    issuerAid = await retryWithBackoff(
        () => client.identifiers().get(issuerAidName),
        'Get identifier',
        5,
        2000
    );
} catch (error: any) {
    console.error(`  ✗ ERROR: Could not find identifier '${issuerAidName}'`);
    
    try {
        const identifiers = await client.identifiers().list();
        console.log(`  Available identifiers: ${identifiers.aids?.length || 0}`);
        identifiers.aids?.forEach((aid: any) => {
            console.log(`    - ${aid.name}: ${aid.prefix}`);
        });
    } catch (e) { /* ignore */ }
    
    process.exit(1);
}

const issuerPrefix = issuerAid.prefix;
console.log(`  Issuer AID: ${issuerPrefix}`);

// Parse invoice data
const invoiceData = JSON.parse(invoiceDataJson);

// Determine which schema SAID to use
const schemaSaid = invoiceSchemaSaidArg && invoiceSchemaSaidArg !== "EInvoiceSchemaPlaceholder" 
    ? invoiceSchemaSaidArg 
    : INVOICE_SCHEMA_SAID;

console.log(`\nUsing schema SAID: ${schemaSaid}`);

// ============================================================================
// CRITICAL FIX: Multi-strategy schema resolution
// ============================================================================

let schemaResolved = false;
const schemaOobi = `http://schema:7723/oobi/${schemaSaid}`;

// Strategy 1: Try to resolve directly with agent's client
console.log(`\nStrategy 1: Resolving schema OOBI with agent's client...`);
console.log(`  OOBI: ${schemaOobi}`);

try {
    await retryWithBackoff(
        async () => {
            await resolveOobi(client, schemaOobi, 'invoice-schema');
            console.log(`  ✓ Schema OOBI resolved successfully (Strategy 1)`);
        },
        'Schema OOBI resolution (agent client)',
        3,
        2000
    );
    schemaResolved = true;
} catch (error: any) {
    console.log(`  ✗ Strategy 1 failed: ${error.message}`);
}

// Strategy 2: Check if schema was already resolved in agent's cache
if (!schemaResolved) {
    console.log(`\nStrategy 2: Checking if schema exists in agent's cache...`);
    try {
        const schemas = await client.schemas().list();
        const found = schemas.find((s: any) => s.$id === schemaSaid);
        if (found) {
            console.log(`  ✓ Schema found in cache (Strategy 2)`);
            schemaResolved = true;
        } else {
            console.log(`  ✗ Schema not in cache`);
        }
    } catch (e) {
        console.log(`  ✗ Could not check schema cache`);
    }
}

// Strategy 3: Use shared client to resolve schema, then verify agent can see it
if (!schemaResolved) {
    console.log(`\nStrategy 3: Using shared client to resolve schema...`);
    try {
        const sharedClient = await getOrCreateClient(SHARED_PASSCODE, env);
        console.log(`  Shared client connected`);
        
        await retryWithBackoff(
            async () => {
                await resolveOobi(sharedClient, schemaOobi, 'invoice-schema-shared');
            },
            'Schema OOBI resolution (shared client)',
            3,
            2000
        );
        
        // Now try with agent's client again
        await new Promise(resolve => setTimeout(resolve, 2000));
        await resolveOobi(client, schemaOobi, 'invoice-schema');
        console.log(`  ✓ Schema resolved via shared client (Strategy 3)`);
        schemaResolved = true;
    } catch (error: any) {
        console.log(`  ✗ Strategy 3 failed: ${error.message}`);
    }
}

// Strategy 4: Direct schema injection (last resort)
if (!schemaResolved) {
    console.log(`\nStrategy 4: Attempting direct schema access...`);
    try {
        // Try to get schema directly - it might work even without OOBI resolution
        const schema = await client.schemas().get(schemaSaid);
        if (schema && schema.$id === schemaSaid) {
            console.log(`  ✓ Schema accessible directly (Strategy 4)`);
            schemaResolved = true;
        }
    } catch (e) {
        console.log(`  ✗ Strategy 4 failed`);
    }
}

if (!schemaResolved) {
    console.error(`\n✗ ERROR: All schema resolution strategies failed`);
    console.error(`\n  Please run these commands to diagnose:`);
    console.error(`  1. docker compose ps schema`);
    console.error(`  2. docker compose logs schema --tail=30`);
    console.error(`  3. docker compose exec keria wget -qO- http://schema:7723/oobi/${schemaSaid}`);
    console.error(`  4. docker compose restart schema && sleep 15`);
    console.error(`\n  Then re-run this script.`);
    process.exit(1);
}

// Get the registry
const registries = await client.registries().list(issuerAidName);
const issRegistry = registries.find((reg: any) => reg.name === registryName);

if (!issRegistry) {
    console.error(`\n✗ ERROR: Registry '${registryName}' not found for ${issuerAidName}`);
    registries.forEach((reg: any) => {
        console.log(`    - ${reg.name}: ${reg.regk}`);
    });
    process.exit(1);
}

console.log(`\nRegistry found: ${issRegistry.regk}`);

// NO EDGE SECTION - Self-attested credential
const credEdge = undefined;

// Construct rules for self-attested invoice
const credRules = {
    d: '',
    usageDisclaimer: {
        l: 'This is a self-attested invoice credential issued by the agent. The issuer and holder are the same entity.'
    },
    selfAttestation: {
        l: 'This credential is self-attested. The trust chain derives from the agent delegation, not from credential chaining.'
    }
};

// Prepare credential subject (attributes)
const kargsSub = {
    i: issuerPrefix,
    dt: createTimestamp(),
    ...invoiceData,
};

// Prepare credential data
const issData = {
    i: issuerPrefix,
    ri: issRegistry.regk,
    s: schemaSaid,
    a: kargsSub,
    e: credEdge,
    r: credRules,
};

console.log(`\nIssuing credential...`);
console.log(`  Issuer = Issuee = ${issuerPrefix} (self-attested)`);

// Issue the credential
let issResult;
try {
    issResult = await retryWithBackoff(
        async () => {
            const result = await client.credentials().issue(issuerAidName, issData);
            const op = await waitOperation(client, result.op);
            if (op.error) {
                throw new Error(`Credential issuance failed: ${JSON.stringify(op.error)}`);
            }
            return { ...result, response: op.response };
        },
        'Credential issuance',
        5,
        3000
    );
} catch (error: any) {
    console.error(`\n✗ ERROR: Failed to issue credential`);
    console.error(`  ${error.message}`);
    process.exit(1);
}

const credentialSaid = issResult.response?.ced?.d;

if (!credentialSaid) {
    console.error(`\n✗ ERROR: No credential SAID returned`);
    process.exit(1);
}

// Get the full credential
const cred = await client.credentials().get(credentialSaid);

const said = cred.sad.d;
const issuer = cred.sad.i;
const issuee = cred.sad?.a?.i;

console.log(`\n✓ Self-attested invoice credential created: ${said}`);
console.log(`  Issuer: ${issuer}`);
console.log(`  Issuee: ${issuee}`);
console.log(`  Self-attested: ${issuer === issuee ? 'YES ✓' : 'NO'}`);

if (issuer !== issuee) {
    console.error(`\n❌ ERROR: Credential is not self-attested!`);
    process.exit(1);
}

// Save credential info
const credInfo = {
    said,
    issuer,
    issuee,
    selfAttested: true,
    hasEdge: false,
    schema: schemaSaid,
    registry: issRegistry.regk,
    invoiceNumber: invoiceData.invoiceNumber,
    totalAmount: invoiceData.totalAmount,
    currency: invoiceData.currency,
    dueDate: invoiceData.dueDate,
    paymentMethod: invoiceData.paymentMethod,
    paymentChainID: invoiceData.paymentChainID,
    paymentWalletAddress: invoiceData.paymentWalletAddress,
    ref_uri: invoiceData.ref_uri,
    acdc: issResult.acdc,
    anc: issResult.anc,
    iss: issResult.iss
};

await fs.promises.writeFile(outputPath, JSON.stringify(credInfo, null, 2));
console.log(`\n✓ Credential info saved to ${outputPath}`);
console.log(`\n✨ Self-attested invoice credential issued successfully!`);
