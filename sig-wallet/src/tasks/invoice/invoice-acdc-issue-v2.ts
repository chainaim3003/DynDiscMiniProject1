/**
 * invoice-acdc-issue-v2.ts
 * 
 * FIXED VERSION - Based on official vLEI documentation
 * 
 * This script follows the exact pattern from:
 * - 102_20_KERIA_Signify_Credential_Issuance.md
 * - 101_65_ACDC_Issuance.md
 * 
 * Key fixes:
 * 1. Uses shared passcode for schema resolution (same client that resolved OOR/LE schemas)
 * 2. Properly structures credential data to match schema
 * 3. Follows IPEX Grant pattern from official documentation
 */

import fs from "fs";
import { Serder } from "signify-ts";
import { getOrCreateClient } from "../../client/identifiers.js";
import { waitOperation } from "../../client/operations.js";
import { resolveOobi } from "../../client/oobis.js";
import { createTimestamp } from "../../time.js";

// ============================================================================
// CONFIGURATION
// ============================================================================

// Load Schema SAID from config (single source of truth)
import path from "path";

function getInvoiceSchemaSaid(): string {
    const possiblePaths = [
        '/vlei/appconfig/schemaSaids.json',
        './appconfig/schemaSaids.json',
        path.join(process.cwd(), 'appconfig/schemaSaids.json'),
    ];
    for (const configPath of possiblePaths) {
        try {
            if (fs.existsSync(configPath)) {
                const config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
                if (config.invoiceSchema?.said) return config.invoiceSchema.said;
            }
        } catch (e) { /* continue */ }
    }
    return "EIKpV6ZqOn2Rg-DY86bIKDixNlgdvUQoSpijhVqs_EPu"; // Fallback
}

const INVOICE_SCHEMA_SAID = getInvoiceSchemaSaid();

// CRITICAL: Use the SAME passcode that was used for schema resolution during LE/OOR setup
// This is the "shared" passcode from the original workflow
const SHARED_PASSCODE = "AEqhkGD_S_SRiO-oeJA4x";

// Schema server URL - from inside Docker network
const SCHEMA_SERVER = "http://schema:7723";

// ============================================================================
// RETRY HELPER
// ============================================================================

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
            const errorMsg = error.message || String(error);
            const isRetryable = 
                errorMsg.includes('fetch failed') ||
                errorMsg.includes('ECONNREFUSED') ||
                errorMsg.includes('EAI_AGAIN') ||
                errorMsg.includes('ETIMEDOUT') ||
                errorMsg.includes('not found');
            
            console.log(`  [Retry] ${operationName} failed on attempt #${attempt} of ${maxRetries}`);
            console.log(`    Error: ${errorMsg}`);
            
            if (attempt === maxRetries || !isRetryable) {
                throw error;
            }
            
            const delay = initialDelayMs * Math.pow(1.5, attempt - 1);
            console.log(`  [Retry] Waiting ${Math.round(delay)}ms before next attempt...`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    throw new Error('Unexpected: exited retry loop');
}

// ============================================================================
// SCHEMA RESOLUTION - CRITICAL FIX
// ============================================================================

/**
 * Resolves schema OOBI using multiple strategies
 * 
 * The key insight from official documentation:
 * "Every client that needs to issue credentials must resolve the schema OOBI"
 * 
 * But the agent's SignifyClient is a DIFFERENT instance than the one used
 * for OOR/LE credentials. So we need to either:
 * 1. Use the same passcode to get the same client that has the schema cached
 * 2. Or resolve the schema OOBI with the agent's client
 */
async function resolveSchemaWithFallback(
    agentClient: any,
    agentPasscode: string,
    schemaSaid: string,
    env: 'docker' | 'testnet'
): Promise<boolean> {
    const schemaOobi = `${SCHEMA_SERVER}/oobi/${schemaSaid}`;
    
    console.log(`\n📋 Resolving schema OOBI: ${schemaOobi}`);
    
    // Strategy 1: Try direct resolution with agent's client
    console.log(`  Strategy 1: Direct resolution with agent client...`);
    try {
        await retryWithBackoff(
            async () => {
                await resolveOobi(agentClient, schemaOobi, `invoice-schema-${Date.now()}`);
                console.log(`    ✓ Schema resolved directly`);
            },
            'Direct schema resolution',
            3,
            2000
        );
        return true;
    } catch (e: any) {
        console.log(`    ✗ Direct resolution failed: ${e.message}`);
    }
    
    // Strategy 2: Check if schema already exists in agent's cache
    console.log(`  Strategy 2: Check schema cache...`);
    try {
        const schemas = await agentClient.schemas().list();
        const found = schemas.find((s: any) => s.$id === schemaSaid);
        if (found) {
            console.log(`    ✓ Schema already in cache`);
            return true;
        }
        console.log(`    ✗ Schema not in cache (${schemas.length} schemas found)`);
    } catch (e: any) {
        console.log(`    ✗ Could not check cache: ${e.message}`);
    }
    
    // Strategy 3: Use SHARED passcode to get client with schema, then resolve with agent
    console.log(`  Strategy 3: Use shared client to resolve schema...`);
    try {
        // Get the shared client (same one used for OOR/LE setup)
        const sharedClient = await getOrCreateClient(SHARED_PASSCODE, env);
        console.log(`    Shared client connected: ${sharedClient.controller.pre}`);
        
        // Resolve with shared client first
        await retryWithBackoff(
            async () => {
                await resolveOobi(sharedClient, schemaOobi, `invoice-schema-shared`);
            },
            'Shared client schema resolution',
            3,
            2000
        );
        console.log(`    ✓ Schema resolved via shared client`);
        
        // Now try again with agent client
        await retryWithBackoff(
            async () => {
                await resolveOobi(agentClient, schemaOobi, `invoice-schema`);
            },
            'Agent schema resolution after shared',
            3,
            2000
        );
        console.log(`    ✓ Schema resolved for agent client`);
        return true;
    } catch (e: any) {
        console.log(`    ✗ Shared client approach failed: ${e.message}`);
    }
    
    // Strategy 4: Direct schema get (if KERIA has it from another client)
    console.log(`  Strategy 4: Direct schema fetch...`);
    try {
        const schema = await agentClient.schemas().get(schemaSaid);
        if (schema && schema.$id === schemaSaid) {
            console.log(`    ✓ Schema fetched directly from KERIA`);
            return true;
        }
    } catch (e: any) {
        console.log(`    ✗ Direct fetch failed: ${e.message}`);
    }
    
    return false;
}

// ============================================================================
// MAIN SCRIPT
// ============================================================================

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

console.log(`\n╔════════════════════════════════════════════════════════════╗`);
console.log(`║     SELF-ATTESTED INVOICE CREDENTIAL ISSUANCE (V2)        ║`);
console.log(`╚════════════════════════════════════════════════════════════╝\n`);

console.log(`Configuration:`);
console.log(`  Issuer/Issuee: ${issuerAidName} (self-attested)`);
console.log(`  Registry: ${registryName}`);
console.log(`  Environment: ${env}`);

// Determine actual passcode
let actualPasscode = passcode;
if (!passcode || passcode.trim() === '') {
    const branFilePath = `${taskDataDir}/${issuerAidName}-bran.txt`;
    if (fs.existsSync(branFilePath)) {
        actualPasscode = fs.readFileSync(branFilePath, 'utf-8').trim();
        console.log(`  Passcode: from BRAN file`);
    } else {
        console.error(`\n✗ ERROR: No passcode provided and no BRAN file at ${branFilePath}`);
        process.exit(1);
    }
} else {
    console.log(`  Passcode: provided`);
}

// Determine schema SAID
const schemaSaid = invoiceSchemaSaidArg && 
                   invoiceSchemaSaidArg !== "EInvoiceSchemaPlaceholder" &&
                   invoiceSchemaSaidArg.startsWith('E')
    ? invoiceSchemaSaidArg 
    : INVOICE_SCHEMA_SAID;

console.log(`  Schema SAID: ${schemaSaid}`);

// Parse invoice data
const invoiceData = JSON.parse(invoiceDataJson);
console.log(`  Invoice #: ${invoiceData.invoiceNumber}`);

// ============================================================================
// CONNECT TO KERIA
// ============================================================================

console.log(`\n🔗 Connecting to KERIA...`);
const client = await retryWithBackoff(
    () => getOrCreateClient(actualPasscode, env),
    'KERIA connection',
    5,
    5000
);
console.log(`  ✓ Connected: ${client.controller.pre}`);

// Get issuer AID
let issuerAid;
try {
    issuerAid = await client.identifiers().get(issuerAidName);
    console.log(`  ✓ Issuer AID: ${issuerAid.prefix}`);
} catch (error: any) {
    console.error(`\n✗ ERROR: Could not find identifier '${issuerAidName}'`);
    
    // List available identifiers
    try {
        const identifiers = await client.identifiers().list();
        console.log(`  Available identifiers:`);
        identifiers.aids?.forEach((aid: any) => {
            console.log(`    - ${aid.name}: ${aid.prefix}`);
        });
    } catch (e) { /* ignore */ }
    
    process.exit(1);
}

const issuerPrefix = issuerAid.prefix;

// ============================================================================
// RESOLVE SCHEMA (CRITICAL STEP)
// ============================================================================

const schemaResolved = await resolveSchemaWithFallback(client, actualPasscode, schemaSaid, env);

if (!schemaResolved) {
    console.error(`\n╔════════════════════════════════════════════════════════════╗`);
    console.error(`║  SCHEMA RESOLUTION FAILED - ALL STRATEGIES EXHAUSTED       ║`);
    console.error(`╚════════════════════════════════════════════════════════════╝\n`);
    console.error(`The invoice schema (${schemaSaid}) could not be resolved.`);
    console.error(`\nPossible causes:`);
    console.error(`  1. Schema file not properly SAIDified`);
    console.error(`  2. Schema not loaded by vLEI-server`);
    console.error(`  3. Docker network connectivity issue`);
    console.error(`\nDiagnostic steps:`);
    console.error(`  1. Check schema file: curl http://localhost:7723/oobi/${schemaSaid}`);
    console.error(`  2. Check from KERIA: docker compose exec keria wget -qO- http://schema:7723/oobi/${schemaSaid}`);
    console.error(`  3. Verify schema container: docker compose logs schema`);
    console.error(`  4. Restart schema: docker compose restart schema`);
    process.exit(1);
}

// ============================================================================
// GET REGISTRY
// ============================================================================

console.log(`\n📦 Getting credential registry...`);
const registries = await client.registries().list(issuerAidName);
const issRegistry = registries.find((reg: any) => reg.name === registryName);

if (!issRegistry) {
    console.error(`\n✗ ERROR: Registry '${registryName}' not found`);
    console.log(`  Available registries:`);
    registries.forEach((reg: any) => {
        console.log(`    - ${reg.name}: ${reg.regk}`);
    });
    process.exit(1);
}

console.log(`  ✓ Registry: ${issRegistry.regk}`);

// ============================================================================
// PREPARE CREDENTIAL DATA (Following Official Documentation Pattern)
// ============================================================================

console.log(`\n📝 Preparing credential data...`);

// Attributes - following the exact pattern from 102_20_KERIA_Signify_Credential_Issuance.md
const credentialAttributes = {
    i: issuerPrefix,  // Issuee = Issuer for self-attested
    dt: createTimestamp(),
    ...invoiceData,
};

// Rules - simple structure for self-attested credential
const credentialRules = {
    d: '', // SAID will be calculated
    usageDisclaimer: {
        l: 'This is a self-attested invoice credential. The issuer and holder are the same entity.'
    },
    selfAttestation: {
        l: 'This credential is self-attested. Trust derives from the agent delegation chain to the GLEIF root.'
    }
};

// Full credential data structure
const credentialData = {
    i: issuerPrefix,           // Issuer
    ri: issRegistry.regk,      // Registry SAID
    s: schemaSaid,             // Schema SAID
    a: credentialAttributes,   // Attributes
    e: undefined,              // No edges for self-attested
    r: credentialRules,        // Rules
};

console.log(`  Issuer: ${credentialData.i}`);
console.log(`  Registry: ${credentialData.ri}`);
console.log(`  Schema: ${credentialData.s}`);
console.log(`  Issuee: ${credentialAttributes.i} (self-attested)`);

// ============================================================================
// ISSUE CREDENTIAL
// ============================================================================

console.log(`\n🎫 Issuing credential...`);

let issResult;
try {
    issResult = await retryWithBackoff(
        async () => {
            const result = await client.credentials().issue(issuerAidName, credentialData);
            const op = await waitOperation(client, result.op);
            if (op.error) {
                throw new Error(`Issuance failed: ${JSON.stringify(op.error)}`);
            }
            return { ...result, response: op.response };
        },
        'Credential issuance',
        3,
        5000
    );
} catch (error: any) {
    console.error(`\n✗ ERROR: Credential issuance failed`);
    console.error(`  ${error.message}`);
    
    // Common error diagnostics
    if (error.message.includes('schema') || error.message.includes('Schema')) {
        console.error(`\n  Schema-related error. Check:`);
        console.error(`  - Schema SAID matches $id in file`);
        console.error(`  - Schema is properly SAIDified`);
        console.error(`  - Credential attributes match schema requirements`);
    }
    if (error.message.includes('400')) {
        console.error(`\n  Bad Request (400). Credential data may not match schema.`);
        console.error(`  Credential data sent:`, JSON.stringify(credentialData, null, 2));
    }
    
    process.exit(1);
}

const credentialSaid = issResult.response?.ced?.d;

if (!credentialSaid) {
    console.error(`\n✗ ERROR: No credential SAID returned`);
    console.log(`  Response:`, JSON.stringify(issResult.response, null, 2));
    process.exit(1);
}

// Get full credential
const cred = await client.credentials().get(credentialSaid);

const said = cred.sad.d;
const issuer = cred.sad.i;
const issuee = cred.sad?.a?.i;

// ============================================================================
// VERIFICATION
// ============================================================================

console.log(`\n✅ CREDENTIAL ISSUED SUCCESSFULLY`);
console.log(`  SAID: ${said}`);
console.log(`  Issuer: ${issuer}`);
console.log(`  Issuee: ${issuee}`);
console.log(`  Self-attested: ${issuer === issuee ? 'YES ✓' : 'NO ✗'}`);

if (issuer !== issuee) {
    console.error(`\n⚠️  WARNING: Credential is not self-attested!`);
}

// ============================================================================
// SAVE OUTPUT
// ============================================================================

const credInfo = {
    said,
    issuer,
    issuee,
    selfAttested: issuer === issuee,
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
    paymentTerms: invoiceData.paymentTerms || null,
    // For IPEX Grant if needed later
    acdc: issResult.acdc,
    anc: issResult.anc,
    iss: issResult.iss
};

await fs.promises.writeFile(outputPath, JSON.stringify(credInfo, null, 2));
console.log(`\n📄 Credential info saved to: ${outputPath}`);

console.log(`\n╔════════════════════════════════════════════════════════════╗`);
console.log(`║                    ISSUANCE COMPLETE                       ║`);
console.log(`╚════════════════════════════════════════════════════════════╝`);
console.log(`  Next step: Use IPEX Grant to present credential to verifier`);
