/**
 * invoice-acdc-issue-self-attested-only.ts
 * 
 * Issues a self-attested invoice credential where issuer = issuee.
 * 
 * IMPORTANT: Schema SAID is read from appconfig/schemaSaids.json
 * Run ./saidify-with-docker.sh after schema changes to update SAIDs.
 */

import fs from "fs";
import path from "path";
import { getOrCreateClient } from "../../client/identifiers.js";
import { waitOperation } from "../../client/operations.js";
import { resolveOobi } from "../../client/oobis.js";
import { createTimestamp } from "../../time.js";

// ============================================================================
// LOAD SCHEMA SAID FROM CONFIG (SINGLE SOURCE OF TRUTH)
// ============================================================================
function getInvoiceSchemaSaid(): string {
    // Try multiple possible locations for the config file
    const possiblePaths = [
        '/vlei/appconfig/schemaSaids.json',           // Inside Docker container
        './appconfig/schemaSaids.json',                // Relative to working dir
        path.join(process.cwd(), 'appconfig/schemaSaids.json'),
    ];
    
    for (const configPath of possiblePaths) {
        try {
            if (fs.existsSync(configPath)) {
                const config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
                if (config.invoiceSchema?.said) {
                    console.log(`  Loaded invoice schema SAID from: ${configPath}`);
                    console.log(`  Config file value: ${config.invoiceSchema.said}`);
                    return config.invoiceSchema.said;
                }
            }
        } catch (e) {
            // Continue to next path
        }
    }
    
    // Fallback - this should be updated by saidify-with-docker.sh
    console.warn(`  WARNING: Could not load schemaSaids.json, using fallback SAID`);
    return "EIKpV6ZqOn2Rg-DY86bIKDixNlgdvUQoSpijhVqs_EPu";
}

const INVOICE_SCHEMA_SAID = getInvoiceSchemaSaid();

// SHARED_PASSCODE - This is the GEDA_SALT used for GEDA/QVI/LE setup
const SHARED_PASSCODE = "0AD45YWdzWSwNREuAoitH_CC";

// Retry helper function with exponential backoff
async function retryWithBackoff<T>(
    operation: () => Promise<T>,
    operationName: string,
    maxRetries: number = 5,
    initialDelayMs: number = 5000
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
const invoiceSchemaSaidArg = args[4]; // Optional override from command line
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
    5000
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
        3000
    );
} catch (error: any) {
    console.error(`  ✗ ERROR: Could not find identifier '${issuerAidName}'`);
    
    try {
        const identifiers = await client.identifiers().list();
        console.log(`  Available identifiers in this client: ${identifiers.aids?.length || 0}`);
        if (identifiers.aids && identifiers.aids.length > 0) {
            identifiers.aids.forEach((aid: any) => {
                console.log(`    - ${aid.name}: ${aid.prefix}`);
            });
        }
    } catch (e) {
        console.log(`  Could not list identifiers`);
    }
    
    process.exit(1);
}

const issuerPrefix = issuerAid.prefix;
console.log(`  Issuer AID: ${issuerPrefix}`);

// Parse invoice data
const invoiceData = JSON.parse(invoiceDataJson);

// Determine which schema SAID to use (command line arg overrides config)
// CRITICAL: Debug logging to diagnose SAID mismatch issues
console.log(`\n╔══════════════════════════════════════════════════════════════╗`);
console.log(`║  SCHEMA SAID RESOLUTION (DEBUG v2)                          ║`);
console.log(`╚══════════════════════════════════════════════════════════════╝`);
console.log(`  Command line arg (args[4]): '${invoiceSchemaSaidArg}'`);
console.log(`  Config SAID (from schemaSaids.json): '${INVOICE_SCHEMA_SAID}'`);
console.log(`  Expected correct SAID: 'EIKpV6ZqOn2Rg-DY86bIKDixNlgdvUQoSpijhVqs_EPu'`);
console.log(``);
console.log(`  Arg validation:`);
console.log(`    - Is arg truthy? ${!!invoiceSchemaSaidArg}`);
console.log(`    - Not placeholder? ${invoiceSchemaSaidArg !== "EInvoiceSchemaPlaceholder"}`);
console.log(`    - Starts with 'E'? ${invoiceSchemaSaidArg?.startsWith('E')}`);

const schemaSaid = invoiceSchemaSaidArg && 
                   invoiceSchemaSaidArg !== "EInvoiceSchemaPlaceholder" &&
                   invoiceSchemaSaidArg.startsWith('E')
    ? invoiceSchemaSaidArg 
    : INVOICE_SCHEMA_SAID;

const usingSource = invoiceSchemaSaidArg && invoiceSchemaSaidArg !== "EInvoiceSchemaPlaceholder" && invoiceSchemaSaidArg.startsWith('E') ? 'COMMAND LINE ARG' : 'CONFIG FILE';
console.log(``);
console.log(`  ═══════════════════════════════════════════════════════════`);
console.log(`  >>> USING SAID FROM: ${usingSource}`);
console.log(`  >>> FINAL SAID: ${schemaSaid}`);

// CRITICAL: Warn if using the OLD/WRONG schema SAID  
const KNOWN_OLD_SAID = 'EIKpV6ZqOn2Rg-DY86bIKDixNlgdvUQoSpijhVqs_EPu';
if (schemaSaid === KNOWN_OLD_SAID) {
    console.error(`\n  ╔════════════════════════════════════════════════════════════════╗`);
    console.error(`  ║  ⚠️  WARNING: USING OLD/CACHED SCHEMA SAID!                   ║`);
    console.error(`  ║  This SAID is missing lineItems and paymentTerms fields.      ║`);
    console.error(`  ║  Expected: EIKpV6ZqOn2Rg-DY86bIKDixNlgdvUQoSpijhVqs_EPu       ║`);
    console.error(`  ║  Got:      ${schemaSaid}       ║`);
    console.error(`  ║  FIX: Run ./stop.sh && ./deploy.sh to clear KERIA cache       ║`);
    console.error(`  ╚════════════════════════════════════════════════════════════════╝\n`);
}
console.log(`  ═══════════════════════════════════════════════════════════`);
console.log(`\nUsing schema SAID: ${schemaSaid}`);

// CRITICAL: Resolve schema OOBI before credential issuance
const schemaOobi = `http://schema:7723/oobi/${schemaSaid}`;
console.log(`\nResolving schema OOBI: ${schemaOobi}`);

// Multi-strategy schema resolution
async function resolveSchemaWithFallback(): Promise<void> {
    console.log(`  Attempting multi-strategy schema resolution...`);
    
    // Strategy 1: Try direct resolution with agent's client
    console.log(`  Strategy 1: Direct resolution with agent client...`);
    try {
        await retryWithBackoff(
            async () => {
                await resolveOobi(client, schemaOobi, `invoice-schema-${Date.now()}`);
            },
            'Direct schema resolution',
            2,
            2000
        );
        console.log(`  ✓ Strategy 1 SUCCESS: Schema resolved directly`);
        return;
    } catch (e: any) {
        console.log(`  ✗ Strategy 1 failed: ${e.message}`);
    }
    
    // Strategy 2: Check if schema is already in agent's cache
    console.log(`  Strategy 2: Check agent's schema cache...`);
    try {
        const schemas = await client.schemas().list();
        const found = schemas.find((s: any) => s.$id === schemaSaid);
        if (found) {
            console.log(`  ✓ Strategy 2 SUCCESS: Schema already in agent cache`);
            return;
        }
        console.log(`  ✗ Strategy 2: Schema not in agent cache (${schemas.length} schemas found)`);
    } catch (e: any) {
        console.log(`  ✗ Strategy 2 failed: ${e.message}`);
    }
    
    // Strategy 3: Use SHARED CLIENT to prime KERIA's cache
    console.log(`  Strategy 3: Use shared client to prime KERIA cache...`);
    try {
        const sharedClient = await getOrCreateClient(SHARED_PASSCODE, env);
        console.log(`    Shared client connected: ${sharedClient.controller.pre}`);
        
        await retryWithBackoff(
            async () => {
                await resolveOobi(sharedClient, schemaOobi, `invoice-schema-shared-${Date.now()}`);
            },
            'Shared client schema resolution',
            3,
            3000
        );
        console.log(`    ✓ Schema resolved via shared client - KERIA cache primed`);
        
        await retryWithBackoff(
            async () => {
                await resolveOobi(client, schemaOobi, `invoice-schema-agent-${Date.now()}`);
            },
            'Agent client schema resolution (post-prime)',
            3,
            3000
        );
        console.log(`  ✓ Strategy 3 SUCCESS: Schema resolved via shared client priming`);
        return;
    } catch (e: any) {
        console.log(`  ✗ Strategy 3 failed: ${e.message}`);
    }
    
    // Strategy 4: Direct fetch from KERIA
    console.log(`  Strategy 4: Direct schema fetch from KERIA...`);
    try {
        const schema = await client.schemas().get(schemaSaid);
        if (schema && schema.$id === schemaSaid) {
            console.log(`  ✓ Strategy 4 SUCCESS: Schema fetched from KERIA`);
            return;
        }
    } catch (e: any) {
        console.log(`  ✗ Strategy 4 failed: ${e.message}`);
    }
    
    throw new Error(`All schema resolution strategies failed for ${schemaSaid}`);
}

try {
    await resolveSchemaWithFallback();
    console.log(`  ✓ Schema OOBI resolved successfully`);
} catch (error: any) {
    console.error(`\n✗ ERROR: Failed to resolve schema OOBI after all strategies`);
    console.error(`  ${error.message}`);
    console.error(`\n  To fix:`);
    console.error(`  1. Run: ./saidify-with-docker.sh`);
    console.error(`  2. Run: docker compose restart schema`);
    console.error(`  3. Test: curl http://localhost:7723/oobi/${schemaSaid}`);
    process.exit(1);
}

// NO EDGE - Self-attested credential
const credEdge = undefined;

// Rules for self-attested invoice
const credRules = {
    d: '',
    usageDisclaimer: {
        l: 'This is a self-attested invoice credential issued by the agent. The issuer and holder are the same entity.'
    },
    selfAttestation: {
        l: 'This credential is self-attested. The trust chain derives from the agent delegation to the GLEIF root.'
    }
};

console.log(`\nIssuing self-attested credential to registry ${registryName}...`);
console.log(`  Issuer = Issuee = ${issuerPrefix} (self-attested)`);
console.log(`  Edge: NONE (no OOR chain)`);

// Get the registry
const registries = await client.registries().list(issuerAidName);
const issRegistry = registries.find((reg: any) => reg.name === registryName);

if (!issRegistry) {
    console.error(`\n✗ ERROR: Registry '${registryName}' not found for ${issuerAidName}`);
    console.log(`  Available registries:`);
    registries.forEach((reg: any) => {
        console.log(`    - ${reg.name}: ${reg.regk}`);
    });
    process.exit(1);
}

console.log(`  Registry found: ${issRegistry.regk}`);

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

console.log(`\nCredential data prepared:`);
console.log(`  Issuer: ${issData.i}`);
console.log(`  Registry: ${issData.ri}`);
console.log(`  Schema: ${issData.s}`);
console.log(`  Issuee (in attributes): ${kargsSub.i}`);

// Issue the credential
let issResult;
try {
    issResult = await retryWithBackoff(
        async () => {
            const result = await client.credentials().issue(issuerAidName, issData);
            const op = await waitOperation(client, result.op);
            if (op.error) {
                throw new Error(`Credential issuance operation failed: ${JSON.stringify(op.error)}`);
            }
            return { ...result, response: op.response };
        },
        'Credential issuance',
        5,
        5000
    );
} catch (error: any) {
    console.error(`\n✗ ERROR: Failed to issue credential after all retries`);
    console.error(`  ${error.message}`);
    process.exit(1);
}

const credentialSaid = issResult.response?.ced?.d;

if (!credentialSaid) {
    console.error(`\n✗ ERROR: No credential SAID returned from issuance`);
    console.log(`  Response:`, JSON.stringify(issResult.response, null, 2));
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
    paymentTerms: invoiceData.paymentTerms || null,
    acdc: issResult.acdc,
    anc: issResult.anc,
    iss: issResult.iss
};

await fs.promises.writeFile(outputPath, JSON.stringify(credInfo, null, 2));
console.log(`✓ Self-attested invoice credential info saved to ${outputPath}`);
