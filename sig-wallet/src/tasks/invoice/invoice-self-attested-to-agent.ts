/**
 * invoice-self-attested-to-agent.ts
 * 
 * Issues a SELF-ATTESTED invoice credential where the delegated agent
 * is BOTH the issuer AND the issuee.
 * 
 * Trust Chain:
 *   GEDA → QVI → LE → OOR Holder → Agent (jupiterSellerAgent)
 *                                      ↓
 *                          Issues credential TO ITSELF
 * 
 * This is valid because:
 * 1. The agent has a valid delegation chain to the GLEIF root
 * 2. Self-attestation is a recognized pattern in vLEI
 * 3. The agent has its own registry for credential management
 */

import fs from "fs";
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
const SHARED_PASSCODE = "AEqhkGD_S_SRiO-oeJA4x"; // For schema resolution fallback
const SCHEMA_SERVER = "http://schema:7723";

// ============================================================================
// HELPERS
// ============================================================================

async function retry<T>(fn: () => Promise<T>, name: string, attempts = 5, delayMs = 3000): Promise<T> {
    for (let i = 1; i <= attempts; i++) {
        try {
            return await fn();
        } catch (e: any) {
            console.log(`  [${name}] Attempt ${i}/${attempts} failed: ${e.message}`);
            if (i === attempts) throw e;
            await new Promise(r => setTimeout(r, delayMs * i));
        }
    }
    throw new Error('Unreachable');
}

async function resolveSchemaForAgent(client: any, agentPasscode: string, schemaSaid: string, env: 'docker' | 'testnet'): Promise<void> {
    const schemaOobi = `${SCHEMA_SERVER}/oobi/${schemaSaid}`;
    console.log(`\n📋 Resolving schema: ${schemaSaid}`);

    // Try 1: Direct with agent client
    try {
        await retry(() => resolveOobi(client, schemaOobi, `schema-${Date.now()}`), 'Direct', 2, 2000);
        console.log(`  ✓ Resolved directly`);
        return;
    } catch (e) { /* continue */ }

    // Try 2: Check cache
    try {
        const schemas = await client.schemas().list();
        if (schemas.find((s: any) => s.$id === schemaSaid)) {
            console.log(`  ✓ Already in cache`);
            return;
        }
    } catch (e) { /* continue */ }

    // Try 3: Use shared client first
    console.log(`  Using shared client for schema resolution...`);
    try {
        const shared = await getOrCreateClient(SHARED_PASSCODE, env);
        await retry(() => resolveOobi(shared, schemaOobi, 'schema-shared'), 'Shared', 2, 2000);
        await retry(() => resolveOobi(client, schemaOobi, 'schema-agent'), 'Agent', 2, 2000);
        console.log(`  ✓ Resolved via shared client`);
        return;
    } catch (e) { /* continue */ }

    // Try 4: Direct fetch
    try {
        const schema = await client.schemas().get(schemaSaid);
        if (schema?.$id === schemaSaid) {
            console.log(`  ✓ Fetched from KERIA cache`);
            return;
        }
    } catch (e) { /* continue */ }

    throw new Error(`Failed to resolve schema ${schemaSaid}`);
}

// ============================================================================
// MAIN
// ============================================================================

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const agentAidName = args[1];      // e.g., "jupiterSellerAgent"
const passcode = args[2];          // Agent's passcode or empty to use BRAN file
const registryName = args[3];      // e.g., "jupiterSellerAgent-registry"
const schemaSaidArg = args[4];     // Optional override
const invoiceDataJson = args[5];   // JSON string with invoice data
const outputPath = args[6];        // Where to save credential info
const taskDataDir = args[7] || '/task-data';

console.log(`\n╔════════════════════════════════════════════════════════════╗`);
console.log(`║   SELF-ATTESTED INVOICE CREDENTIAL TO DELEGATED AGENT     ║`);
console.log(`╚════════════════════════════════════════════════════════════╝\n`);

// Get agent's passcode
let agentPasscode = passcode;
if (!passcode || passcode.trim() === '') {
    const branFile = `${taskDataDir}/${agentAidName}-bran.txt`;
    if (fs.existsSync(branFile)) {
        agentPasscode = fs.readFileSync(branFile, 'utf-8').trim();
        console.log(`✓ Loaded agent BRAN from ${branFile}`);
    } else {
        console.error(`✗ No passcode and no BRAN file at ${branFile}`);
        process.exit(1);
    }
}

// Parse invoice data
const invoiceData = JSON.parse(invoiceDataJson);
const schemaSaid = (schemaSaidArg && schemaSaidArg.startsWith('E')) ? schemaSaidArg : INVOICE_SCHEMA_SAID;

console.log(`\nConfiguration:`);
console.log(`  Agent: ${agentAidName}`);
console.log(`  Registry: ${registryName}`);
console.log(`  Schema: ${schemaSaid}`);
console.log(`  Invoice #: ${invoiceData.invoiceNumber}`);

// Connect to KERIA with agent's credentials
console.log(`\n🔗 Connecting to KERIA...`);
const client = await retry(() => getOrCreateClient(agentPasscode, env), 'KERIA', 5, 5000);
console.log(`  ✓ Connected`);

// Get agent AID
const agentAid = await client.identifiers().get(agentAidName);
const agentPrefix = agentAid.prefix;
console.log(`  ✓ Agent AID: ${agentPrefix}`);

// Resolve schema
await resolveSchemaForAgent(client, agentPasscode, schemaSaid, env);

// Get registry
const registries = await client.registries().list(agentAidName);
const registry = registries.find((r: any) => r.name === registryName);
if (!registry) {
    console.error(`\n✗ Registry '${registryName}' not found`);
    console.log(`  Available: ${registries.map((r: any) => r.name).join(', ')}`);
    process.exit(1);
}
console.log(`\n✓ Registry: ${registry.regk}`);

// ============================================================================
// ISSUE SELF-ATTESTED CREDENTIAL
// ============================================================================

console.log(`\n🎫 Issuing SELF-ATTESTED credential...`);
console.log(`  Issuer: ${agentPrefix}`);
console.log(`  Issuee: ${agentPrefix} (SAME - self-attested)`);
console.log(`  Edges: NONE (self-attested)`);

const credentialData = {
    i: agentPrefix,        // Issuer = Agent
    ri: registry.regk,     // Agent's registry
    s: schemaSaid,         // Invoice schema
    a: {                   // Attributes
        i: agentPrefix,    // Issuee = Agent (SELF-ATTESTED!)
        dt: createTimestamp(),
        ...invoiceData,
    },
    e: undefined,          // NO EDGES - this is what makes it self-attested
    r: {                   // Rules
        d: '',
        usageDisclaimer: {
            l: 'This is a self-attested invoice credential. The issuer and issuee are the same entity (the delegated agent).'
        },
        selfAttestation: {
            l: 'Trust in this credential derives from the agent delegation chain: GEDA → QVI → LE → OOR Holder → Agent. The agent is authorized to issue self-attested business credentials.'
        }
    },
};

// Issue
const issResult = await retry(async () => {
    const result = await client.credentials().issue(agentAidName, credentialData);
    const op = await waitOperation(client, result.op);
    if (op.error) throw new Error(`Issuance failed: ${JSON.stringify(op.error)}`);
    return { ...result, response: op.response };
}, 'Issuance', 3, 5000);

const credSaid = issResult.response?.ced?.d;
if (!credSaid) {
    console.error(`\n✗ No credential SAID returned`);
    process.exit(1);
}

// Verify
const cred = await client.credentials().get(credSaid);
const issuer = cred.sad.i;
const issuee = cred.sad?.a?.i;

console.log(`\n╔════════════════════════════════════════════════════════════╗`);
console.log(`║              CREDENTIAL ISSUED SUCCESSFULLY                ║`);
console.log(`╚════════════════════════════════════════════════════════════╝`);
console.log(`  SAID: ${credSaid}`);
console.log(`  Issuer: ${issuer}`);
console.log(`  Issuee: ${issuee}`);
console.log(`  Self-Attested: ${issuer === issuee ? '✅ YES' : '❌ NO'}`);

if (issuer !== issuee) {
    console.error(`\n⚠️  WARNING: Not self-attested! Issuer ≠ Issuee`);
}

// Save output
const output = {
    said: credSaid,
    issuer,
    issuee,
    selfAttested: issuer === issuee,
    hasEdge: false,
    schema: schemaSaid,
    registry: registry.regk,
    agentName: agentAidName,
    invoiceNumber: invoiceData.invoiceNumber,
    totalAmount: invoiceData.totalAmount,
    currency: invoiceData.currency,
    dueDate: invoiceData.dueDate,
    paymentMethod: invoiceData.paymentMethod,
    paymentChainID: invoiceData.paymentChainID,
    paymentWalletAddress: invoiceData.paymentWalletAddress,
    ref_uri: invoiceData.ref_uri,
    // For IPEX if needed
    acdc: issResult.acdc,
    anc: issResult.anc,
    iss: issResult.iss,
};

await fs.promises.writeFile(outputPath, JSON.stringify(output, null, 2));
console.log(`\n📄 Saved to: ${outputPath}`);

console.log(`\n✅ Self-attested invoice credential issued to ${agentAidName}`);
console.log(`   The agent can now present this credential to verifiers.`);
