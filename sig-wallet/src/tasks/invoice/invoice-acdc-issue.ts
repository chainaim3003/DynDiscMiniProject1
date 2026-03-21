import fs from "fs";
import { getOrCreateClient } from "../../client/identifiers.js";
import { issueCredential, ipexGrantCredential } from "../../client/credentials.js";
import { resolveOobi } from "../../client/oobis.js";

// Process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const issuerAidName = args[1];
const passcode = args[2];
const registryName = args[3];
const invoiceSchemaSaid = args[4];
const holderAid = args[5];
const oorCredentialSaid = args[6];
const oorSchemaSaid = args[7];
const invoiceDataJson = args[8];
const outputPath = args[9];

console.log(`Issuing invoice credential...`);
console.log(`  Issuer: ${issuerAidName}`);
console.log(`  Holder AID: ${holderAid}`);
console.log(`  OOR Credential SAID: ${oorCredentialSaid}`);
console.log(`  Registry: ${registryName}`);

// Get client
const client = await getOrCreateClient(passcode, env);

// Resolve schema OOBI
console.log(`Resolving invoice schema OOBI...`);
await resolveOobi(client, `http://schema:7723/oobi/${invoiceSchemaSaid}`, 'schema');

// Parse invoice data
const invoiceData = JSON.parse(invoiceDataJson);

// Construct edge to OOR credential
console.log(`Creating edge to OOR credential: ${oorCredentialSaid}`);
const credEdge = {
    d: '', // SAID will be calculated by KERIpy
    oor: {
        n: oorCredentialSaid,
        s: oorSchemaSaid,
        o: 'I2I' // Issuer-to-issuer chain
    }
};

// Construct rules
const credRules = {
    d: '', // SAID will be calculated by KERIpy
    usageDisclaimer: {
        l: 'This invoice credential is issued based on the authority granted by the OOR credential. Payment obligations are subject to the terms specified in the invoice and applicable law.'
    },
    invoiceTerms: {
        l: `Payment is due by ${invoiceData.dueDate}. Late payments may incur interest charges. All amounts are in ${invoiceData.currency}.`
    }
};

console.log(`Issuing credential to registry ${registryName}...`);

// Issue credential
const { said, issuer, issuee, acdc, anc, iss } = await issueCredential(
    client,
    issuerAidName,
    registryName,
    invoiceSchemaSaid,
    holderAid,
    invoiceData,
    credEdge,
    credRules
);
console.log(`✓ Invoice credential created: ${said}`);

// Grant to holder via IPEX
console.log(`Granting credential to ${holderAid} via IPEX...`);
const grantOp = await ipexGrantCredential(client, holderAid, issuerAidName, acdc, anc, iss);
console.log(`✓ Invoice credential granted`);
console.log(`  Grant SAID: ${grantOp.response.said}`);

// Save credential info
const credInfo = {
    said,
    issuer,
    issuee,
    grantSaid: grantOp.response.said,
    invoiceNumber: invoiceData.invoiceNumber,
    totalAmount: invoiceData.totalAmount,
    currency: invoiceData.currency,
    dueDate: invoiceData.dueDate,
    paymentMethod: invoiceData.paymentMethod,
    paymentChainID: invoiceData.paymentChainID,
    paymentWalletAddress: invoiceData.paymentWalletAddress,
    ref_uri: invoiceData.ref_uri,
    paymentTerms: invoiceData.paymentTerms || null
};

await fs.promises.writeFile(outputPath, JSON.stringify(credInfo, null, 2));
console.log(`✓ Invoice credential info saved to ${outputPath}`);
