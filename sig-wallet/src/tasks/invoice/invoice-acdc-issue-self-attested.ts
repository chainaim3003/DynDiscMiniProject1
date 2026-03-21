import fs from "fs";
import { getOrCreateClient } from "../../client/identifiers.js";
import { issueCredential, ipexGrantCredential } from "../../client/credentials.js";
import { resolveOobi } from "../../client/oobis.js";

// Process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const issuerAidName = args[1]; // jupiterSalesAgent (self-attested)
const passcode = args[2];
const registryName = args[3];
const invoiceSchemaSaid = args[4];
const recipientAid = args[5]; // tommyBuyerAgent (will receive via IPEX)
const invoiceDataJson = args[6];
const outputPath = args[7];

console.log(`Issuing self-attested invoice credential...`);
console.log(`  Issuer (self-attested): ${issuerAidName}`);
console.log(`  Recipient (IPEX): ${recipientAid}`);
console.log(`  Registry: ${registryName}`);

// Get client
const client = await getOrCreateClient(passcode, env);

// Get issuer AID
const issuerAid = await client.identifiers().get(issuerAidName);
const issuerPrefix = issuerAid.prefix;

console.log(`  Issuer AID: ${issuerPrefix}`);

// Resolve schema OOBI
console.log(`Resolving invoice schema OOBI...`);
await resolveOobi(client, `http://schema:7723/oobi/${invoiceSchemaSaid}`, 'schema');

// Parse invoice data
const invoiceData = JSON.parse(invoiceDataJson);

// NO EDGE SECTION - Self-attested credential
// Edge section is optional for self-attested credentials
const credEdge = undefined;

// Construct rules
const credRules = {
    d: '', // SAID will be calculated by KERIpy
    usageDisclaimer: {
        l: 'This is a self-attested invoice credential issued by the agent. The issuer and holder are the same entity. This credential represents a business transaction.'
    },
    invoiceTerms: {
        l: `Payment is due by ${invoiceData.dueDate}. All amounts are in ${invoiceData.currency}. This invoice is self-attested by ${issuerAidName}.`
    }
};

console.log(`Issuing self-attested credential to registry ${registryName}...`);
console.log(`  Issuer = Issuee = ${issuerPrefix} (self-attested)`);

// Issue credential - issuer and holder are the SAME (self-attested)
const { said, issuer, issuee, acdc, anc, iss } = await issueCredential(
    client,
    issuerAidName,
    registryName,
    invoiceSchemaSaid,
    issuerPrefix, // SAME as issuer - self-attested!
    invoiceData,
    credEdge,
    credRules
);
console.log(`✓ Self-attested invoice credential created: ${said}`);
console.log(`  Issuer: ${issuer}`);
console.log(`  Issuee: ${issuee}`);
console.log(`  Self-attested: ${issuer === issuee ? 'YES' : 'NO'}`);

// Grant to recipient (tommyBuyerAgent) via IPEX
console.log(``);
console.log(`Granting credential to ${recipientAid} via IPEX...`);
const grantOp = await ipexGrantCredential(client, recipientAid, issuerAidName, acdc, anc, iss);
console.log(`✓ Invoice credential granted to recipient`);
console.log(`  Grant SAID: ${grantOp.response.said}`);

// Save credential info
const credInfo = {
    said,
    issuer,
    issuee,
    selfAttested: issuer === issuee,
    grantSaid: grantOp.response.said,
    recipientAid: recipientAid,
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
console.log(`✓ Self-attested invoice credential info saved to ${outputPath}`);
