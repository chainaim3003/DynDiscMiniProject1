import fs from "fs";
import {getOrCreateClient} from "../../client/identifiers.js";
import {ipexGrantCredential} from "../../client/credentials.js";
import {resolveOobi} from "../../client/oobis.js";
import {waitOperation} from "../../client/operations.js";
import {SignifyClient, CredentialSubject, CredentialData, Serder, Salter} from "signify-ts";
import {createTimestamp} from "../../time.js";

// process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const qviAidName = args[1];
const passcode = args[2];
const registryName = args[3];
const ecrSchemaSaid = args[4];
const personAid = args[5];
const personName = args[6];
const personEcr = args[7];
const ecrAuthCredSaid = args[8];
const ecrCredentialInfoPath = args[9];

// get client and issue ECR ACDC
const client = await getOrCreateClient(passcode, env);

// Resolve schema OOBI
console.log(`Resolving schema OOBI http://schema:7723/oobi/${ecrSchemaSaid}`);
await resolveOobi(client, `http://schema:7723/oobi/${ecrSchemaSaid}`, 'schema');

// Get the QVI AID and registry
const qviAid = await client.identifiers().get(qviAidName);
const registries = await client.registries().list(qviAidName);
const registry = registries.find((reg) => reg.name === registryName);

if (!registry) {
    throw new Error(`Registry ${registryName} not found`);
}

// Construct the edge for the ECR credential
console.log(`Using ECR Auth credential SAID for edge: ${ecrAuthCredSaid}`);
const credEdge = {
    d: '', // SAID will be calculated by KERIpy
    auth: {
        n: ecrAuthCredSaid,
        s: 'EH6ekLjSr8V32WyFbGe1zXjTzFs9PkTYmupJ9H65O14g', // ECR Auth Schema SAID
        o: 'I2I' // Operator indicating this node is the issuer
    }
};

// Construct the rules for the ECR credential
const credRules = {
    d: '', // SAID will be calculated by KERIpy
    usageDisclaimer: {
        l: 'Usage of a valid, unexpired, and non-revoked vLEI Credential, as defined in the associated Ecosystem Governance Framework, does not assert that the Legal Entity is trustworthy, honest, reputable in its business dealings, safe to do business with, or compliant with any laws or that an implied or expressly intended purpose will be fulfilled.'
    },
    issuanceDisclaimer: {
        l: 'All information in a valid, unexpired, and non-revoked vLEI Credential, as defined in the associated Ecosystem Governance Framework, is accurate as of the date the validation process was complete. The vLEI Credential has been issued to the legal entity or person named in the vLEI Credential as the subject; and the qualified vLEI Issuer exercised reasonable care to perform the validation process set forth in the vLEI Ecosystem Governance Framework.'
    },
    privacyDisclaimer: {
        l: 'It is the sole responsibility of Holders as Issuees of an ECR vLEI Credential to present that Credential in a privacy-preserving manner using the mechanisms provided in the Issuance and Presentation Exchange (IPEX) protocol specification and the Authentic Chained Data Container (ACDC) specification. https://github.com/WebOfTrust/IETF-IPEX and https://github.com/trustoverip/tswg-acdc-specification.'
    }
};

// Create the credential subject with u field
const kargsSub: CredentialSubject = {
    i: personAid,
    dt: createTimestamp(),
    u: new Salter({}).qb64,
    LEI: '254900OPPU84GM83MG36', // Same LEI as LE credential
    personLegalName: personName,
    engagementContextRole: personEcr
};

// Create the credential data with top-level u field
const kargsIss: CredentialData = {
    u: new Salter({}).qb64,
    i: qviAid.prefix,
    ri: registry.regk,
    s: ecrSchemaSaid,
    a: kargsSub,
    e: credEdge,
    r: credRules,
};

console.log(`Credential data:`, kargsIss);

// Issue the credential directly using KERI
const issResult = await client.credentials().issue(qviAidName, kargsIss);
const issueOp = await waitOperation(client, issResult.op);
if (issueOp.error) {
    throw new Error(`Credential issuance failed: ${JSON.stringify(issueOp.error)}`);
}

const credentialSad: any = issueOp.response; // The full Self-Addressing Data (SAD) of the credential
const credentialSaid = credentialSad?.ced?.d; // The SAID of the credential

const cred: any = await client.credentials().get(credentialSaid);
const said = cred.sad.d;
const issuer = cred.sad.i;
const issuee = cred.sad.a.i;
const acdc = issResult.acdc;
const anc = issResult.anc;
const iss = issResult.iss;

console.log(`ECR credential created: ${said}`);

const grantOp = await ipexGrantCredential(client, personAid, qviAidName, acdc, anc, iss);
console.log("Grant result:", grantOp);

await fs.promises.writeFile(ecrCredentialInfoPath, JSON.stringify({said, issuer, issuee, grantSaid: grantOp.response.said}));
console.log(`ECR credential info written to ${ecrCredentialInfoPath}`)
