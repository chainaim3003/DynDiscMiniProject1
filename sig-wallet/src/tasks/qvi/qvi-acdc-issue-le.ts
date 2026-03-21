import fs from "fs";
import {resolveEnvironment} from "../../client/resolve-env.js";
import {getOrCreateClient} from "../../client/identifiers.js";
import {issueCredential, ipexGrantCredential} from "../../client/credentials.js";
import {resolveOobi} from "../../client/oobis.js";

// process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const aidName = args[1];
const passcode = args[2];
const leAid = args[3];
const leLei = args[4];
const credentialInfoPath = args[5];

// LE Schema SAID
const LE_SCHEMA_SAID = 'ENPXp1vQzRF6JwIuS-mp2U8Uf1MoADoP_GqQ62VsDZWY';

// QVI Schema SAID (for edge)
const QVI_SCHEMA_SAID = 'EBfdlu8R27Fbx-ehrqwImnK-8Cm79sqbAQ4MmvEAYqao';

// get client
const client = await getOrCreateClient(passcode, env);

// Resolve schema OOBI
const schemaOobi = `http://schema:7723/oobi/${LE_SCHEMA_SAID}`;
console.log(`Resolving schema OOBI ${schemaOobi}`);
await resolveOobi(client, schemaOobi);

// Get QVI credential SAID for edge
const qviCredentials = await client.credentials().list({
    filter: {
        '-s': QVI_SCHEMA_SAID
    }
});

if (qviCredentials.length === 0) {
    throw new Error('QVI credential not found. Please ensure QVI credential has been issued and admitted.');
}

const qviCredentialSaid = qviCredentials[0].sad.d;
console.log(`Using QVI credential SAID for edge: ${qviCredentialSaid}`);

// Prepare credential data
const credData = {
    LEI: leLei
};

// Prepare edge data (QVI credential reference)
const edgeData = {
    d: "", // Will be SAIDified
    qvi: {
        n: qviCredentialSaid,
        s: QVI_SCHEMA_SAID
    }
};

// Prepare rules data
const rulesData = {
    d: "", // Will be SAIDified
    usageDisclaimer: {
        l: "Usage of a valid, unexpired, and non-revoked vLEI Credential, as defined in the associated Ecosystem Governance Framework, does not assert that the Legal Entity is trustworthy, honest, reputable in its business dealings, safe to do business with, or compliant with any laws or that an implied or expressly intended purpose will be fulfilled."
    },
    issuanceDisclaimer: {
        l: "All information in a valid, unexpired, and non-revoked vLEI Credential, as defined in the associated Ecosystem Governance Framework, is accurate as of the date the validation process was complete. The vLEI Credential has been issued to the legal entity or person named in the vLEI Credential as the subject; and the qualified vLEI Issuer exercised reasonable care to perform the validation process set forth in the vLEI Ecosystem Governance Framework."
    }
};

console.log(`Issuing LE credential schema ${LE_SCHEMA_SAID} to ${leAid}`);
console.log("Cred data:", credData);

// Issue the credential
const credentialResult = await issueCredential(
    client,
    aidName,
    'qvi-le-registry',
    LE_SCHEMA_SAID,
    leAid,
    credData,
    edgeData,
    rulesData
);

console.log(`LE credential created: ${credentialResult.said}`);

// Grant the credential via IPEX
console.log(`IPEX Granting credential to issuee: ${leAid}`);
const grantResult = await ipexGrantCredential(
    client,
    leAid,
    aidName,
    credentialResult.acdc,
    credentialResult.anc,
    credentialResult.iss
);

console.log('Grant message sent');
console.log('Grant result:', grantResult);

// Save credential info
const credentialInfo = {
    said: credentialResult.said,
    issuer: credentialResult.issuer,
    issuee: credentialResult.issuee,
    grantSaid: grantResult.response.said
};

await fs.promises.writeFile(credentialInfoPath, JSON.stringify(credentialInfo, null, 2));
console.log(`LE credential info written to ${credentialInfoPath}`);
