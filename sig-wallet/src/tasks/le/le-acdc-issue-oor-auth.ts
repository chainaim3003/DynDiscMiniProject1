import fs from "fs";
import {getOrCreateClient} from "../../client/identifiers.js";
import {issueCredential, ipexGrantCredential} from "../../client/credentials.js";
import {resolveOobi} from "../../client/oobis.js";

// process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const leAidName = args[1];
const passcode = args[2];
const registryName = args[3];
const oorAuthSchemaSaid = args[4];
const qviAid = args[5];
const personAid = args[6];
const personName = args[7];
const personOor = args[8];
const leCredSaid = args[9];
const leLei = args[10];
const oorAuthCredentialInfoPath = args[11];

// get client and issue OOR Auth ACDC
const client = await getOrCreateClient(passcode, env);

// Resolve schema OOBI
console.log(`Resolving schema OOBI http://schema:7723/oobi/${oorAuthSchemaSaid}`);
await resolveOobi(client, `http://schema:7723/oobi/${oorAuthSchemaSaid}`, 'schema');

// Construct the edge for the OOR Auth credential
console.log(`Using LE credential SAID for edge: ${leCredSaid}`);
const credEdge = {
    d: '', // SAID will be calculated by KERIpy
    le: {
        n: leCredSaid,
        s: 'ENPXp1vQzRF6JwIuS-mp2U8Uf1MoADoP_GqQ62VsDZWY' // LE Schema SAID
    }
};

// Construct the rules for the OOR Auth credential
const credRules = {
    d: '', // SAID will be calculated by KERIpy
    usageDisclaimer: {
        l: 'Usage of a valid, unexpired, and non-revoked vLEI Credential, as defined in the associated Ecosystem Governance Framework, does not assert that the Legal Entity is trustworthy, honest, reputable in its business dealings, safe to do business with, or compliant with any laws or that an implied or expressly intended purpose will be fulfilled.'
    },
    issuanceDisclaimer: {
        l: 'All information in a valid, unexpired, and non-revoked vLEI Credential, as defined in the associated Ecosystem Governance Framework, is accurate as of the date the validation process was complete. The vLEI Credential has been issued to the legal entity or person named in the vLEI Credential as the subject; and the qualified vLEI Issuer exercised reasonable care to perform the validation process set forth in the vLEI Ecosystem Governance Framework.'
    }
};

const oorAuthCredData = {
    AID: personAid,
    LEI: leLei, // LEI from configuration
    personLegalName: personName,
    officialRole: personOor
};

const {said, issuer, issuee, acdc, anc, iss} = await issueCredential(
    client,
    leAidName,
    registryName,
    oorAuthSchemaSaid,
    qviAid,
    oorAuthCredData,
    credEdge,
    credRules
);
console.log(`OOR Auth credential created: ${said}`);

const grantOp = await ipexGrantCredential(client, qviAid, leAidName, acdc, anc, iss);
console.log("Grant result:", grantOp);

await fs.promises.writeFile(oorAuthCredentialInfoPath, JSON.stringify({said, issuer, issuee, grantSaid: grantOp.response.said}));
console.log(`OOR Auth credential info written to ${oorAuthCredentialInfoPath}`)

