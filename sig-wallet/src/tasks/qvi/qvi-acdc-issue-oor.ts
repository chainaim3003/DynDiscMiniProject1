import fs from "fs";
import {getOrCreateClient} from "../../client/identifiers.js";
import {issueCredential, ipexGrantCredential} from "../../client/credentials.js";
import {resolveOobi} from "../../client/oobis.js";

// process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const qviAidName = args[1];
const passcode = args[2];
const registryName = args[3];
const oorSchemaSaid = args[4];
const personAid = args[5];
const personName = args[6];
const personOor = args[7];
const oorAuthCredSaid = args[8];
const leLei = args[9];
const oorCredentialInfoPath = args[10];

// get client and issue OOR ACDC
const client = await getOrCreateClient(passcode, env);

// Resolve schema OOBI
console.log(`Resolving schema OOBI http://schema:7723/oobi/${oorSchemaSaid}`);
await resolveOobi(client, `http://schema:7723/oobi/${oorSchemaSaid}`, 'schema');

// Construct the edge for the OOR credential
console.log(`Using OOR Auth credential SAID for edge: ${oorAuthCredSaid}`);
const credEdge = {
    d: '', // SAID will be calculated by KERIpy
    auth: {
        n: oorAuthCredSaid,
        s: 'EKA57bKBKxr_kN7iN5i7lMUxpMG-s19dRcmov1iDxz-E', // OOR Auth Schema SAID
        o: 'I2I' // Operator indicating this node is the issuer
    }
};

// Construct the rules for the OOR credential
const credRules = {
    d: '', // SAID will be calculated by KERIpy
    usageDisclaimer: {
        l: 'Usage of a valid, unexpired, and non-revoked vLEI Credential, as defined in the associated Ecosystem Governance Framework, does not assert that the Legal Entity is trustworthy, honest, reputable in its business dealings, safe to do business with, or compliant with any laws or that an implied or expressly intended purpose will be fulfilled.'
    },
    issuanceDisclaimer: {
        l: 'All information in a valid, unexpired, and non-revoked vLEI Credential, as defined in the associated Ecosystem Governance Framework, is accurate as of the date the validation process was complete. The vLEI Credential has been issued to the legal entity or person named in the vLEI Credential as the subject; and the qualified vLEI Issuer exercised reasonable care to perform the validation process set forth in the vLEI Ecosystem Governance Framework.'
    }
};

const oorCredData = {
    LEI: leLei, // LEI from configuration
    personLegalName: personName,
    officialRole: personOor
};

const {said, issuer, issuee, acdc, anc, iss} = await issueCredential(
    client,
    qviAidName,
    registryName,
    oorSchemaSaid,
    personAid,
    oorCredData,
    credEdge,
    credRules
);
console.log(`OOR credential created: ${said}`);

const grantOp = await ipexGrantCredential(client, personAid, qviAidName, acdc, anc, iss);
console.log("Grant result:", grantOp);

await fs.promises.writeFile(oorCredentialInfoPath, JSON.stringify({said, issuer, issuee, grantSaid: grantOp.response.said}));
console.log(`OOR credential info written to ${oorCredentialInfoPath}`)