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
const ecrAuthSchemaSaid = args[4];
const qviAid = args[5];
const personAid = args[6];
const personName = args[7];
const personEcr = args[8];
const leCredSaid = args[9];
const ecrAuthCredentialInfoPath = args[10];

// get client and issue ECR Auth ACDC
const client = await getOrCreateClient(passcode, env);

// Resolve schema OOBI
console.log(`Resolving schema OOBI http://schema:7723/oobi/${ecrAuthSchemaSaid}`);
await resolveOobi(client, `http://schema:7723/oobi/${ecrAuthSchemaSaid}`, 'schema');

// Construct the edge for the ECR Auth credential
console.log(`Using LE credential SAID for edge: ${leCredSaid}`);
const credEdge = {
    d: '', // SAID will be calculated by KERIpy
    le: {
        n: leCredSaid,
        s: 'ENPXp1vQzRF6JwIuS-mp2U8Uf1MoADoP_GqQ62VsDZWY' // LE Schema SAID
    }
};

// Construct the rules for the ECR Auth credential
const credRules = {
    d: '', // SAID will be calculated by KERIpy
    usageDisclaimer: {
        l: 'Usage of a valid, unexpired, and non-revoked vLEI Credential, as defined in the associated Ecosystem Governance Framework, does not assert that the Legal Entity is trustworthy, honest, reputable in its business dealings, safe to do business with, or compliant with any laws or that an implied or expressly intended purpose will be fulfilled.'
    },
    issuanceDisclaimer: {
        l: 'All information in a valid, unexpired, and non-revoked vLEI Credential, as defined in the associated Ecosystem Governance Framework, is accurate as of the date the validation process was complete. The vLEI Credential has been issued to the legal entity or person named in the vLEI Credential as the subject; and the qualified vLEI Issuer exercised reasonable care to perform the validation process set forth in the vLEI Ecosystem Governance Framework.'
    },
    privacyDisclaimer: {
        l: 'Privacy Considerations are applicable to QVI ECR AUTH vLEI Credentials.  It is the sole responsibility of QVIs as Issuees of QVI ECR AUTH vLEI Credentials to present these Credentials in a privacy-preserving manner using the mechanisms provided in the Issuance and Presentation Exchange (IPEX) protocol specification and the Authentic Chained Data Container (ACDC) specification.  https://github.com/WebOfTrust/IETF-IPEX and https://github.com/trustoverip/tswg-acdc-specification.'
    }
};

const ecrAuthCredData = {
    AID: personAid,
    LEI: '254900OPPU84GM83MG36', // Same LEI as LE credential
    personLegalName: personName,
    engagementContextRole: personEcr
};

const {said, issuer, issuee, acdc, anc, iss} = await issueCredential(
    client,
    leAidName,
    registryName,
    ecrAuthSchemaSaid,
    qviAid,
    ecrAuthCredData,
    credEdge,
    credRules
);
console.log(`ECR Auth credential created: ${said}`);

const grantOp = await ipexGrantCredential(client, qviAid, leAidName, acdc, anc, iss);
console.log("Grant result:", grantOp);

await fs.promises.writeFile(ecrAuthCredentialInfoPath, JSON.stringify({said, issuer, issuee, grantSaid: grantOp.response.said}));
console.log(`ECR Auth credential info written to ${ecrAuthCredentialInfoPath}`)
