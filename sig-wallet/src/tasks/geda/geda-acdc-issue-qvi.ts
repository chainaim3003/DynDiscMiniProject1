import fs from "fs";
import {getOrCreateClient} from "../../client/identifiers.js";
import {ipexGrantCredential, issueCredential} from "../../client/credentials.js";
import {resolveOobi} from "../../client/oobis.js";

// process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const issAidName = args[1];
const issPasscode = args[2];
const registry = args[3];
const schema = args[4];
const hldPrefix = args[5];
const lei = args[6];
const qviCredInfoPath = args[7];


// get client
const issClient = await getOrCreateClient(issPasscode, env);

// resolve schema OOBI
console.log(`Resolving schema OOBI http://schema:7723/oobi/${schema}`);
await resolveOobi(issClient, `http://schema:7723/oobi/${schema}`, 'qvi-schema')

// prepare credential data
let credData  = {
    "LEI": `${lei}`
}

// issue credential
console.log(`Issuing QVI credential schema ${schema} to ${hldPrefix}`);
const issRes: any = await issueCredential(
    issClient,
    issAidName,
    registry,
    schema,
    hldPrefix,
    credData
);
// console.log(issRes);
console.log("IPEX Granting credential to issuee:", issRes.issuee);
const grantRes: any = await ipexGrantCredential(
    issClient,
    issRes.issuee,
    issAidName,
    issRes.acdc,
    issRes.anc,
    issRes.iss
);
console.log("Grant result:", grantRes);
issRes.grantSAID = grantRes.response.said;
await fs.promises.writeFile(qviCredInfoPath, JSON.stringify(issRes));
console.log(grantRes);
console.log("QVI credential created and granted");
