import { getOrCreateClient } from "../../client/identifiers.js";
const args = process.argv.slice(2);
const env = args[0];
const passcode = args[1];
const aidName = args[2];
async function getOobi(client, name) {
    const oobiResp = await client.oobis().get(name, 'agent');
    return oobiResp.oobis[0];
}
const client = await getOrCreateClient(passcode, env);
const oobi = await getOobi(client, aidName);
console.log(`Agent OOBI for Identifier ${aidName}:`, oobi);
