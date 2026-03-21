import {getOrCreateClient} from "../../client/identifiers.js";
import {resolveOobi} from "../../client/oobis.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1];
const oobi = args[2];

const client = await getOrCreateClient(passcode, env);
await resolveOobi(client, oobi)
console.log(`OOBI Resolved: ${oobi}`);