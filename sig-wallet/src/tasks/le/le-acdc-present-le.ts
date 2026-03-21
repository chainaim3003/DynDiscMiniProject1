import {getOrCreateClient} from "../../client/identifiers.js";
import {grantCredential} from "../../client/credentials.js";

// process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1]
const senderAidName = args[2]
const credentialSAID = args[3]
const recipientPrefix = args[4]

const client = await getOrCreateClient(passcode, env);
const granted: string = await grantCredential(client, senderAidName, credentialSAID, recipientPrefix);
console.log(granted);
