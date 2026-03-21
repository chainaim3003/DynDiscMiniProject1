import { getOrCreateClient } from "../../client/identifiers.js";
import { presentCredential } from "../../client/credentials.js";
import { waitOperation } from "../../client/identifiers.js";

// Process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const holderAidName = args[1];
const passcode = args[2];
const credentialSaid = args[3];

console.log(`Presenting invoice credential to Sally verifier...`);
console.log(`  Holder: ${holderAidName}`);
console.log(`  Credential SAID: ${credentialSaid}`);

// Get client
const client = await getOrCreateClient(passcode, env);

// Present credential to Sally verifier
const verifierAidName = 'verifier';

console.log(`Presenting to verifier: ${verifierAidName}...`);
const presentOp = await presentCredential(
    client,
    holderAidName,
    verifierAidName,
    credentialSaid
);

await waitOperation(client, presentOp);
console.log(`✓ Invoice credential presented successfully to Sally verifier`);
