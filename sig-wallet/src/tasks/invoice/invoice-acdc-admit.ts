import { getOrCreateClient } from "../../client/identifiers.js";
import { ipexAdmitCredential } from "../../client/credentials.js";
import { waitForAndGetNotification } from "../../client/notifications.js";

// Process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const holderAidName = args[1];
const passcode = args[2];

console.log(`Admitting invoice credential...`);
console.log(`  Holder: ${holderAidName}`);

// Get client
const client = await getOrCreateClient(passcode, env);

// Wait for IPEX grant notification
console.log(`Waiting for invoice credential grant notification...`);
const IPEX_GRANT_ROUTE = `/identifiers/${holderAidName}/ipex/grant`;
const notifications = await waitForAndGetNotification(client, IPEX_GRANT_ROUTE);

if (notifications.length === 0) {
    throw new Error('No grant notifications received for invoice credential');
}

// Get the credential from notification
const grantNotification = notifications[0];
const credentialSaid = grantNotification.a.credential.sad.d;
console.log(`Received invoice credential: ${credentialSaid}`);

// Admit the credential
console.log(`Admitting invoice credential...`);
const admitOp = await ipexAdmitCredential(client, holderAidName, grantNotification.i);
console.log(`✓ Invoice credential admitted successfully`);
console.log(`  Admit operation SAID: ${admitOp.response.said}`);
