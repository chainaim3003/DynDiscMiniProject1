import {getOrCreateClient} from "../../client/identifiers.js";
import { ipexAdmitGrant } from "../../client/credentials.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const lePasscode = args[1];
const grantSAID = args[2];
const qviPrefix = args[3];
const leAlias = args[4] || 'le';  // Use provided alias or default to 'le'

const leClient = await getOrCreateClient(lePasscode, env);

const op: any = await ipexAdmitGrant(leClient, leAlias, qviPrefix, grantSAID)
const creds = await leClient.credentials().list();
console.log("LE IPEX Admit:", op.operation?.response?.said || "admitted successfully");
console.log("LE credential admitted:", creds[0].sad.d);
