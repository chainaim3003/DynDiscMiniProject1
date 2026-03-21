import {getOrCreateClient} from "../../client/identifiers.js";
import { ipexAdmitGrant } from "../../client/credentials.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const qviPasscode = args[1];
const grantSAID = args[2];
const lePrefix = args[3];

const qviClient = await getOrCreateClient(qviPasscode, env);

const op: any = await ipexAdmitGrant(qviClient, 'qvi', lePrefix, grantSAID)
const creds = await qviClient.credentials().list();
console.log("QVI IPEX Admit:", op.operation?.response?.said || "admitted successfully");
console.log("ECR Auth credential admitted:", creds[0].sad.d);
