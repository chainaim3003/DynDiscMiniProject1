import {getOrCreateClient} from "../../client/identifiers.js";
import { ipexAdmitGrant } from "../../client/credentials.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const personPasscode = args[1];
const grantSAID = args[2];
const qviPrefix = args[3];

const personClient = await getOrCreateClient(personPasscode, env);

const op: any = await ipexAdmitGrant(personClient, 'person', qviPrefix, grantSAID)
const creds = await personClient.credentials().list();
console.log("Person IPEX Admit:", op.operation?.response?.said || "admitted successfully");
console.log("ECR credential admitted:", creds[0].sad.d);
