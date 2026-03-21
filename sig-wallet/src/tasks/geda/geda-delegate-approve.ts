import {approveDelegation, getOrCreateClient} from "../../client/identifiers.js";
import fs from "fs";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const delegatorPasscode = args[1];
const delegatorName = args[2];
const delegateInfoFilePath = args[3];

// Get delegator client and approve delegation
const dgrClient = await getOrCreateClient(delegatorPasscode, env);
const delegateInfo = JSON.parse(await fs.promises.readFile(delegateInfoFilePath, 'utf-8'));
const delegatorApproved = await approveDelegation(dgrClient, delegatorName, delegateInfo.aid);
console.log(`Delegator ${delegatorName} approved delegation of ${delegateInfo.aid}: ${delegatorApproved}`);
