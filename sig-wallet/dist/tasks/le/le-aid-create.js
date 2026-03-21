import fs from 'fs';
import { createAid, getOrCreateClient } from "../../client/identifiers.js";
const args = process.argv.slice(2);
const env = args[0];
const lePasscode = args[1];
const dataDir = args[2];
// get client and create LE AID
const client = await getOrCreateClient(lePasscode, env);
const leInfo = await createAid(client, 'le');
await fs.promises.writeFile(`${dataDir}/le-aid.txt`, leInfo.aid);
await fs.promises.writeFile(`${dataDir}/le-info.json`, JSON.stringify(leInfo));
console.log(`LE info written to ${dataDir}/le-*`);
