import fs from 'fs';
import { createAid, getOrCreateClient } from "../../client/identifiers.js";
const args = process.argv.slice(2);
const env = args[0];
const personPasscode = args[1];
const dataDir = args[2];
// get client and create Person AID
const client = await getOrCreateClient(personPasscode, env);
const personInfo = await createAid(client, 'person');
await fs.promises.writeFile(`${dataDir}/person-aid.txt`, personInfo.aid);
await fs.promises.writeFile(`${dataDir}/person-info.json`, JSON.stringify(personInfo));
console.log(`Person info written to ${dataDir}/person-*`);
