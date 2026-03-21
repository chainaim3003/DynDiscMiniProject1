import fs from 'fs';
import { createAid, getOrCreateClient } from "../../client/identifiers.js";
const args = process.argv.slice(2);
const env = args[0];
const qviPasscode = args[1];
const dataDir = args[2];
// get client and create GEDA AID
const client = await getOrCreateClient(qviPasscode, env);
const gedaInfo = await createAid(client, 'geda');
await fs.promises.writeFile(`${dataDir}/geda-aid.txt`, gedaInfo.aid);
await fs.promises.writeFile(`${dataDir}/geda-info.json`, JSON.stringify(gedaInfo));
console.log(`GEDA info written to ${dataDir}/geda-*`);
