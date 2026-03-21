import fs from 'fs';
import {createAid, getOrCreateClient} from "../../client/identifiers.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const qviPasscode = args[1];
const dataDir = args[2];

// get client and create GEDA AID
const client = await getOrCreateClient(qviPasscode, env);
const gedaInfo: any = await createAid(client, 'geda');

// Use synchronous writes to ensure files are persisted before script exits
fs.writeFileSync(`${dataDir}/geda-aid.txt`, gedaInfo.aid);
fs.writeFileSync(`${dataDir}/geda-info.json`, JSON.stringify(gedaInfo, null, 2));

// Verify files were written
if (!fs.existsSync(`${dataDir}/geda-aid.txt`)) {
    throw new Error(`Failed to write ${dataDir}/geda-aid.txt`);
}
if (!fs.existsSync(`${dataDir}/geda-info.json`)) {
    throw new Error(`Failed to write ${dataDir}/geda-info.json`);
}

console.log(`GEDA info written to ${dataDir}/geda-*`)
