import fs from 'fs';
import { createDelegate, getOrCreateClient } from "../../client/identifiers.js";
const args = process.argv.slice(2);
const env = args[0];
const qviPasscode = args[1];
const dataDir = args[2];
// GEDA delegator info
const gedaInfo = JSON.parse(await fs.promises.readFile(`${dataDir}/geda-info.json`, 'utf-8'));
// get delegate client and send delegation request to delegator (GEDA)
const dgtClient = await getOrCreateClient(qviPasscode, env);
const clientInfo = await createDelegate(dgtClient, 'qvi', gedaInfo.aid, 'geda', gedaInfo.oobi);
const qviPath = `${dataDir}/qvi-delegate-info.json`;
await fs.promises.writeFile(qviPath, JSON.stringify(clientInfo));
console.log(`QVI info written to ${qviPath}`);
