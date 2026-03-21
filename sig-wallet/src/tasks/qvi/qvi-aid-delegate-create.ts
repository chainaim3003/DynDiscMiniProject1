import fs from 'fs';
import {createDelegate, getOrCreateClient} from "../../client/identifiers.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const qviPasscode = args[1];
const dataDir = args[2];

// GEDA delegator info - use synchronous read
const gedaInfoPath = `${dataDir}/geda-info.json`;
if (!fs.existsSync(gedaInfoPath)) {
    throw new Error(`GEDA info file not found: ${gedaInfoPath}`);
}
const gedaInfo = JSON.parse(fs.readFileSync(gedaInfoPath, 'utf-8'));

// get delegate client and send delegation request to delegator (GEDA)
const dgtClient = await getOrCreateClient(qviPasscode, env);
const clientInfo: any = await createDelegate(dgtClient, 'qvi', gedaInfo.aid, 'geda', gedaInfo.oobi);
const qviPath = `${dataDir}/qvi-delegate-info.json`;

// Use synchronous write to ensure persistence
fs.writeFileSync(qviPath, JSON.stringify(clientInfo, null, 2));

// Verify file was written
if (!fs.existsSync(qviPath)) {
    throw new Error(`Failed to write ${qviPath}`);
}

console.log(`QVI info written to ${qviPath}`);
