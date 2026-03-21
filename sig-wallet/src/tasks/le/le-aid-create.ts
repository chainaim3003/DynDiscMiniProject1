import fs from 'fs';
import {createAid, getOrCreateClient} from "../../client/identifiers.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const lePasscode = args[1];
const dataDir = args[2];
const leAlias = args[3] || 'le';  // Use provided alias or default to 'le'

// get client and create LE AID
const client = await getOrCreateClient(lePasscode, env);
const leInfo: any = await createAid(client, leAlias);

// Use synchronous writes with alias-based filenames
fs.writeFileSync(`${dataDir}/${leAlias}-aid.txt`, leInfo.aid);
fs.writeFileSync(`${dataDir}/${leAlias}-info.json`, JSON.stringify(leInfo, null, 2));

// Also write legacy format for backwards compatibility
fs.writeFileSync(`${dataDir}/le-aid.txt`, leInfo.aid);
fs.writeFileSync(`${dataDir}/le-info.json`, JSON.stringify(leInfo, null, 2));

// Verify files were written
if (!fs.existsSync(`${dataDir}/${leAlias}-info.json`)) {
    throw new Error(`Failed to write ${dataDir}/${leAlias}-info.json`);
}

console.log(`LE info written to ${dataDir}/le-*`)
console.log(`   Prefix: ${leInfo.aid}`)
console.log(`   OOBI: ${leInfo.oobi}`)
