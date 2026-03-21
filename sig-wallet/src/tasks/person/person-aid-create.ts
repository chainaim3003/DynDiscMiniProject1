import fs from 'fs';
import {createAid, getOrCreateClient} from "../../client/identifiers.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const personPasscode = args[1];
const dataDir = args[2];
const personAlias = args[3] || 'person';  // Use provided alias or default to 'person'

// get client and create Person AID
const client = await getOrCreateClient(personPasscode, env);
const personInfo: any = await createAid(client, personAlias);

// Use synchronous writes with alias-based filenames
fs.writeFileSync(`${dataDir}/${personAlias}-aid.txt`, personInfo.aid);
fs.writeFileSync(`${dataDir}/${personAlias}-info.json`, JSON.stringify(personInfo, null, 2));

// Also write legacy format for backwards compatibility
fs.writeFileSync(`${dataDir}/person-aid.txt`, personInfo.aid);
fs.writeFileSync(`${dataDir}/person-info.json`, JSON.stringify(personInfo, null, 2));

// Verify files were written
if (!fs.existsSync(`${dataDir}/${personAlias}-info.json`)) {
    throw new Error(`Failed to write ${dataDir}/${personAlias}-info.json`);
}

console.log(`Person info written to ${dataDir}/${personAlias}-*`)
console.log(`   Prefix: ${personInfo.aid}`)
console.log(`   OOBI: ${personInfo.oobi}`)
