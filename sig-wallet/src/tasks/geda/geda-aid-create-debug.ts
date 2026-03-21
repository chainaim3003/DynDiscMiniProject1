import fs from 'fs';
import {createAid, getOrCreateClient} from "../../client/identifiers.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const qviPasscode = args[1];
const dataDir = args[2];

console.log('=== GEDA AID Creation Debug ===');
console.log(`Environment: ${env}`);
console.log(`Data directory: ${dataDir}`);
console.log(`Passcode length: ${qviPasscode.length}`);
console.log('');

console.log('Step 1: Getting or creating SignifyClient...');
try {
    const client = await getOrCreateClient(qviPasscode, env);
    console.log('✓ Client connected successfully');
    console.log(`  Agent: ${client.agent?.pre}`);
    console.log(`  Controller: ${client.controller.pre}`);
    console.log('');
    
    console.log('Step 2: Creating GEDA AID...');
    const gedaInfo: any = await createAid(client, 'geda');
    console.log('✓ GEDA AID created successfully');
    console.log(`  AID: ${gedaInfo.aid}`);
    console.log('');
    
    console.log('Step 3: Writing files...');
    fs.writeFileSync(`${dataDir}/geda-aid.txt`, gedaInfo.aid);
    console.log('  ✓ Written geda-aid.txt');
    
    fs.writeFileSync(`${dataDir}/geda-info.json`, JSON.stringify(gedaInfo, null, 2));
    console.log('  ✓ Written geda-info.json');
    console.log('');
    
    console.log('Step 4: Verifying files...');
    if (!fs.existsSync(`${dataDir}/geda-aid.txt`)) {
        throw new Error(`Failed to write ${dataDir}/geda-aid.txt`);
    }
    if (!fs.existsSync(`${dataDir}/geda-info.json`)) {
        throw new Error(`Failed to write ${dataDir}/geda-info.json`);
    }
    console.log('  ✓ Files verified');
    console.log('');
    
    console.log(`GEDA info written to ${dataDir}/geda-*`);
    console.log('=== SUCCESS ===');
} catch (error) {
    console.error('=== ERROR ===');
    console.error('Failed at:', error);
    console.error('Stack:', (error as Error).stack);
    process.exit(1);
}
