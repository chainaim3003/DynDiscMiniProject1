import fs from 'fs';
import { getOrCreateClient } from "../../client/identifiers.js";
import { createRegistry } from "../../client/credentials.js";

// Process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const issuerAidName = args[1];
const passcode = args[2]; // This can be empty if we need to look it up from bran file
const registryName = args[3];
const taskDataDir = args[4] || '/task-data';

console.log(`Creating invoice credential registry...`);
console.log(`  Issuer: ${issuerAidName}`);
console.log(`  Registry: ${registryName}`);
console.log(`  Task data dir: ${taskDataDir}`);

// Determine the actual passcode to use
let actualPasscode = passcode;

// If passcode is empty or not provided, try to read from agent's bran file
if (!passcode || passcode.trim() === '') {
    const branFilePath = `${taskDataDir}/${issuerAidName}-bran.txt`;
    console.log(`  No passcode provided, checking for BRAN file: ${branFilePath}`);
    
    if (fs.existsSync(branFilePath)) {
        actualPasscode = fs.readFileSync(branFilePath, 'utf-8').trim();
        console.log(`  ✓ Found agent's unique BRAN: ${actualPasscode.substring(0, 20)}...`);
    } else {
        // Also check for shared passcode file
        const sharedPasscodePath = `${taskDataDir}/shared-passcode.txt`;
        if (fs.existsSync(sharedPasscodePath)) {
            actualPasscode = fs.readFileSync(sharedPasscodePath, 'utf-8').trim();
            console.log(`  ✓ Found shared passcode from: ${sharedPasscodePath}`);
        } else {
            console.error(`  ✗ ERROR: No BRAN file found at ${branFilePath}`);
            console.error(`  ✗ ERROR: No shared passcode found at ${sharedPasscodePath}`);
            console.error(`  The agent must have been created with a unique BRAN first.`);
            process.exit(1);
        }
    }
}

console.log(`  Using passcode: ${actualPasscode.substring(0, 20)}...`);

// Get client with the agent's unique passcode
const client = await getOrCreateClient(actualPasscode, env);
console.log(`  Client controller: ${client.controller.pre}`);
console.log(`  Client agent: ${client.agent?.pre}`);

// List existing identifiers to verify
try {
    const identifiers = await client.identifiers().list();
    console.log(`  Available identifiers in this client: ${identifiers.aids?.length || 0}`);
    if (identifiers.aids && identifiers.aids.length > 0) {
        identifiers.aids.forEach((aid: any) => {
            console.log(`    - ${aid.name}: ${aid.prefix}`);
        });
    }
} catch (e) {
    console.log(`  Could not list identifiers: ${e}`);
}

// Create registry for invoice credentials
try {
    const registryResult = await createRegistry(client, issuerAidName, registryName);
    console.log(`✓ Registry created with ID: ${registryResult.regk}`);
    
    // Save registry info
    const registryInfoPath = `${taskDataDir}/${issuerAidName}-invoice-registry-info.json`;
    const registryInfo = {
        registryName: registryName,
        regk: registryResult.regk,
        agentAlias: issuerAidName,
        createdAt: new Date().toISOString()
    };
    fs.writeFileSync(registryInfoPath, JSON.stringify(registryInfo, null, 2));
    console.log(`✓ Registry info saved to: ${registryInfoPath}`);
    
} catch (error: any) {
    console.error(`✗ Failed to create registry: ${error.message}`);
    
    // Provide more diagnostic info
    if (error.message.includes('404') || error.message.includes('Not Found')) {
        console.error(`\nDIAGNOSTIC: The identifier '${issuerAidName}' was not found.`);
        console.error(`This could mean:`);
        console.error(`  1. The BRAN/passcode used to connect doesn't match the one used to create the agent`);
        console.error(`  2. The agent identifier was created with a different name`);
        console.error(`  3. The agent creation didn't complete successfully`);
        console.error(`\nPlease verify:`);
        console.error(`  - BRAN file exists: ${taskDataDir}/${issuerAidName}-bran.txt`);
        console.error(`  - Agent info exists: ${taskDataDir}/${issuerAidName}-info.json`);
        console.error(`  - Agent delegate info exists: ${taskDataDir}/${issuerAidName}-delegate-info.json`);
    }
    
    process.exit(1);
}
