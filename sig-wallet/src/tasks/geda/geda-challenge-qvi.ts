import {getOrCreateClient} from "../../client/identifiers.js";
import {waitOperation} from "../../client/operations.js";

// process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const gedaPasscode = args[1];
const qviAid = args[2];
const challengeInfoPath = args[3];

// get client and generate challenge
const gedaClient = await getOrCreateClient(gedaPasscode, env);

console.log(`GEDA generating challenge for QVI: ${qviAid}`);

// Generate challenge words (using 128-bit challenge like in the test)
const challengeResult = await gedaClient.challenges().generate(128);
console.log(`Generated challenge with ${challengeResult.words.length} words`);

// Store challenge info for QVI to respond to
const challengeInfo = {
    words: challengeResult.words,
    qviAid: qviAid,
    timestamp: new Date().toISOString()
};

// Write challenge info to file for QVI to use
await import('fs').then(fs => 
    fs.promises.writeFile(challengeInfoPath, JSON.stringify(challengeInfo, null, 2))
);

console.log(`Challenge info written to ${challengeInfoPath}`);
console.log(`Challenge words: ${challengeResult.words.join(' ')}`);

