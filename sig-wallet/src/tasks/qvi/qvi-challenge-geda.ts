import {getOrCreateClient} from "../../client/identifiers.js";

// process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const qviPasscode = args[1];
const gedaAid = args[2];
const challengeInfoPath = args[3];

// get client and generate challenge
const qviClient = await getOrCreateClient(qviPasscode, env);

console.log(`QVI generating challenge for GEDA: ${gedaAid}`);

// Generate challenge words (using 128-bit challenge like in the test)
const challengeResult = await qviClient.challenges().generate(128);
console.log(`Generated challenge with ${challengeResult.words.length} words`);

// Store challenge info for GEDA to respond to
const challengeInfo = {
    words: challengeResult.words,
    gedaAid: gedaAid,
    timestamp: new Date().toISOString()
};

// Write challenge info to file for GEDA to use
await import('fs').then(fs => 
    fs.promises.writeFile(challengeInfoPath, JSON.stringify(challengeInfo, null, 2))
);

console.log(`Challenge info written to ${challengeInfoPath}`);
console.log(`Challenge words: ${challengeResult.words.join(' ')}`);


