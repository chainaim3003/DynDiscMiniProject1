import {getOrCreateClient} from "../../client/identifiers.js";

// process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const qviPasscode = args[1];
const gedaAid = args[2];
const challengeWords = args[3];

// get client and respond to challenge
const qviClient = await getOrCreateClient(qviPasscode, env);

console.log(`QVI responding to GEDA challenge: ${gedaAid}`);

// Respond to the challenge with the provided words
await qviClient.challenges().respond('qvi', gedaAid, challengeWords.split(' '));
console.log('QVI responded to GEDA challenge with signed words');

