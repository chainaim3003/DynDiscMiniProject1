import {getOrCreateClient} from "../../client/identifiers.js";

// process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const gedaPasscode = args[1];
const qviAid = args[2];
const challengeWords = args[3];

// get client and respond to challenge
const gedaClient = await getOrCreateClient(gedaPasscode, env);

console.log(`GEDA responding to QVI challenge: ${qviAid}`);

// Respond to the challenge with the provided words
await gedaClient.challenges().respond('geda', qviAid, challengeWords.split(' '));
console.log('GEDA responded to QVI challenge with signed words');


