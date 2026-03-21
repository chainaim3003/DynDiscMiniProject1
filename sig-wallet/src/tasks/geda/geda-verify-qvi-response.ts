import {getOrCreateClient} from "../../client/identifiers.js";
import {waitOperation} from "../../client/operations.js";
import {Serder} from "signify-ts";

// process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const gedaPasscode = args[1];
const qviAid = args[2];
const challengeWords = args[3];

// get client and verify challenge response
const gedaClient = await getOrCreateClient(gedaPasscode, env);

console.log(`GEDA verifying QVI challenge response: ${qviAid}`);

// Verify QVI's response to the challenge
const verifyOperation = await waitOperation(
    gedaClient,
    await gedaClient.challenges().verify(qviAid, challengeWords.split(' '))
);
console.log('GEDA verified QVI challenge response');

// Mark response as accepted
const verifyResponse = verifyOperation.response as {
    exn: Record<string, unknown>;
};
const exn = new Serder(verifyResponse.exn);

await gedaClient.challenges().responded(qviAid, exn.sad.d);
console.log('GEDA marked QVI challenge response as accepted');


