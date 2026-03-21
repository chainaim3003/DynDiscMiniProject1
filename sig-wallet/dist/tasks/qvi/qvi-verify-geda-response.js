import { getOrCreateClient } from "../../client/identifiers.js";
import { waitOperation } from "../../client/operations.js";
import { Serder } from "signify-ts";
// process arguments
const args = process.argv.slice(2);
const env = args[0];
const qviPasscode = args[1];
const gedaAid = args[2];
const challengeWords = args[3];
// get client and verify challenge response
const qviClient = await getOrCreateClient(qviPasscode, env);
console.log(`QVI verifying GEDA challenge response: ${gedaAid}`);
// Verify GEDA's response to the challenge
const verifyOperation = await waitOperation(qviClient, await qviClient.challenges().verify(gedaAid, challengeWords.split(' ')));
console.log('QVI verified GEDA challenge response');
// Mark response as accepted
const verifyResponse = verifyOperation.response;
const exn = new Serder(verifyResponse.exn);
await qviClient.challenges().responded(gedaAid, exn.sad.d);
console.log('QVI marked GEDA challenge response as accepted');
