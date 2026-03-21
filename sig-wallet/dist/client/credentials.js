import { Serder } from "signify-ts";
import { createTimestamp, DEFAULT_TIMEOUT_MS } from "../time.js";
import { waitOperation } from "./operations.js";
/**
 * Creates a credential registry under the specified AID.
 *
 * @returns {Promise<{registryRegk: string}>} Object containing the delegatee QVI multisig AID OOBI
 * @param client
 * @param aidName
 * @param registryName
 */
export async function createRegistry(client, aidName, registryName) {
    const aid = await client.identifiers().get(aidName);
    const existing = await client.registries().list(aidName);
    if (existing.find((reg) => reg.name === registryName)) {
        console.log(`Registry "${registryName}" already exists under AID "${aidName}". Skipping creation.`);
        return { regk: existing.find((reg) => reg.name === registryName)?.regk, operation: null };
    }
    try {
        const createRegistryResult = await client
            .registries()
            .create({ name: aidName, registryName: registryName });
        const operationDetails = await createRegistryResult.op();
        const completedOperation = await client
            .operations()
            .wait(operationDetails, { signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS) });
        if (completedOperation.error) {
            throw new Error(`Credential registry creation failed: ${JSON.stringify(completedOperation.error)}`);
        }
        const registrySaid = completedOperation?.response?.anchor?.i;
        console.log(`Successfully created credential registry: ${registrySaid}`);
        await client.operations().delete(completedOperation.name);
        return { regk: registrySaid, operation: completedOperation };
    }
    catch (error) {
        console.error(`Failed to create credential registry "${registryName}":`, error);
        throw error;
    }
}
/**
* Finds matching credentials based on a filter.
* @param {SignifyClient} client - The SignifyClient instance of the holder.
* @param {any} filter - The filter object to apply (e.g., { '-s': schemaSaid, '-a-attributeName': value }).
* @returns {Promise<any[]>} An array of matching credentials.
*/
export async function findMatchingCredentials(client, filter) {
    console.log('Finding matching credentials with filter:', filter);
    try {
        const matchingCredentials = await client.credentials().list({ filter });
        console.log(`Found ${matchingCredentials.length} matching credentials.`);
        return matchingCredentials;
    }
    catch (error) {
        console.error('Failed to find matching credentials:', error);
        throw error;
    }
}
/**
 * Returns a credential that has been received through an IPEX Admit by the client.
 * @param client SignifyClient for the recipient or for multisig the client of one of the recipients
 * @param credId SAID of the credential to retrieve
 * @returns the credential body
 */
export async function getReceivedCredential(client, credId) {
    const credentialList = await findMatchingCredentials(client, {
        '-d': credId,
    });
    let credential;
    if (credentialList.length > 0) {
        credential = credentialList[0];
    }
    return credential;
}
/**
 * Searches for ACDC credentials by schema SAID and issuer prefix, returning any matches,
 * @param client
 * @param schemaSAID
 * @param issuerPrefix
 */
export async function getReceivedCredBySchemaAndIssuer(client, schemaSAID, issuerPrefix) {
    return await findMatchingCredentials(client, {
        '-s': schemaSAID,
        '-i': issuerPrefix
    });
}
/**
 * Issues a credential
 *
 * @param issClient The client issuing the credential
 * @param issAidName The name of the AID that is issuing the credential
 * @param registry the registry to use for the credential
 * @param schema the schema SAID for the credential
 * @param hldPrefix identifier prefix for the holder of the credential
 * @param credData
 * @param credEdge
 * @param credRules
 * @returns {Promise<>} Object containing holder's credential information
 */
export async function issueCredential(issClient, issAidName, registry, schema, hldPrefix, credData, credEdge, credRules) {
    // Ensure the registry exists
    const issAid = await issClient.identifiers().get(issAidName);
    const registries = await issClient.registries().list(issAidName);
    const issRegistry = registries.find((reg) => reg.name === registry);
    // console.log(`Found registry: ${issRegistry?.regk}`, issRegistry);
    console.log(`Cred data:`, credData);
    const kargsSub = {
        i: hldPrefix,
        dt: createTimestamp(),
        ...credData,
    };
    const issData = {
        i: issAid.prefix,
        ri: issRegistry.regk,
        s: schema,
        a: kargsSub,
        e: credEdge,
        r: credRules,
    };
    console.log("Credential data:", issData);
    const issResult = await issClient.credentials().issue(issAidName, issData);
    const issueOp = await waitOperation(issClient, issResult.op);
    if (issueOp.error) {
        throw new Error(`Credential issuance failed: ${JSON.stringify(issueOp.error)}`);
    }
    // console.log("Issuance succeeded:", issueOp)
    const credentialSad = issueOp.response; // The full Self-Addressing Data (SAD) of the credential
    const credentialSaid = credentialSad?.ced?.d; // The SAID of the credential
    const cred = await issClient.credentials().get(credentialSaid);
    return {
        said: cred.sad.d,
        issuer: cred.sad.i,
        // @ts-ignore
        issuee: cred.sad?.a.i,
        acdc: issResult.acdc,
        anc: issResult.anc,
        iss: issResult.iss
    };
}
export async function ipexGrantCredential(client, issueeAid, issName, acdc, anc, iss) {
    const dt = createTimestamp();
    const [grant, gsigs, end] = await client.ipex().grant({
        senderName: issName,
        recipient: issueeAid,
        datetime: dt,
        acdc: acdc,
        anc: anc,
        iss: iss,
    });
    let op = await client
        .ipex()
        .submitGrant(issName, grant, gsigs, end, [issueeAid]);
    op = await waitOperation(client, op);
    console.log('Grant message sent');
    return op;
}
/**
 * Submits an IPEX admit (accepts a grant).
 * @param {SignifyClient} client - The SignifyClient instance of the holder.
 * @param {string} senderAidAlias - The alias of the AID admitting the grant.
 * @param {string} recipientAidPrefix - The AID prefix of the original grantor.
 * @param {string} grantSaid - The SAID of the grant being admitted.
 * @param {string} [message=''] - Optional message for the admit.
 * @returns {Promise<{ operation: Operation<any> }>} The operation details.
 */
export async function ipexAdmitGrant(client, senderAidAlias, recipientAidPrefix, grantSaid, message = '') {
    console.log(`AID "${senderAidAlias}" admitting IPEX grant "${grantSaid}" from AID "${recipientAidPrefix}"...`);
    try {
        const [admit, sigs, aend] = await client.ipex().admit({
            senderName: senderAidAlias,
            message: message,
            grantSaid: grantSaid,
            recipient: recipientAidPrefix,
            datetime: createTimestamp(),
        });
        const admitOperationDetails = await client
            .ipex()
            .submitAdmit(senderAidAlias, admit, sigs, aend, [recipientAidPrefix]);
        const completedOperation = await client
            .operations()
            .wait(admitOperationDetails, { signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS) });
        if (completedOperation.error) {
            throw new Error(`IPEX admit submission failed: ${JSON.stringify(completedOperation.error)}`);
        }
        console.log(`Successfully submitted IPEX admit for grant "${grantSaid}".`);
        await client.operations().delete(completedOperation.name);
        return { operation: completedOperation };
    }
    catch (error) {
        console.error('Failed to submit IPEX admit:', error);
        throw error;
    }
}
/**
 * Uses IPEX to grant a credential to a recipient AID.
 *
 * @param client SignifyClient instance of the client performing the grant
 * @param senderAidName name of the AID sending the credential
 * @param credentialSAID The SAID of the credential to be granted
 * @param recipientPrefix identifier of the recipient AID who will receive the credential presentation
 * @returns {Promise<string>} String true/false if QVI credential exists or not for the QAR
 */
export async function grantCredential(client, senderAidName, credentialSAID, recipientPrefix) {
    // Check to see if the credential exists
    let receivedCred = await getReceivedCredential(client, credentialSAID);
    if (!receivedCred) {
        throw Error(`Credential ${credentialSAID} not found.`);
    }
    const grantTime = createTimestamp();
    console.log(`IPEX Granting credential ${credentialSAID} to ${recipientPrefix}...`);
    const [grant, gsigs, gend] = await client.ipex().grant({
        senderName: senderAidName,
        acdc: new Serder(receivedCred.sad),
        anc: new Serder(receivedCred.anc),
        iss: new Serder(receivedCred.iss),
        ancAttachment: receivedCred.ancatc,
        recipient: recipientPrefix,
        datetime: grantTime,
    });
    const op = await client
        .ipex()
        .submitGrant(senderAidName, grant, gsigs, gend, [
        recipientPrefix,
    ]);
    await waitOperation(client, op);
    return op.response;
}
