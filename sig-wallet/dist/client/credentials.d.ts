import { CredentialResult, Operation, SignifyClient } from "signify-ts";
/**
 * Creates a credential registry under the specified AID.
 *
 * @returns {Promise<{registryRegk: string}>} Object containing the delegatee QVI multisig AID OOBI
 * @param client
 * @param aidName
 * @param registryName
 */
export declare function createRegistry(client: SignifyClient, aidName: string, registryName: string): Promise<{
    regk: any;
    operation: Operation<any>;
}>;
/**
* Finds matching credentials based on a filter.
* @param {SignifyClient} client - The SignifyClient instance of the holder.
* @param {any} filter - The filter object to apply (e.g., { '-s': schemaSaid, '-a-attributeName': value }).
* @returns {Promise<any[]>} An array of matching credentials.
*/
export declare function findMatchingCredentials(client: SignifyClient, filter: any): Promise<CredentialResult[]>;
/**
 * Returns a credential that has been received through an IPEX Admit by the client.
 * @param client SignifyClient for the recipient or for multisig the client of one of the recipients
 * @param credId SAID of the credential to retrieve
 * @returns the credential body
 */
export declare function getReceivedCredential(client: SignifyClient, credId: string): Promise<CredentialResult>;
/**
 * Searches for ACDC credentials by schema SAID and issuer prefix, returning any matches,
 * @param client
 * @param schemaSAID
 * @param issuerPrefix
 */
export declare function getReceivedCredBySchemaAndIssuer(client: SignifyClient, schemaSAID: string, issuerPrefix: string): Promise<CredentialResult[]>;
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
export declare function issueCredential(issClient: SignifyClient, issAidName: string, registry: string, schema: string, hldPrefix: string, credData: any, credEdge?: any, credRules?: any): Promise<{
    said: string;
    issuer: string;
    issuee: string;
    acdc: any;
    anc: any;
    iss: any;
}>;
export declare function ipexGrantCredential(client: SignifyClient, issueeAid: string, issName: string, acdc: any, anc: any, iss: any): Promise<any>;
/**
 * Submits an IPEX admit (accepts a grant).
 * @param {SignifyClient} client - The SignifyClient instance of the holder.
 * @param {string} senderAidAlias - The alias of the AID admitting the grant.
 * @param {string} recipientAidPrefix - The AID prefix of the original grantor.
 * @param {string} grantSaid - The SAID of the grant being admitted.
 * @param {string} [message=''] - Optional message for the admit.
 * @returns {Promise<{ operation: Operation<any> }>} The operation details.
 */
export declare function ipexAdmitGrant(client: SignifyClient, senderAidAlias: string, recipientAidPrefix: string, grantSaid: string, message?: string): Promise<{
    operation: Operation<any>;
}>;
/**
 * Uses IPEX to grant a credential to a recipient AID.
 *
 * @param client SignifyClient instance of the client performing the grant
 * @param senderAidName name of the AID sending the credential
 * @param credentialSAID The SAID of the credential to be granted
 * @param recipientPrefix identifier of the recipient AID who will receive the credential presentation
 * @returns {Promise<string>} String true/false if QVI credential exists or not for the QAR
 */
export declare function grantCredential(client: SignifyClient, senderAidName: string, credentialSAID: string, recipientPrefix: string): Promise<string>;
