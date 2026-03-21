import { SignifyClient } from 'signify-ts';
import { EnvType } from "./resolve-env.js";
/**
 * Connect or boot a SignifyClient instance
 */
export declare function getOrCreateClient(bran?: string | undefined, environment?: EnvType | undefined): Promise<SignifyClient>;
/**
 * Create an AID with an agent role and return its AID and OOBI
 * @param client
 * @param name
 */
export declare function createAid(client: SignifyClient, name: string): Promise<{
    aid: any;
    oobi: any;
}>;
/**
 *
 * @param dgtName delegate name
 * @param delPre delegator prefix
 * @param dgrName delegator name
 * @param delOobi delegator OOBI - used to resolve key state
 */
export declare function createDelegate(dgtClient: SignifyClient, dgtName: string, delPre: string, dgrName: string, delOobi: string): Promise<{
    aid: any;
    icpOpName: any;
}>;
/**
 *
 * Approve delegation for a delegate prefix from a delegator client
 * @param dgrClient delegator SignifyClient
 * @param dgrName delegator name
 * @param dgtPre delegate prefix
 */
export declare function approveDelegation(dgrClient: SignifyClient, dgrName: string, dgtPre: string): Promise<boolean>;
