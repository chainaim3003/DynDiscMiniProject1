import {
    randomPasscode,
    ready,
    SignifyClient,
    Tier,
} from 'signify-ts';
import {EnvType, resolveEnvironment} from "./resolve-env.js";
import {waitOperation} from "./operations.js";
import {resolveOobi} from "./oobis.js";

/**
 * Connect or boot a SignifyClient instance
 */
export async function getOrCreateClient(
    bran: string | undefined = undefined,
    environment: EnvType | undefined = undefined,
): Promise<SignifyClient> {
    const env = resolveEnvironment(environment);
    await ready();
    bran ??= randomPasscode();
    bran = bran.padEnd(21, '_');
    let adminUrl = env.adminUrl;
    let bootUrl = env.bootUrl;
    const client = new SignifyClient(adminUrl, bran, Tier.low, bootUrl);
    try {
        await client.connect();
    } catch {
        const res = await client.boot();
        if (!res.ok) throw new Error();
        await client.connect();
    }
    // console.log('SignifyClient', {agent: client.agent?.pre, controller: client.controller.pre});
    return client;
}

/**
 * Create an AID with an agent role and return its AID and OOBI
 * @param client
 * @param name
 */
export async function createAid(client: SignifyClient, name: string) {
    // incept
    const icpResp = await client.identifiers()
        .create(name);
    const op = await icpResp.op();
    const aid = op.name.split('.')[1];

    // add endpoint role
    const endRoleOp = await client.identifiers()
        .addEndRole(name, 'agent', client!.agent!.pre);
    await waitOperation(client, await endRoleOp.op());

    // get OOBI
    const oobiResp = await client.oobis().get(name, 'agent');
    const oobi = oobiResp.oobis[0];

    return {aid, oobi};
}

/**
 *
 * @param dgtName delegate name
 * @param delPre delegator prefix
 * @param dgrName delegator name
 * @param delOobi delegator OOBI - used to resolve key state
 */
export async function createDelegate(dgtClient: SignifyClient, dgtName: string, delPre: string, dgrName: string, delOobi: string) {
    // Update the delegator OOBI to know where to send the delegation request to
    await resolveOobi(dgtClient, delOobi, dgrName)

    // create delegate
    const icpRes = await dgtClient.identifiers().create(dgtName, {
        delpre: delPre
    });
    const op = await icpRes.op();
    const delegatePre = op.name.split('.')[1];

    return {
        aid: delegatePre,
        icpOpName: op.name
    }
}

/**
 *
 * Approve delegation for a delegate prefix from a delegator client
 * @param dgrClient delegator SignifyClient
 * @param dgrName delegator name
 * @param dgtPre delegate prefix
 */
export async function approveDelegation(dgrClient: SignifyClient, dgrName: string, dgtPre: string) {
    // delegator anchor of delegate prefix in key event log - is the approval of delegation
    const anchor = {
        i: dgtPre,
        s: '0',
        d: dgtPre,
    };
    const apprDelRes = await dgrClient
        .delegations()
        .approve(dgrName, anchor);
    const opResp = await waitOperation(dgrClient, await apprDelRes.op());
    return opResp.done;
}

/**
 * Get an AID by name from the client
 * @param client SignifyClient instance
 * @param name AID name to retrieve
 */
export async function getAID(client: SignifyClient, name: string) {
    const aid = await client.identifiers().get(name);
    return aid;
}