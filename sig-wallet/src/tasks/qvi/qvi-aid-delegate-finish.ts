import fs from 'fs';
import {getOrCreateClient} from "../../client/identifiers.js";
import {waitOperation} from "../../client/operations.js";
import {SignifyClient} from "signify-ts";

// Pull in arguments from the command line and configuration
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1];
const delegateAidName = args[2];
const delegatorInfoPath = args[3];
const delegateInfoPath = args[4];
const delegateOutputPath = args[5];

async function finishDelegation(
    dgtClient: SignifyClient,
    delPre: string,
    delegateName: string,
    delOpName: string) {
    console.log(`Finishing ${delPre} delegation to ${delegateName} on op: ${delOpName}...`);

    // refresh delegator key state to discover delegation anchor
    const op: any = await dgtClient.keyStates().query(delPre, '1');
    await waitOperation(dgtClient, op);

    // wait for delegate inception to complete
    const delOp: any = await dgtClient.operations().get(delOpName);
    await waitOperation(dgtClient, delOp);

    // finish identifier setup
    // add endpoint role
    const endRoleOp = await dgtClient.identifiers()
        .addEndRole(delegateName, 'agent', dgtClient!.agent!.pre);
    await waitOperation(dgtClient, await endRoleOp.op());

    // get oobi
    const oobiResp = await dgtClient.oobis().get(delegateName, 'agent');
    const oobi = oobiResp.oobis[0]

    const aid = await dgtClient.identifiers().get(delegateName)
    return {
        aid: aid.prefix,
        oobi
    }
}

const dgtClient = await getOrCreateClient(passcode, env);

// Read delegator info - use synchronous read
if (!fs.existsSync(delegatorInfoPath)) {
    throw new Error(`Delegator info file not found: ${delegatorInfoPath}`);
}
const dgrInfo = JSON.parse(fs.readFileSync(delegatorInfoPath, 'utf-8'));

// Read delegate inception info - use synchronous read
if (!fs.existsSync(delegateInfoPath)) {
    throw new Error(`Delegate info file not found: ${delegateInfoPath}`);
}
const dgtInfo = JSON.parse(fs.readFileSync(delegateInfoPath, 'utf-8'));
console.log(`${delegateAidName} delegate info`, dgtInfo);

// finish delegation and write data to file
const delegationInfo: any = await finishDelegation(dgtClient, dgrInfo.aid, delegateAidName, dgtInfo.icpOpName);

// Use synchronous write
fs.writeFileSync(delegateOutputPath, JSON.stringify(delegationInfo, null, 2));

// Verify file was written
if (!fs.existsSync(delegateOutputPath)) {
    throw new Error(`Failed to write ${delegateOutputPath}`);
}

console.log(`Delegate data written to ${delegateOutputPath}`);
