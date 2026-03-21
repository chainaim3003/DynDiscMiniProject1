import fs from 'fs';
import { getOrCreateClient } from "../../client/identifiers.js";
import { waitOperation } from "../../client/operations.js";
// Pull in arguments from the command line and configuration
const args = process.argv.slice(2);
const env = args[0];
const passcode = args[1];
const delegateAidName = args[2];
const delegatorInfoPath = args[3];
const delegateInfoPath = args[4];
const delegateOutputPath = args[5];
async function finishDelegation(dgtClient, delPre, delegateName, delOpName) {
    console.log(`Finishing ${delPre} delegation to ${delegateName} on op: ${delOpName}...`);
    // refresh delegator key state to discover delegation anchor
    const op = await dgtClient.keyStates().query(delPre, '1');
    await waitOperation(dgtClient, op);
    // wait for delegate inception to complete
    const delOp = await dgtClient.operations().get(delOpName);
    await waitOperation(dgtClient, delOp);
    // finish identifier setup
    // add endpoint role
    const endRoleOp = await dgtClient.identifiers()
        .addEndRole(delegateName, 'agent', dgtClient.agent.pre);
    await waitOperation(dgtClient, await endRoleOp.op());
    // get oobi
    const oobiResp = await dgtClient.oobis().get(delegateName, 'agent');
    const oobi = oobiResp.oobis[0];
    const aid = await dgtClient.identifiers().get(delegateName);
    return {
        aid: aid.prefix,
        oobi
    };
}
const dgtClient = await getOrCreateClient(passcode, env);
// Read delegator info
const dgrInfo = JSON.parse(await fs.promises.readFile(delegatorInfoPath, 'utf-8'));
// Read delegate inception info
const dgtInfo = JSON.parse(await fs.promises.readFile(delegateInfoPath, 'utf-8'));
console.log(`${delegateAidName} delegate info`, dgtInfo);
// finish delegation and write data to file
const delegationInfo = await finishDelegation(dgtClient, dgrInfo.aid, delegateAidName, dgtInfo.icpOpName);
await fs.promises.writeFile(delegateOutputPath, JSON.stringify(delegationInfo));
console.log(`Delegate data written to ${delegateOutputPath}`);
