import { waitOperation } from "./operations.js";
export async function resolveOobi(client, oobi, alias) {
    const op = await client.oobis().resolve(oobi, alias);
    await waitOperation(client, op);
}
