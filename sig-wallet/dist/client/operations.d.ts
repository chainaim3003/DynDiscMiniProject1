import { Operation, SignifyClient } from 'signify-ts';
/**
 * Poll for operation to become completed.
 * Removes completed operation
 */
export declare function waitOperation<T = any>(client: SignifyClient, op: Operation<T> | string, signal?: AbortSignal): Promise<Operation<T>>;
