// Constants
export const DEFAULT_TIMEOUT_MS = 30000; // 30 seconds for operations
export const DEFAULT_DELAY_MS = 5000; // 5 seconds for operations

export function createTimestamp() {
  return new Date().toISOString().replace('Z', '000+00:00');
}
