export const isBrowser = typeof window !== "undefined" && typeof document !== "undefined";
export const isNode = typeof process !== "undefined" && !!(process.versions as any)?.node;
export const isBun = typeof (globalThis as any).Bun !== "undefined";
