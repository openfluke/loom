// Vite/webpack style import queries
declare module "*?raw" {
  const src: string;
  export default src;
}
declare module "*?url" {
  const url: string;
  export default url;
}

// Go wasm_exec globals
declare class Go {
  importObject: WebAssembly.Imports;
  run(instance: WebAssembly.Instance): Promise<void> | void;
}

declare const NewNetworkFloat32: any;
declare const NewNetworkInt32: any;
declare const NewNetworkUint32: any;
