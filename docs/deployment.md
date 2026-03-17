# Deployment: TypeScript, WASM, and NPM

Loom is designed to be **isomorphic**, meaning the exact same mathematical engine runs in both Node.js (backend) and the Browser (frontend) via a bit-perfect WebAssembly (WASM) bridge.

---

## 📦 The NPM Package: `@openfluke/welvet`

The primary way to use Loom in the JavaScript ecosystem is through the **Welvet** SDK.

### Installation
```bash
npm install @openfluke/welvet
```

### Quick Start (Node.js)
```typescript
import { init, createNetwork } from '@openfluke/welvet';

// Initialize the WASM runtime
await init();

// Build a network from a JSON specification
const net = await createNetwork({
    id: "demo-net",
    depth: 1, rows: 2, cols: 1, layers_per_cell: 1,
    layers: [
        { z: 0, y: 0, x: 0, l: 0, type: "Dense", input_height: 128, output_height: 64, activation: "ReLU" },
        { z: 0, y: 1, x: 0, l: 0, type: "Dense", input_height: 64, output_height: 10, activation: "Linear" }
    ]
});

// Run a forward pass
const input = new Float32Array(128).fill(0.5);
const output = await net.sequentialForward(input);
console.log("Network output:", output);
```

---

## 🌐 WASM & FFI Bridge

The TypeScript SDK communicates with the Go-compiled core via the **Universal C-ABI**. This ensures that complex logic (like NEAT evolution or DNA extraction) remains fast while providing a high-level, idiomatic JS interface.

### Verified Capabilities (v0.74.0)
The isomorphic bridge has been verified through a 36-count diagnostic suite:
- **Core Exports**: 8/8 internal WASM symbols verified.
* **Network Methods**: 16/16 functional wrappers (Forward, DNA, Morph, etc.) passed.
* **NEAT Population**: 8/8 evolutionary logic methods verified.
* **Bit-Perfect Parity**: 0.000000% divergence vs the Go native reference.

---

## 🖼️ Browser Deployment (WebGPU)

When running in the browser, the WASM runtime can automatically detect and utilize **WebGPU** for massive parallel speedups.

```typescript
import { setupWebGPU } from '@openfluke/welvet';

// Initialize WebGPU context
await setupWebGPU();

// Networks created after this point will utilize GPU kernels
// for forward and backward passes.
```

### Performance Tiers
| Environment | Backend | Best For |
| :--- | :--- | :--- |
| **Node.js** | WASM (SIMD) | Backend inference, server-side DNA comparison. |
| **Browser** | WASM + WebGPU | High-performance interactive AI, on-device training. |
| **Mobile Web** | WASM | Lightweight edge execution. |

---

## 🧬 DNA & Evolution in JS

The TypeScript SDK provides full access to the DNA logic:
- **`net.extractDNA()`**: Generates a topological fingerprint.
- **`compareLoomDNA(dnaA, dnaB)`**: Cross-platform similarity score.
- **`createLoomNEATPopulation(id, size, cfg)`**: High-speed evolutionary architecture search.

For more details on the underlying DNA math, see [dna.md](dna.md).
