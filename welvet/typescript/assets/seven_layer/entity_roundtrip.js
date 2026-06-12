/**
 * Fast .entity save/reload smoke — all layers × 21 dtypes.
 * No training, no SC/MC bench. WASM/CABI parity with lucy [7] entity checkpoints.
 */

import {
  ALL_DTYPES, LAYER_SUITES, maxAbsDiff,
} from './bench.js';

function f32(data) {
  return data instanceof Float32Array ? data : new Float32Array(data);
}

function toArray(out) {
  if (typeof out === 'string') {
    const j = JSON.parse(out);
    if (j.error) throw new Error(j.error);
  }
  return Array.from(out);
}

function morphAll(net) {
  const n = JSON.parse(net.getInfo()).total_layers;
  return (dtype) => {
    for (let i = 0; i < n; i++) net.morphLayer(i, dtype);
  };
}

function resetNet(net) {
  if (typeof net.resetLayerState === 'function') net.resetLayerState();
}

function entityNativeOK(net, wire) {
  const n = JSON.parse(net.getInfo()).total_layers;
  for (let i = 0; i < n; i++) {
    // Residual / no-weight layers: nothing to persist (lucy skips nil WeightStore).
    let w;
    try {
      w = net.getLayerWeights(i);
    } catch {
      continue;
    }
    if (!w || w.length === 0 || (typeof w === 'string' && w.includes('error'))) continue;
    const raw = globalThis.layerPersistenceFromEntity(wire, i);
    const r = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (r.error || !r.native || !r.weights) return false;
  }
  return true;
}

/** One grid per suite — 1×1×1 is enough for wire format coverage. */
function primaryGrid(suite) {
  return suite.grids[0];
}

/**
 * @param {(msg: string) => void} [log]
 * @param {string|null} [filterLayer]
 */
export function runEntityRoundtripAll(log = console.log, filterLayer = null) {
  let passed = 0;
  let failed = 0;

  for (const suite of LAYER_SUITES) {
    if (filterLayer && suite.name !== filterLayer) continue;
    const g = primaryGrid(suite);
    const label = `${suite.name} ${g.depth}×${g.rows}×${g.cols}`;
    log(`\n── ${label} (.entity roundtrip) ──`);

    for (const tc of ALL_DTYPES) {
      let net;
      try {
        net = globalThis.createLoomNetwork(suite.build(g, tc.jsonName));
        morphAll(net)(tc.dtype);
        if (typeof net.syncInferenceWeights === 'function') {
          net.syncInferenceWeights();
        }

        const { data: inp, shape: inShape } = suite.makeInput(g);
        const d = f32(inp);
        const sh = JSON.stringify(inShape);

        resetNet(net);
        const out0 = toArray(net.forwardPolymorphic(d, sh));

        const wire = net.serializeEntity();
        if (wire?.error) throw new Error(wire.error);

        const reloaded = globalThis.deserializeLoomEntity(wire);
        if (typeof reloaded === 'string' && reloaded.includes('"error"')) {
          throw new Error(JSON.parse(reloaded).error);
        }

        resetNet(reloaded);
        const out1 = toArray(reloaded.forwardPolymorphic(d, sh));

        const fwdOk = maxAbsDiff(out0, out1) <= tc.tolerance;
        const nativeOk = entityNativeOK(net, wire);

        reloaded.free();
        net.free();
        if (typeof globalThis.loomGC === 'function') globalThis.loomGC();

        if (fwdOk && nativeOk) {
          passed++;
          log(`  ✓ ${tc.name}`);
        } else {
          failed++;
          log(`  ✗ ${tc.name}  fwd=${fwdOk} native=${nativeOk}`);
        }
      } catch (e) {
        failed++;
        if (net) try { net.free(); } catch { /* */ }
        log(`  ✗ ${tc.name}  ${e.message || e}`);
      }
    }
  }

  log(`\n.entity roundtrip: ${passed} passed, ${failed} failed`);
  return failed === 0;
}
