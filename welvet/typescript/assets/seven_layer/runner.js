/**
 * Seven-layer suite — drives Loom only through welvet bindings (WASM / same API as CABI).
 */

import {
  ALL_DTYPES, LAYER_SUITES, TrainingMode,
  trainEpochsForGrid, benchItersForGrid,
  sinTarget, maxAbsDiff, trainingOK, trainingLR,
  isDenseNativeTrainDType,
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

function configureInferenceNet(net) {
  net.setReleaseFP32MasterWhenIdle(true);
  net.syncInferenceWeights();
}

function configureTrainingNet(net, primary, dtype) {
  net.setUseExactDType(primary === 'DENSE' && isDenseNativeTrainDType(dtype));
}

function prepareTrainNet(net) {
  net.setReleaseFP32MasterWhenIdle(true);
}

function loomGC() {
  if (typeof globalThis.loomGC === 'function') globalThis.loomGC();
}

function forwardBind(net, data, shape, mode, iters) {
  net.setTrainingMode(mode);
  const d = f32(data);
  const sh = JSON.stringify(shape);
  for (let i = 0; i < 3; i++) {
    resetNet(net);
    toArray(net.forwardPolymorphic(d, sh));
  }
  let last = null;
  for (let i = 0; i < iters; i++) {
    resetNet(net);
    last = toArray(net.forwardPolymorphic(d, sh));
  }
  return last;
}

function backwardBind(net, inp, inShape, tgt, tgtShape, mode) {
  net.setTrainingMode(mode);
  resetNet(net);
  const r = JSON.parse(net.backwardPolymorphic(
    f32(inp), JSON.stringify(inShape), f32(tgt), JSON.stringify(tgtShape),
  ));
  if (r.error) throw new Error(r.error);
  return r;
}

/** Target tensor shape: match input rank when volumes agree (MHA/CNN), else [batch, features]. */
function targetShape(inShape, outLen, isEmbedding) {
  const inVol = inShape.reduce((a, b) => a * b, 1);
  if (outLen === inVol) return [...inShape];
  if (isEmbedding) return [inShape[0], outLen / inShape[0]];
  if (inShape.length === 1) return [outLen];
  return [inShape[0], outLen / inShape[0]];
}

function trainBind(net, inp, tgt, inShape, tgtShape, epochs, mode, lr) {
  const batches = [{
    input: { shape: inShape, data: Array.from(f32(inp)) },
    target: { shape: tgtShape, data: Array.from(f32(tgt)) },
  }];
  const cfg = JSON.stringify({
    Epochs: epochs, LearningRate: lr, LossType: 'mse', Mode: mode,
    GradientClip: 1.0, Verbose: false, UseGPU: false,
  });
  const r = JSON.parse(net.train(JSON.stringify(batches), epochs, lr, cfg));
  if (r.error) throw new Error(r.error);
  return r.loss_history || [];
}

function saveReloadBind(net, inp, shape, tol, after) {
  const d = f32(inp);
  const sh = JSON.stringify(shape);
  resetNet(net);
  const out0 = toArray(net.forwardPolymorphic(d, sh));
  const wire = net.serialize();
  const reloaded = globalThis.deserializeLoomNetwork(wire);
  resetNet(reloaded);
  const out1 = toArray(reloaded.forwardPolymorphic(d, sh));
  reloaded.free();
  return maxAbsDiff(out0, out1) <= tol * (after ? 100 : 1);
}

/** Native .entity wire — parity with lucy [7] checkEntitySaveReload. */
function entityNativeOK(net, wire) {
  const n = JSON.parse(net.getInfo()).total_layers;
  for (let i = 0; i < n; i++) {
    const raw = globalThis.layerPersistenceFromEntity(wire, i);
    const r = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (r.error || !r.native || !r.weights) return false;
  }
  return true;
}

function saveReloadEntityBind(net, inp, shape, tol, after) {
  const d = f32(inp);
  const sh = JSON.stringify(shape);
  resetNet(net);
  const out0 = toArray(net.forwardPolymorphic(d, sh));
  const wire = net.serializeEntity();
  if (wire && wire.error) throw new Error(wire.error);
  const reloaded = globalThis.deserializeLoomEntity(wire);
  if (typeof reloaded === 'string' && reloaded.includes('"error"')) {
    throw new Error(JSON.parse(reloaded).error);
  }
  resetNet(reloaded);
  const out1 = toArray(reloaded.forwardPolymorphic(d, sh));
  const fwdOk = maxAbsDiff(out0, out1) <= tol * (after ? 100 : 1);
  const nativeOk = entityNativeOK(net, wire);
  reloaded.free();
  return fwdOk && nativeOk;
}

export function runLayerSuite(suite, log = console.log) {
  let passed = 0, failed = 0;
  for (const g of suite.grids) {
    const epochs = trainEpochsForGrid(g);
    const iters = benchItersForGrid(g);
    const label = `${suite.name} ${g.depth}×${g.rows}×${g.cols}`;
    log(`\n${'═'.repeat(70)}`);
    log(`  ${label} — bindings → Loom (CPU SC/MC · train · JSON + .entity save/reload)`);
    log('═'.repeat(70));

    for (const tc of ALL_DTYPES) {
      log(`  · ${tc.name.padEnd(10)} `);
      try {
        const net = globalThis.createLoomNetwork(suite.build(g, tc.jsonName));
        morphAll(net)(tc.dtype);
        configureTrainingNet(net, suite.primary, tc.dtype);
        configureInferenceNet(net);

        const { data: inp, shape: inShape } = suite.makeInput(g);

        const outSc = forwardBind(net, inp, inShape, TrainingMode.CPUSC, iters);
        const outMc = forwardBind(net, inp, inShape, TrainingMode.CPUMC, iters);
        const fwdScmc = maxAbsDiff(outSc, outMc);

        const tgt = sinTarget(outSc);
        const tgtShape = targetShape(inShape, tgt.length, suite.isEmbedding);

        const bSc = backwardBind(net, inp, inShape, tgt, tgtShape, TrainingMode.CPUSC);
        const bMc = backwardBind(net, inp, inShape, tgt, tgtShape, TrainingMode.CPUMC);
        const bwdScmc = maxAbsDiff(
          [...(bSc.dx || []), ...(bSc.dw || [])],
          [...(bMc.dx || []), ...(bMc.dw || [])],
        );

        const detTol = Math.max(tc.tolerance, 1e-10);
        const detOk = fwdScmc <= detTol && bwdScmc <= detTol * 10;
        const jsonBeforeOk = saveReloadBind(net, inp, inShape, tc.tolerance, false);
        const entityBeforeOk = saveReloadEntityBind(net, inp, inShape, tc.tolerance, false);

        const lr = trainingLR(tc.dtype);
        const netSc = globalThis.createLoomNetwork(suite.build(g, tc.jsonName));
        morphAll(netSc)(tc.dtype);
        configureTrainingNet(netSc, suite.primary, tc.dtype);
        prepareTrainNet(netSc);
        trainBind(netSc, inp, tgt, inShape, tgtShape, epochs, TrainingMode.CPUSC, lr);
        netSc.free();

        const netMc = globalThis.createLoomNetwork(suite.build(g, tc.jsonName));
        morphAll(netMc)(tc.dtype);
        configureTrainingNet(netMc, suite.primary, tc.dtype);
        prepareTrainNet(netMc);
        const hist = trainBind(netMc, inp, tgt, inShape, tgtShape, epochs, TrainingMode.CPUMC, lr);

        const lossInit = hist[0] ?? 0;
        const lossFinal = hist[hist.length - 1] ?? 0;
        const requiresLearn = suite.primary !== 'RESIDUAL' && !suite.noLearn;
        const learned = trainingOK(lossInit, lossFinal, tc.dtype) || !requiresLearn;
        const jsonAfterOk = saveReloadBind(netMc, inp, inShape, tc.tolerance, true);
        const entityAfterOk = saveReloadEntityBind(netMc, inp, inShape, tc.tolerance, true);
        const overall = jsonBeforeOk && entityBeforeOk && jsonAfterOk && entityAfterOk && learned && detOk;

        net.free();
        netMc.free();
        loomGC();

        if (overall) {
          passed++;
          log(`PASS  loss ${lossInit.toExponential(4)}→${lossFinal.toExponential(4)} det=${detOk} json=${jsonAfterOk} entity=${entityAfterOk}`);
        } else {
          failed++;
          log(`FAIL  loss ${lossInit.toExponential(4)}→${lossFinal.toExponential(4)} learn=${learned} json=${jsonBeforeOk && jsonAfterOk} entity=${entityBeforeOk && entityAfterOk} det=${detOk}`);
        }
      } catch (e) {
        failed++;
        log(`ERR   ${e.message || e}`);
        loomGC();
      }
    }
  }
  return failed === 0;
}

export function runAllSuites(log = console.log, filterName = null) {
  let ok = true;
  for (const suite of LAYER_SUITES) {
    if (filterName && suite.name !== filterName) continue;
    if (!runLayerSuite(suite, log)) ok = false;
  }
  return ok;
}
