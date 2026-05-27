/** Seven-layer test vectors (lucy/examples/seven_layer). No Loom calls — bindings only in runner. */

export const LAYERS_PER_CELL = 7;
export const TrainingMode = { CPUSC: 1, CPUMC: 2 };

export const ALL_DTYPES = [
  { name: 'Float64', jsonName: 'FLOAT64', dtype: 0, tolerance: 1e-3 },
  { name: 'Float32', jsonName: 'FLOAT32', dtype: 1, tolerance: 1e-5 },
  { name: 'Float16', jsonName: 'FLOAT16', dtype: 2, tolerance: 1e-3 },
  { name: 'BFloat16', jsonName: 'BFLOAT16', dtype: 3, tolerance: 1e-3 },
  { name: 'FP8-E4M3', jsonName: 'FP8E4M3', dtype: 4, tolerance: 1e-3 },
  { name: 'FP8-E5M2', jsonName: 'FP8E5M2', dtype: 5, tolerance: 1e-3 },
  { name: 'Int64', jsonName: 'INT64', dtype: 6, tolerance: 1e-3 },
  { name: 'Uint64', jsonName: 'UINT64', dtype: 10, tolerance: 1e-3 },
  { name: 'Int32', jsonName: 'INT32', dtype: 7, tolerance: 1e-3 },
  { name: 'Uint32', jsonName: 'UINT32', dtype: 11, tolerance: 1e-3 },
  { name: 'Int16', jsonName: 'INT16', dtype: 8, tolerance: 1e-3 },
  { name: 'Uint16', jsonName: 'UINT16', dtype: 12, tolerance: 1e-3 },
  { name: 'Int8', jsonName: 'INT8', dtype: 9, tolerance: 1e-3 },
  { name: 'Uint8', jsonName: 'UINT8', dtype: 13, tolerance: 1e-3 },
  { name: 'Int4', jsonName: 'INT4', dtype: 14, tolerance: 1e-3 },
  { name: 'Uint4', jsonName: 'UINT4', dtype: 15, tolerance: 1e-3 },
  { name: 'FP4', jsonName: 'FP4', dtype: 16, tolerance: 1e-3 },
  { name: 'Int2', jsonName: 'INT2', dtype: 17, tolerance: 1e-3 },
  { name: 'Uint2', jsonName: 'UINT2', dtype: 18, tolerance: 1e-3 },
  { name: 'Ternary', jsonName: 'TERNARY', dtype: 19, tolerance: 1e-3 },
  { name: 'Binary', jsonName: 'BINARY', dtype: 20, tolerance: 1e-3 },
];

export const StandardGrids = [{ depth: 1, rows: 1, cols: 1 }, { depth: 2, rows: 2, cols: 2 }, { depth: 3, rows: 3, cols: 3 }];
export const ConvGrids = [{ depth: 1, rows: 1, cols: 1 }, { depth: 2, rows: 2, cols: 2 }];
export const CNN3Grids = [{ depth: 1, rows: 1, cols: 1 }];

const cells = g => g.depth * g.rows * g.cols;
export const trainEpochsForGrid = g => { const c = cells(g); return c === 1 ? 50 : c === 8 ? 12 : 6; };
export const benchItersForGrid = g => { const c = cells(g); return c === 1 ? 25 : c === 8 ? 10 : 5; };

function flat(w) { return Array(LAYERS_PER_CELL + 1).fill(w); }
function denseEndpoints(g) {
  const c = cells(g);
  if (c === 1) return [16, 24, 32, 48, 64, 48, 32, 8];
  return flat(c === 8 ? 8 : 4);
}
function swigluEndpoints(g) {
  const c = cells(g);
  if (c === 1) return [...Array(7).fill(32), 16];
  return flat(c === 8 ? 16 : 8);
}
function rnnEndpoints(g) {
  const c = cells(g);
  if (c === 1) return [16, 24, 32, 32, 32, 24, 16, 8];
  return flat(c === 8 ? 8 : 4);
}
function cnnCh(g) {
  if (cells(g) === 1) return [3, 6, 8, 8, 8, 16, 16, 16];
  return flat(2);
}
function cnn3Ch(g) {
  if (cells(g) === 1) return [2, 4, 4, 4, 8, 8, 8, 8];
  return flat(2);
}
function cnnSp(g) { const c = cells(g); return c === 1 ? 16 : c === 8 ? 8 : 4; }
function mhaShape(g) {
  const c = cells(g);
  if (c === 1) return { dModel: 64, heads: 4, seq: 8 };
  if (c === 8) return { dModel: 16, heads: 2, seq: 4 };
  return { dModel: 8, heads: 2, seq: 4 };
}
function embDims(g) {
  const c = cells(g);
  if (c === 1) return [32, 32, 32, 24, 16, 12, 8];
  return Array(7).fill(c === 8 ? 8 : 4);
}
function embVocab(g) { return cells(g) === 1 ? 50 : 20; }
function embSeq(g) { return cells(g) === 1 ? 8 : 4; }
function resDim(g) { const c = cells(g); return c === 1 ? 32 : c === 8 ? 16 : 8; }

function header(id, g) {
  return `{"id":"${id}","depth":${g.depth},"rows":${g.rows},"cols":${g.cols},"layers_per_cell":${LAYERS_PER_CELL},"layers":[`;
}
function eachCell(g, fn) {
  for (let z = 0; z < g.depth; z++)
    for (let y = 0; y < g.rows; y++)
      for (let x = 0; x < g.cols; x++) fn(z, y, x);
}

export function buildDenseJSON(g, dt) {
  const dims = denseEndpoints(g);
  let s = header('loom-seven-dense', g), first = true;
  eachCell(g, (z, y, x) => {
    for (let i = 0; i < LAYERS_PER_CELL; i++) {
      const L = `{"z":${z},"y":${y},"x":${x},"l":${i},"type":"DENSE","activation":"LINEAR","dtype":"${dt}","input_height":${dims[i]},"output_height":${dims[i + 1]}}`;
      s += (first ? '' : ',') + L; first = false;
    }
  });
  return s + ']}';
}

export function buildSwiGLUJSON(g, dt) {
  const dims = swigluEndpoints(g);
  let s = header('loom-seven-swiglu', g), first = true;
  eachCell(g, (z, y, x) => {
    for (let i = 0; i < LAYERS_PER_CELL; i++) {
      const L = `{"z":${z},"y":${y},"x":${x},"l":${i},"type":"SWIGLU","activation":"RELU","dtype":"${dt}","input_height":${dims[i]},"output_height":${dims[i + 1]}}`;
      s += (first ? '' : ',') + L; first = false;
    }
  });
  return s + ']}';
}

export function buildMHAJSON(g, dt) {
  const m = mhaShape(g);
  let s = header('loom-seven-mha', g), first = true;
  eachCell(g, (z, y, x) => {
    for (let i = 0; i < LAYERS_PER_CELL; i++) {
      const L = `{"z":${z},"y":${y},"x":${x},"l":${i},"type":"MHA","activation":"RELU","dtype":"${dt}","d_model":${m.dModel},"num_heads":${m.heads},"seq_length":${m.seq}}`;
      s += (first ? '' : ',') + L; first = false;
    }
  });
  return s + ']}';
}

export function buildCNN1JSON(g, dt) {
  const ch = cnnCh(g), sp = cnnSp(g);
  let s = header('loom-seven-cnn1', g), first = true;
  eachCell(g, (z, y, x) => {
    for (let i = 0; i < LAYERS_PER_CELL; i++) {
      const L = `{"z":${z},"y":${y},"x":${x},"l":${i},"type":"CNN1","activation":"RELU","dtype":"${dt}","input_channels":${ch[i]},"filters":${ch[i + 1]},"input_height":${sp},"output_height":${sp},"kernel_size":3,"stride":1,"padding":1}`;
      s += (first ? '' : ',') + L; first = false;
    }
  });
  return s + ']}';
}

export function buildCNN2JSON(g, dt) {
  const ch = cnnCh(g), sp = cnnSp(g);
  let s = header('loom-seven-cnn2', g), first = true;
  eachCell(g, (z, y, x) => {
    for (let i = 0; i < LAYERS_PER_CELL; i++) {
      const L = `{"z":${z},"y":${y},"x":${x},"l":${i},"type":"CNN2","activation":"RELU","dtype":"${dt}","input_channels":${ch[i]},"filters":${ch[i + 1]},"input_height":${sp},"input_width":${sp},"output_height":${sp},"output_width":${sp},"kernel_size":3,"stride":1,"padding":1}`;
      s += (first ? '' : ',') + L; first = false;
    }
  });
  return s + ']}';
}

export function buildCNN3JSON(g, dt) {
  const ch = cnn3Ch(g);
  let s = header('loom-seven-cnn3', g), first = true;
  eachCell(g, (z, y, x) => {
    for (let i = 0; i < LAYERS_PER_CELL; i++) {
      const L = `{"z":${z},"y":${y},"x":${x},"l":${i},"type":"CNN3","activation":"RELU","dtype":"${dt}","input_channels":${ch[i]},"filters":${ch[i + 1]},"input_depth":8,"input_height":8,"input_width":8,"output_depth":8,"output_height":8,"output_width":8,"kernel_size":3,"stride":1,"padding":1}`;
      s += (first ? '' : ',') + L; first = false;
    }
  });
  return s + ']}';
}

export function buildRNNJSON(g, dt) {
  const dims = rnnEndpoints(g);
  let s = header('loom-seven-rnn', g), first = true;
  eachCell(g, (z, y, x) => {
    for (let i = 0; i < LAYERS_PER_CELL; i++) {
      const L = `{"z":${z},"y":${y},"x":${x},"l":${i},"type":"RNN","activation":"TANH","dtype":"${dt}","input_height":${dims[i]},"output_height":${dims[i + 1]}}`;
      s += (first ? '' : ',') + L; first = false;
    }
  });
  return s + ']}';
}

export function buildLSTMJSON(g, dt) {
  const dims = rnnEndpoints(g);
  let s = header('loom-seven-lstm', g), first = true;
  eachCell(g, (z, y, x) => {
    for (let i = 0; i < LAYERS_PER_CELL; i++) {
      const L = `{"z":${z},"y":${y},"x":${x},"l":${i},"type":"LSTM","activation":"TANH","dtype":"${dt}","input_height":${dims[i]},"output_height":${dims[i + 1]}}`;
      s += (first ? '' : ',') + L; first = false;
    }
  });
  return s + ']}';
}

export function buildEmbeddingJSON(g, dt) {
  const dims = embDims(g), dense = denseEndpoints(g), vocab = embVocab(g);
  const acts = ['RELU', 'RELU', 'RELU', 'RELU', 'RELU', 'SIGMOID'];
  let s = header('loom-seven-embedding', g), first = true;
  const add = L => { s += (first ? '' : ',') + L; first = false; };
  eachCell(g, (z, y, x) => {
    if (z === 0 && y === 0 && x === 0) {
      add(`{"z":0,"y":0,"x":0,"l":0,"type":"EMBEDDING","dtype":"${dt}","vocab_size":${vocab},"embedding_dim":${dims[0]}}`);
      for (let i = 0; i < dims.length - 1; i++)
        add(`{"z":0,"y":0,"x":0,"l":${i + 1},"type":"DENSE","activation":"${acts[i]}","dtype":"${dt}","input_height":${dims[i]},"output_height":${dims[i + 1]}}`);
    } else {
      for (let i = 0; i < LAYERS_PER_CELL; i++)
        add(`{"z":${z},"y":${y},"x":${x},"l":${i},"type":"DENSE","activation":"${acts[i % acts.length]}","dtype":"${dt}","input_height":${dense[i]},"output_height":${dense[i + 1]}}`);
    }
  });
  return s + ']}';
}

export function buildResidualJSON(g, dt) {
  const dim = resDim(g);
  let s = header('loom-seven-residual', g), first = true;
  eachCell(g, (z, y, x) => {
    for (let i = 0; i < LAYERS_PER_CELL; i++) {
      const L = `{"z":${z},"y":${y},"x":${x},"l":${i},"type":"RESIDUAL","dtype":"${dt}","input_height":${dim},"output_height":${dim}}`;
      s += (first ? '' : ',') + L; first = false;
    }
  });
  return s + ']}';
}

export function sinInput(batch, ...rest) {
  const n = batch * rest.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = 0.2 * Math.sin(i * 0.11 + 0.3);
  return { data, shape: [batch, ...rest] };
}

export function sinTarget(out) {
  const data = new Float32Array(out.length);
  for (let i = 0; i < data.length; i++) data[i] = 0.5 + 0.3 * Math.sin(i * 0.17);
  return data;
}

export function embeddingInput(g) {
  const seq = embSeq(g), vocab = embVocab(g);
  const data = new Float32Array(seq);
  for (let i = 0; i < seq; i++) data[i] = i % vocab;
  return { data, shape: [seq, 1] };
}

export function maxAbsDiff(a, b) {
  const n = Math.min(a.length, b.length);
  let m = 0;
  for (let i = 0; i < n; i++) {
    const v = Math.abs(a[i] - b[i]);
    if (!Number.isFinite(v)) return NaN;
    if (v > m) m = v;
  }
  return m;
}

export function trainingLR(dtype) {
  if ([10, 11, 12].includes(dtype)) return 0.0005;
  if ([13, 15, 18].includes(dtype)) return 0.005;
  if (dtype >= 4 && dtype <= 20) return 0.01;
  return 0.05;
}

export function trainingOK(li, lf, dtype) {
  if (!Number.isFinite(li) || !Number.isFinite(lf)) return false;
  if (li > 0.05 && lf < 1e-9) return false;
  if (li < 0.01) {
    if (li < 1e-12 && lf < 1e-12) return false;
    if (lf <= li * 2 + 1e-3) return true;
    return dtype >= 4 && lf < 1;
  }
  if (dtype >= 4 && dtype <= 20) {
    const band = [10, 11, 12, 13, 15, 18].includes(dtype) ? 0.22 : 0.15;
    if (lf <= li * (1 + band) + 1e-3) return true;
    return false;
  }
  return lf < li * 0.99;
}

export const LAYER_SUITES = [
  { name: 'Dense', primary: 'DENSE', grids: StandardGrids, build: buildDenseJSON,
    makeInput: g => { const d = denseEndpoints(g); return sinInput(4, d[0]); } },
  { name: 'SwiGLU', primary: 'SWIGLU', grids: StandardGrids, build: buildSwiGLUJSON,
    makeInput: g => { const d = swigluEndpoints(g); return sinInput(4, d[0]); } },
  { name: 'MHA', primary: 'MHA', grids: StandardGrids, build: buildMHAJSON,
    makeInput: g => { const m = mhaShape(g); return sinInput(4, m.seq, m.dModel); } },
  { name: 'CNN1', primary: 'CNN1', grids: ConvGrids, build: buildCNN1JSON,
    makeInput: g => { const ch = cnnCh(g); return sinInput(4, ch[0], cnnSp(g)); } },
  { name: 'CNN2', primary: 'CNN2', grids: ConvGrids, build: buildCNN2JSON,
    makeInput: g => { const ch = cnnCh(g), sp = cnnSp(g); return sinInput(4, ch[0], sp, sp); } },
  { name: 'CNN3', primary: 'CNN3', grids: CNN3Grids, build: buildCNN3JSON,
    makeInput: g => { const ch = cnn3Ch(g); return sinInput(4, ch[0], 8, 8, 8); } },
  { name: 'RNN', primary: 'RNN', grids: StandardGrids, build: buildRNNJSON,
    makeInput: g => { const d = rnnEndpoints(g); return sinInput(4, d[0]); } },
  { name: 'LSTM', primary: 'LSTM', grids: StandardGrids, build: buildLSTMJSON,
    makeInput: g => { const d = rnnEndpoints(g); return sinInput(4, d[0]); } },
  { name: 'Embedding', primary: 'EMBEDDING', grids: StandardGrids, build: buildEmbeddingJSON,
    makeInput: embeddingInput, isEmbedding: true },
  { name: 'Residual', primary: 'RESIDUAL', grids: StandardGrids, build: buildResidualJSON,
    makeInput: g => sinInput(4, resDim(g)), noLearn: true },
];
