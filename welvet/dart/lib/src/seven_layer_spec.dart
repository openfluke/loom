// Test vectors for lucy/examples/seven_layer — mirrors seven_layer_spec.py.

import 'dart:math' as math;

const int layersPerCell = 7;
const int trainingModeCpuSc = 1;
const int trainingModeCpuMc = 2;

typedef DTypeRow = (String, String, int, double);
typedef Grid = (int, int, int);
typedef InputFn = (List<double>, List<int>) Function(Grid g);

const List<DTypeRow> allDtypes = [
  ('Float64', 'FLOAT64', 0, 1e-3),
  ('Float32', 'FLOAT32', 1, 1e-5),
  ('Float16', 'FLOAT16', 2, 1e-3),
  ('BFloat16', 'BFLOAT16', 3, 1e-3),
  ('FP8-E4M3', 'FP8E4M3', 4, 1e-3),
  ('FP8-E5M2', 'FP8E5M2', 5, 1e-3),
  ('Int64', 'INT64', 6, 1e-3),
  ('Uint64', 'UINT64', 10, 1e-3),
  ('Int32', 'INT32', 7, 1e-3),
  ('Uint32', 'UINT32', 11, 1e-3),
  ('Int16', 'INT16', 8, 1e-3),
  ('Uint16', 'UINT16', 12, 1e-3),
  ('Int8', 'INT8', 9, 1e-3),
  ('Uint8', 'UINT8', 13, 1e-3),
  ('Int4', 'INT4', 14, 1e-3),
  ('Uint4', 'UINT4', 15, 1e-3),
  ('FP4', 'FP4', 16, 1e-3),
  ('Int2', 'INT2', 17, 1e-3),
  ('Uint2', 'UINT2', 18, 1e-3),
  ('Ternary', 'TERNARY', 19, 1e-3),
  ('Binary', 'BINARY', 20, 1e-3),
];

const List<Grid> standardGrids = [(1, 1, 1), (2, 2, 2), (3, 3, 3)];
const List<Grid> convGrids = [(1, 1, 1), (2, 2, 2)];
const List<Grid> cnn3Grids = [(1, 1, 1)];

int gridCells(Grid g) => g.$1 * g.$2 * g.$3;

int trainEpochsForGrid(Grid g) {
  final c = gridCells(g);
  if (c == 1) return 50;
  if (c == 8) return 12;
  return 6;
}

int benchItersForGrid(Grid g) {
  final c = gridCells(g);
  if (c == 1) return 25;
  if (c == 8) return 10;
  return 5;
}

List<int> _seven(List<int> v) => v;
List<int> _flat(int w) => List<int>.filled(layersPerCell + 1, w);

List<int> denseEndpoints(Grid g) {
  final c = gridCells(g);
  if (c == 1) return _seven([16, 24, 32, 48, 64, 48, 32, 8]);
  return _flat(c == 8 ? 8 : 4);
}

List<int> swigluEndpoints(Grid g) {
  final c = gridCells(g);
  if (c == 1) return _seven([32, 32, 32, 32, 32, 32, 32, 16]);
  return _flat(c == 8 ? 16 : 8);
}

List<int> rnnEndpoints(Grid g) {
  final c = gridCells(g);
  if (c == 1) return _seven([16, 24, 32, 32, 32, 24, 16, 8]);
  return _flat(c == 8 ? 8 : 4);
}

List<int> cnnChannelEndpoints(Grid g) {
  if (gridCells(g) == 1) return _seven([3, 6, 8, 8, 8, 16, 16, 16]);
  return _flat(2);
}

int cnnSpatial(Grid g) {
  final c = gridCells(g);
  if (c == 1) return 16;
  if (c == 8) return 8;
  return 4;
}

(int, int, int) mhaShapeFor(Grid g) {
  final c = gridCells(g);
  if (c == 1) return (64, 4, 8);
  if (c == 8) return (16, 2, 4);
  return (8, 2, 4);
}

List<int> embeddingDims(Grid g) {
  final c = gridCells(g);
  if (c == 1) return [32, 32, 32, 24, 16, 12, 8];
  if (c == 8) return List<int>.filled(7, 8);
  return List<int>.filled(7, 4);
}

int embeddingVocab(Grid g) => gridCells(g) == 1 ? 50 : 20;
int embeddingSeqLen(Grid g) => gridCells(g) == 1 ? 8 : 4;

int residualDim(Grid g) {
  final c = gridCells(g);
  if (c == 1) return 32;
  if (c == 8) return 16;
  return 8;
}

String _header(String netId, Grid g) {
  return '{"id":"$netId","depth":${g.$1},"rows":${g.$2},"cols":${g.$3},"layers_per_cell":$layersPerCell,"layers":[';
}

List<(int, int, int)> _cells(Grid g) {
  final out = <(int, int, int)>[];
  for (var z = 0; z < g.$1; z++) {
    for (var y = 0; y < g.$2; y++) {
      for (var x = 0; x < g.$3; x++) {
        out.add((z, y, x));
      }
    }
  }
  return out;
}

String buildDenseJson(Grid g, String dt) {
  final dims = denseEndpoints(g);
  final acts = List<String>.filled(7, 'LINEAR');
  var s = _header('loom-seven-dense', g);
  var first = true;
  for (final (z, y, x) in _cells(g)) {
    for (var i = 0; i < layersPerCell; i++) {
      final layer =
          '{"z":$z,"y":$y,"x":$x,"l":$i,"type":"DENSE","activation":"${acts[i]}","dtype":"$dt","input_height":${dims[i]},"output_height":${dims[i + 1]}}';
      s += first ? layer : ',$layer';
      first = false;
    }
  }
  return '$s]}';
}

String buildSwigluJson(Grid g, String dt) {
  final dims = swigluEndpoints(g);
  var s = _header('loom-seven-swiglu', g);
  var first = true;
  for (final (z, y, x) in _cells(g)) {
    for (var i = 0; i < layersPerCell; i++) {
      final layer =
          '{"z":$z,"y":$y,"x":$x,"l":$i,"type":"SWIGLU","activation":"RELU","dtype":"$dt","input_height":${dims[i]},"output_height":${dims[i + 1]}}';
      s += first ? layer : ',$layer';
      first = false;
    }
  }
  return '$s]}';
}

String buildMhaJson(Grid g, String dt) {
  final (dm, heads, seq) = mhaShapeFor(g);
  var s = _header('loom-seven-mha', g);
  var first = true;
  for (final (z, y, x) in _cells(g)) {
    for (var i = 0; i < layersPerCell; i++) {
      final layer =
          '{"z":$z,"y":$y,"x":$x,"l":$i,"type":"MHA","activation":"RELU","dtype":"$dt","d_model":$dm,"num_heads":$heads,"seq_length":$seq}';
      s += first ? layer : ',$layer';
      first = false;
    }
  }
  return '$s]}';
}

String buildCnn1Json(Grid g, String dt) {
  final ch = cnnChannelEndpoints(g);
  final sp = cnnSpatial(g);
  var s = _header('loom-seven-cnn1', g);
  var first = true;
  for (final (z, y, x) in _cells(g)) {
    for (var i = 0; i < layersPerCell; i++) {
      final layer =
          '{"z":$z,"y":$y,"x":$x,"l":$i,"type":"CNN1","activation":"RELU","dtype":"$dt","input_channels":${ch[i]},"filters":${ch[i + 1]},"input_height":$sp,"output_height":$sp,"kernel_size":3,"stride":1,"padding":1}';
      s += first ? layer : ',$layer';
      first = false;
    }
  }
  return '$s]}';
}

String buildCnn2Json(Grid g, String dt) {
  final ch = cnnChannelEndpoints(g);
  final sp = cnnSpatial(g);
  var s = _header('loom-seven-cnn2', g);
  var first = true;
  for (final (z, y, x) in _cells(g)) {
    for (var i = 0; i < layersPerCell; i++) {
      final layer =
          '{"z":$z,"y":$y,"x":$x,"l":$i,"type":"CNN2","activation":"RELU","dtype":"$dt","input_channels":${ch[i]},"filters":${ch[i + 1]},"input_height":$sp,"input_width":$sp,"output_height":$sp,"output_width":$sp,"kernel_size":3,"stride":1,"padding":1}';
      s += first ? layer : ',$layer';
      first = false;
    }
  }
  return '$s]}';
}

List<int> cnn3ChannelEndpoints(Grid g) {
  if (gridCells(g) == 1) return _seven([2, 4, 4, 4, 8, 8, 8, 8]);
  return _flat(2);
}

String buildCnn3Json(Grid g, String dt) {
  final ch = cnn3ChannelEndpoints(g);
  var s = _header('loom-seven-cnn3', g);
  var first = true;
  for (final (z, y, x) in _cells(g)) {
    for (var i = 0; i < layersPerCell; i++) {
      final layer =
          '{"z":$z,"y":$y,"x":$x,"l":$i,"type":"CNN3","activation":"RELU","dtype":"$dt","input_channels":${ch[i]},"filters":${ch[i + 1]},"input_depth":8,"input_height":8,"input_width":8,"output_depth":8,"output_height":8,"output_width":8,"kernel_size":3,"stride":1,"padding":1}';
      s += first ? layer : ',$layer';
      first = false;
    }
  }
  return '$s]}';
}

String buildRnnJson(Grid g, String dt) {
  final dims = rnnEndpoints(g);
  var s = _header('loom-seven-rnn', g);
  var first = true;
  for (final (z, y, x) in _cells(g)) {
    for (var i = 0; i < layersPerCell; i++) {
      final layer =
          '{"z":$z,"y":$y,"x":$x,"l":$i,"type":"RNN","activation":"TANH","dtype":"$dt","input_height":${dims[i]},"output_height":${dims[i + 1]}}';
      s += first ? layer : ',$layer';
      first = false;
    }
  }
  return '$s]}';
}

String buildLstmJson(Grid g, String dt) {
  final dims = rnnEndpoints(g);
  var s = _header('loom-seven-lstm', g);
  var first = true;
  for (final (z, y, x) in _cells(g)) {
    for (var i = 0; i < layersPerCell; i++) {
      final layer =
          '{"z":$z,"y":$y,"x":$x,"l":$i,"type":"LSTM","activation":"TANH","dtype":"$dt","input_height":${dims[i]},"output_height":${dims[i + 1]}}';
      s += first ? layer : ',$layer';
      first = false;
    }
  }
  return '$s]}';
}

String buildEmbeddingJson(Grid g, String dt) {
  final dims = embeddingDims(g);
  final denseOnly = denseEndpoints(g);
  final vocab = embeddingVocab(g);
  final acts = ['RELU', 'RELU', 'RELU', 'RELU', 'RELU', 'SIGMOID'];
  var s = _header('loom-seven-embedding', g);
  var first = true;

  void add(String layer) {
    s += first ? layer : ',$layer';
    first = false;
  }

  for (final (z, y, x) in _cells(g)) {
    if (z == 0 && y == 0 && x == 0) {
      add(
        '{"z":$z,"y":$y,"x":$x,"l":0,"type":"EMBEDDING","dtype":"$dt","vocab_size":$vocab,"embedding_dim":${dims[0]}}',
      );
      for (var i = 0; i < dims.length - 1; i++) {
        add(
          '{"z":$z,"y":$y,"x":$x,"l":${i + 1},"type":"DENSE","activation":"${acts[i]}","dtype":"$dt","input_height":${dims[i]},"output_height":${dims[i + 1]}}',
        );
      }
    } else {
      for (var i = 0; i < layersPerCell; i++) {
        add(
          '{"z":$z,"y":$y,"x":$x,"l":$i,"type":"DENSE","activation":"${acts[i % acts.length]}","dtype":"$dt","input_height":${denseOnly[i]},"output_height":${denseOnly[i + 1]}}',
        );
      }
    }
  }
  return '$s]}';
}

String buildResidualJson(Grid g, String dt) {
  final dim = residualDim(g);
  var s = _header('loom-seven-residual', g);
  var first = true;
  for (final (z, y, x) in _cells(g)) {
    for (var i = 0; i < layersPerCell; i++) {
      final layer =
          '{"z":$z,"y":$y,"x":$x,"l":$i,"type":"RESIDUAL","dtype":"$dt","input_height":$dim,"output_height":$dim}';
      s += first ? layer : ',$layer';
      first = false;
    }
  }
  return '$s]}';
}

(List<double>, List<int>) sinInput(int batch, List<int> rest) {
  final n = batch * rest.fold<int>(1, (a, b) => a * b);
  final data = List<double>.generate(
    n,
    (i) => 0.2 * math.sin(i * 0.11 + 0.3),
  );
  return (data, [batch, ...rest]);
}

(List<double>, List<int>) embeddingInput(Grid g) {
  final seq = embeddingSeqLen(g);
  final vocab = embeddingVocab(g);
  return (
    List<double>.generate(seq, (i) => (i % vocab).toDouble()),
    [seq, 1],
  );
}

List<double> sinTarget(List<double> out) {
  return List<double>.generate(
    out.length,
    (i) => 0.5 + 0.3 * math.sin(i * 0.17),
  );
}

double maxAbsDiff(List<double> a, List<double> b) {
  final n = math.min(a.length, b.length);
  var m = 0.0;
  for (var i = 0; i < n; i++) {
    final v = (a[i] - b[i]).abs();
    if (!v.isFinite) return double.nan;
    m = math.max(m, v);
  }
  return m;
}

double trainingLr(int dtype) {
  if ([10, 11, 12].contains(dtype)) return 0.0005;
  if ([13, 15, 18].contains(dtype)) return 0.005;
  if (dtype >= 4 && dtype <= 20) return 0.01;
  return 0.05;
}

bool _isUnsignedQuant(int dtype) => [10, 11, 12, 13, 15, 18].contains(dtype);

bool trainingOk(double lossInit, double lossFinal, int dtype) {
  if (!lossInit.isFinite || !lossFinal.isFinite) return false;
  if (lossInit > 0.05 && lossFinal < 1e-9) return false;
  if (lossInit > 1e-3 && (lossFinal > lossInit * 50 || lossFinal > 1e10)) {
    return false;
  }
  if (lossInit < 0.01) {
    if (lossInit < 1e-12 && lossFinal < 1e-12) return false;
    if (lossFinal <= lossInit * 2.0 + 1e-3) return true;
    return dtype >= 4 && dtype <= 20 && lossFinal < 1.0;
  }
  if (lossInit > 0 && lossInit < 2.0 && lossFinal > 0 && lossFinal < 2.0) {
    if ((lossFinal - lossInit).abs() < 0.01 && lossFinal <= lossInit * 1.05) {
      return true;
    }
  }
  if (dtype >= 4 && dtype <= 20) {
    final band = _isUnsignedQuant(dtype) ? 0.22 : 0.15;
    if (lossFinal <= lossInit * (1.0 + band) + 1e-3) return true;
    final rel = (lossFinal - lossInit).abs() / (lossInit.abs() + 1e-9);
    if (rel <= band) return true;
    if (_isUnsignedQuant(dtype) &&
        lossInit < 0.35 &&
        lossFinal >= 0.15 &&
        lossFinal <= 0.45) {
      return true;
    }
    if (_isUnsignedQuant(dtype) &&
        lossInit >= 0.35 &&
        lossInit < 0.55 &&
        lossFinal >= 0.15 &&
        lossFinal <= 0.55 &&
        lossFinal <= lossInit * 1.35 + 1e-3) {
      return true;
    }
    return false;
  }
  return lossFinal < lossInit * 0.99;
}

List<int> targetShape(List<int> inShape, int outLen, {bool isEmbedding = false}) {
  var inVol = 1;
  for (final d in inShape) {
    inVol *= d;
  }
  if (outLen == inVol) return List<int>.from(inShape);
  if (isEmbedding) return [inShape[0], outLen ~/ inShape[0]];
  if (inShape.length == 1) return [outLen];
  return [inShape[0], outLen ~/ inShape[0]];
}

typedef JsonBuilder = String Function(Grid g, String dt);

class LayerSuite {
  LayerSuite({
    required this.name,
    required this.primary,
    required this.grids,
    required this.build,
    required this.makeInput,
    this.isEmbedding = false,
    this.noLearn = false,
  });

  final String name;
  final String primary;
  final List<Grid> grids;
  final JsonBuilder build;
  final InputFn makeInput;
  final bool isEmbedding;
  final bool noLearn;
}

final List<LayerSuite> layerSuites = [
  LayerSuite(
    name: 'Dense',
    primary: 'DENSE',
    grids: standardGrids,
    build: buildDenseJson,
    makeInput: (g) => sinInput(4, [denseEndpoints(g)[0]]),
  ),
  LayerSuite(
    name: 'SwiGLU',
    primary: 'SWIGLU',
    grids: standardGrids,
    build: buildSwigluJson,
    makeInput: (g) => sinInput(4, [swigluEndpoints(g)[0]]),
  ),
  LayerSuite(
    name: 'MHA',
    primary: 'MHA',
    grids: standardGrids,
    build: buildMhaJson,
    makeInput: (g) {
      final (dm, _, seq) = mhaShapeFor(g);
      return sinInput(4, [seq, dm]);
    },
  ),
  LayerSuite(
    name: 'CNN1',
    primary: 'CNN1',
    grids: convGrids,
    build: buildCnn1Json,
    makeInput: (g) => sinInput(4, [cnnChannelEndpoints(g)[0], cnnSpatial(g)]),
  ),
  LayerSuite(
    name: 'CNN2',
    primary: 'CNN2',
    grids: convGrids,
    build: buildCnn2Json,
    makeInput: (g) =>
        sinInput(4, [cnnChannelEndpoints(g)[0], cnnSpatial(g), cnnSpatial(g)]),
  ),
  LayerSuite(
    name: 'CNN3',
    primary: 'CNN3',
    grids: cnn3Grids,
    build: buildCnn3Json,
    makeInput: (g) => sinInput(4, [cnn3ChannelEndpoints(g)[0], 8, 8, 8]),
  ),
  LayerSuite(
    name: 'RNN',
    primary: 'RNN',
    grids: standardGrids,
    build: buildRnnJson,
    makeInput: (g) => sinInput(4, [rnnEndpoints(g)[0]]),
  ),
  LayerSuite(
    name: 'LSTM',
    primary: 'LSTM',
    grids: standardGrids,
    build: buildLstmJson,
    makeInput: (g) => sinInput(4, [rnnEndpoints(g)[0]]),
  ),
  LayerSuite(
    name: 'Embedding',
    primary: 'EMBEDDING',
    grids: standardGrids,
    build: buildEmbeddingJson,
    makeInput: embeddingInput,
    isEmbedding: true,
  ),
  LayerSuite(
    name: 'Residual',
    primary: 'RESIDUAL',
    grids: standardGrids,
    build: buildResidualJson,
    makeInput: (g) => sinInput(4, [residualDim(g)]),
    noLearn: true,
  ),
];
