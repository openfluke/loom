# Test vectors for lucy/examples/seven_layer — no Loom calls here, only JSON + criteria.

import math
from typing import Callable, List, Tuple

LAYERS_PER_CELL = 7
TRAINING_MODE_CPU_SC = 1
TRAINING_MODE_CPU_MC = 2

ALL_DTYPES = [
    ("Float64", "FLOAT64", 0, 1e-3),
    ("Float32", "FLOAT32", 1, 1e-5),
    ("Float16", "FLOAT16", 2, 1e-3),
    ("BFloat16", "BFLOAT16", 3, 1e-3),
    ("FP8-E4M3", "FP8E4M3", 4, 1e-3),
    ("FP8-E5M2", "FP8E5M2", 5, 1e-3),
    ("Int64", "INT64", 6, 1e-3),
    ("Uint64", "UINT64", 10, 1e-3),
    ("Int32", "INT32", 7, 1e-3),
    ("Uint32", "UINT32", 11, 1e-3),
    ("Int16", "INT16", 8, 1e-3),
    ("Uint16", "UINT16", 12, 1e-3),
    ("Int8", "INT8", 9, 1e-3),
    ("Uint8", "UINT8", 13, 1e-3),
    ("Int4", "INT4", 14, 1e-3),
    ("Uint4", "UINT4", 15, 1e-3),
    ("FP4", "FP4", 16, 1e-3),
    ("Int2", "INT2", 17, 1e-3),
    ("Uint2", "UINT2", 18, 1e-3),
    ("Ternary", "TERNARY", 19, 1e-3),
    ("Binary", "BINARY", 20, 1e-3),
]

STANDARD_GRIDS = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
CONV_GRIDS = [(1, 1, 1), (2, 2, 2)]
CNN3_GRIDS = [(1, 1, 1)]


def grid_cells(g):
    return g[0] * g[1] * g[2]


def train_epochs_for_grid(g):
    c = grid_cells(g)
    return 50 if c == 1 else (12 if c == 8 else 6)


def bench_iters_for_grid(g):
    c = grid_cells(g)
    return 25 if c == 1 else (10 if c == 8 else 5)


def _seven(v):
    return v


def _flat(w):
    return [w] * (LAYERS_PER_CELL + 1)


def dense_endpoints(g):
    c = grid_cells(g)
    if c == 1:
        return _seven([16, 24, 32, 48, 64, 48, 32, 8])
    return _flat(8 if c == 8 else 4)


def swiglu_endpoints(g):
    c = grid_cells(g)
    if c == 1:
        return _seven([32] * 7 + [16])
    return _flat(16 if c == 8 else 8)


def rnn_endpoints(g):
    c = grid_cells(g)
    if c == 1:
        return _seven([16, 24, 32, 32, 32, 24, 16, 8])
    return _flat(8 if c == 8 else 4)


def cnn_channel_endpoints(g):
    if grid_cells(g) == 1:
        return _seven([3, 6, 8, 8, 8, 16, 16, 16])
    return _flat(2)


def cnn_spatial(g):
    c = grid_cells(g)
    return 16 if c == 1 else (8 if c == 8 else 4)


def mha_shape_for(g):
    c = grid_cells(g)
    if c == 1:
        return 64, 4, 8
    if c == 8:
        return 16, 2, 4
    return 8, 2, 4


def embedding_dims(g):
    c = grid_cells(g)
    if c == 1:
        return [32, 32, 32, 24, 16, 12, 8]
    return [8] * 7 if c == 8 else [4] * 7


def embedding_vocab(g):
    return 50 if grid_cells(g) == 1 else 20


def embedding_seq_len(g):
    return 8 if grid_cells(g) == 1 else 4


def residual_dim(g):
    c = grid_cells(g)
    return 32 if c == 1 else (16 if c == 8 else 8)


def _header(net_id, g):
    d, r, c = g
    return f'{{"id":"{net_id}","depth":{d},"rows":{r},"cols":{c},"layers_per_cell":{LAYERS_PER_CELL},"layers":['


def _cells(g):
    d, r, c = g
    for z in range(d):
        for y in range(r):
            for x in range(c):
                yield z, y, x


def build_dense_json(g, dt):
    dims = dense_endpoints(g)
    acts = ["LINEAR"] * 7
    s = _header("loom-seven-dense", g)
    first = True
    for z, y, x in _cells(g):
        for i in range(LAYERS_PER_CELL):
            L = f'{{"z":{z},"y":{y},"x":{x},"l":{i},"type":"DENSE","activation":"{acts[i]}","dtype":"{dt}","input_height":{dims[i]},"output_height":{dims[i+1]}}}'
            s += ("" if first else ",") + L
            first = False
    return s + "]}"


def build_swiglu_json(g, dt):
    dims = swiglu_endpoints(g)
    s = _header("loom-seven-swiglu", g)
    first = True
    for z, y, x in _cells(g):
        for i in range(LAYERS_PER_CELL):
            L = f'{{"z":{z},"y":{y},"x":{x},"l":{i},"type":"SWIGLU","activation":"RELU","dtype":"{dt}","input_height":{dims[i]},"output_height":{dims[i+1]}}}'
            s += ("" if first else ",") + L
            first = False
    return s + "]}"


def build_mha_json(g, dt):
    dm, heads, seq = mha_shape_for(g)
    s = _header("loom-seven-mha", g)
    first = True
    for z, y, x in _cells(g):
        for i in range(LAYERS_PER_CELL):
            L = f'{{"z":{z},"y":{y},"x":{x},"l":{i},"type":"MHA","activation":"RELU","dtype":"{dt}","d_model":{dm},"num_heads":{heads},"seq_length":{seq}}}'
            s += ("" if first else ",") + L
            first = False
    return s + "]}"


def build_cnn1_json(g, dt):
    ch, sp = cnn_channel_endpoints(g), cnn_spatial(g)
    s = _header("loom-seven-cnn1", g)
    first = True
    for z, y, x in _cells(g):
        for i in range(LAYERS_PER_CELL):
            L = f'{{"z":{z},"y":{y},"x":{x},"l":{i},"type":"CNN1","activation":"RELU","dtype":"{dt}","input_channels":{ch[i]},"filters":{ch[i+1]},"input_height":{sp},"output_height":{sp},"kernel_size":3,"stride":1,"padding":1}}'
            s += ("" if first else ",") + L
            first = False
    return s + "]}"


def build_cnn2_json(g, dt):
    ch, sp = cnn_channel_endpoints(g), cnn_spatial(g)
    s = _header("loom-seven-cnn2", g)
    first = True
    for z, y, x in _cells(g):
        for i in range(LAYERS_PER_CELL):
            L = f'{{"z":{z},"y":{y},"x":{x},"l":{i},"type":"CNN2","activation":"RELU","dtype":"{dt}","input_channels":{ch[i]},"filters":{ch[i+1]},"input_height":{sp},"input_width":{sp},"output_height":{sp},"output_width":{sp},"kernel_size":3,"stride":1,"padding":1}}'
            s += ("" if first else ",") + L
            first = False
    return s + "]}"


def cnn3_channel_endpoints(g):
    if grid_cells(g) == 1:
        return _seven([2, 4, 4, 4, 8, 8, 8, 8])
    return _flat(2)


def build_cnn3_json(g, dt):
    ch = cnn3_channel_endpoints(g)
    s = _header("loom-seven-cnn3", g)
    first = True
    for z, y, x in _cells(g):
        for i in range(LAYERS_PER_CELL):
            L = f'{{"z":{z},"y":{y},"x":{x},"l":{i},"type":"CNN3","activation":"RELU","dtype":"{dt}","input_channels":{ch[i]},"filters":{ch[i+1]},"input_depth":8,"input_height":8,"input_width":8,"output_depth":8,"output_height":8,"output_width":8,"kernel_size":3,"stride":1,"padding":1}}'
            s += ("" if first else ",") + L
            first = False
    return s + "]}"


def build_rnn_json(g, dt):
    dims = rnn_endpoints(g)
    s = _header("loom-seven-rnn", g)
    first = True
    for z, y, x in _cells(g):
        for i in range(LAYERS_PER_CELL):
            L = f'{{"z":{z},"y":{y},"x":{x},"l":{i},"type":"RNN","activation":"TANH","dtype":"{dt}","input_height":{dims[i]},"output_height":{dims[i+1]}}}'
            s += ("" if first else ",") + L
            first = False
    return s + "]}"


def build_lstm_json(g, dt):
    dims = rnn_endpoints(g)
    s = _header("loom-seven-lstm", g)
    first = True
    for z, y, x in _cells(g):
        for i in range(LAYERS_PER_CELL):
            L = f'{{"z":{z},"y":{y},"x":{x},"l":{i},"type":"LSTM","activation":"TANH","dtype":"{dt}","input_height":{dims[i]},"output_height":{dims[i+1]}}}'
            s += ("" if first else ",") + L
            first = False
    return s + "]}"


def build_embedding_json(g, dt):
    dims = embedding_dims(g)
    dense_only = dense_endpoints(g)
    vocab = embedding_vocab(g)
    acts = ["RELU"] * 5 + ["SIGMOID"]
    s = _header("loom-seven-embedding", g)
    first = True

    def add(L):
        nonlocal s, first
        s += ("" if first else ",") + L
        first = False

    for z, y, x in _cells(g):
        if z == y == x == 0:
            add(f'{{"z":{z},"y":{y},"x":{x},"l":0,"type":"EMBEDDING","dtype":"{dt}","vocab_size":{vocab},"embedding_dim":{dims[0]}}}')
            for i in range(len(dims) - 1):
                add(f'{{"z":{z},"y":{y},"x":{x},"l":{i+1},"type":"DENSE","activation":"{acts[i]}","dtype":"{dt}","input_height":{dims[i]},"output_height":{dims[i+1]}}}')
        else:
            for i in range(LAYERS_PER_CELL):
                add(f'{{"z":{z},"y":{y},"x":{x},"l":{i},"type":"DENSE","activation":"{acts[i % len(acts)]}","dtype":"{dt}","input_height":{dense_only[i]},"output_height":{dense_only[i+1]}}}')
    return s + "]}"


def build_residual_json(g, dt):
    dim = residual_dim(g)
    s = _header("loom-seven-residual", g)
    first = True
    for z, y, x in _cells(g):
        for i in range(LAYERS_PER_CELL):
            L = f'{{"z":{z},"y":{y},"x":{x},"l":{i},"type":"RESIDUAL","dtype":"{dt}","input_height":{dim},"output_height":{dim}}}'
            s += ("" if first else ",") + L
            first = False
    return s + "]}"


def sin_input(batch, *rest):
    n = batch * math.prod(rest)
    return [0.2 * math.sin(i * 0.11 + 0.3) for i in range(n)], [batch, *rest]


def sin_target(out):
    return [0.5 + 0.3 * math.sin(i * 0.17) for i in range(len(out))]


def embedding_input(g):
    seq, vocab = embedding_seq_len(g), embedding_vocab(g)
    return [float(i % vocab) for i in range(seq)], [seq, 1]


def max_abs_diff(a, b):
    n = min(len(a), len(b))
    m = 0.0
    for i in range(n):
        v = abs(a[i] - b[i])
        if not math.isfinite(v):
            return float("nan")
        m = max(m, v)
    return m


def mse_loss(out, tgt):
    n = min(len(out), len(tgt))
    return sum((out[i] - tgt[i]) ** 2 for i in range(n)) / n


def training_lr(dtype):
    if dtype in (10, 11, 12):
        return 0.0005
    if dtype in (13, 15, 18):
        return 0.005
    if 4 <= dtype <= 20:
        return 0.01
    if dtype in (4, 5, 16):
        return 0.01
    return 0.05


def training_ok(loss_init, loss_final, dtype):
    if not math.isfinite(loss_init) or not math.isfinite(loss_final):
        return False
    if loss_init > 0.05 and loss_final < 1e-9:
        return False
    if loss_init > 1e-3 and (loss_final > loss_init * 50 or loss_final > 1e10):
        return False
    if loss_init < 0.01:
        if loss_init < 1e-12 and loss_final < 1e-12:
            return False
        if loss_final <= loss_init * 2.0 + 1e-3:
            return True
        return 4 <= dtype <= 20 and loss_final < 1.0
    if 0 < loss_init < 2.0 and 0 < loss_final < 2.0:
        if abs(loss_final - loss_init) < 0.01 and loss_final <= loss_init * 1.05:
            return True
    if 4 <= dtype <= 20:
        band = 0.22 if dtype in (10, 11, 12, 13, 15, 18) else 0.15
        if loss_final <= loss_init * (1.0 + band) + 1e-3:
            return True
        rel = abs(loss_final - loss_init) / (abs(loss_init) + 1e-9)
        if rel <= band:
            return True
        if dtype in (10, 11, 12, 13, 15, 18) and loss_init < 0.35 and 0.15 <= loss_final <= 0.45:
            return True
        return False
    return loss_final < loss_init * 0.99


class Suite:
    def __init__(self, name, primary, grids, build, make_input, is_embedding=False, no_learn=False):
        self.name = name
        self.primary = primary
        self.grids = grids
        self.build = build
        self.make_input = make_input
        self.is_embedding = is_embedding
        self.no_learn = no_learn


LAYER_SUITES = [
    Suite("Dense", "DENSE", STANDARD_GRIDS, build_dense_json, lambda g: sin_input(4, dense_endpoints(g)[0])),
    Suite("SwiGLU", "SWIGLU", STANDARD_GRIDS, build_swiglu_json, lambda g: sin_input(4, swiglu_endpoints(g)[0])),
    Suite("MHA", "MHA", STANDARD_GRIDS, build_mha_json, lambda g: sin_input(4, mha_shape_for(g)[2], mha_shape_for(g)[0])),
    Suite("CNN1", "CNN1", CONV_GRIDS, build_cnn1_json, lambda g: sin_input(4, cnn_channel_endpoints(g)[0], cnn_spatial(g))),
    Suite("CNN2", "CNN2", CONV_GRIDS, build_cnn2_json, lambda g: sin_input(4, cnn_channel_endpoints(g)[0], cnn_spatial(g), cnn_spatial(g))),
    Suite("CNN3", "CNN3", CNN3_GRIDS, build_cnn3_json, lambda g: sin_input(4, cnn3_channel_endpoints(g)[0], 8, 8, 8)),
    Suite("RNN", "RNN", STANDARD_GRIDS, build_rnn_json, lambda g: sin_input(4, rnn_endpoints(g)[0])),
    Suite("LSTM", "LSTM", STANDARD_GRIDS, build_lstm_json, lambda g: sin_input(4, rnn_endpoints(g)[0])),
    Suite("Embedding", "EMBEDDING", STANDARD_GRIDS, build_embedding_json, embedding_input, True),
    Suite("Residual", "RESIDUAL", STANDARD_GRIDS, build_residual_json, lambda g: sin_input(4, residual_dim(g)), no_learn=True),
]
