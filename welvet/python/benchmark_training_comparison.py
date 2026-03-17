import os
import sys
import time
import random

# Add welvet to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
sys.path.insert(0, src_dir)

import welvet
from welvet import (
    Network, train,
    init_wgpu, sync_to_gpu,
    Activation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_dur(seconds):
    """Format duration like Go's time.Duration.Round(ms).String()."""
    ms = round(seconds * 1000)
    if ms < 1000:
        return f"{ms}ms"
    elif ms < 60000:
        s = ms / 1000
        return f"{s:.3f}".rstrip("0").rstrip(".") + "s"
    else:
        mins = ms // 60000
        secs_ms = ms % 60000
        s = secs_ms / 1000
        return f"{mins}m" + f"{s:.3f}".rstrip("0").rstrip(".") + "s"


def _gen_batches(num_batches, batch_size, in_dim, out_dim):
    """Returns (inputs_batched, targets_batched) as list of batches."""
    ins, tgts = [], []
    for _ in range(num_batches):
        b_in = [[random.random() * 2 - 1 for _ in range(in_dim)] for _ in range(batch_size)]
        b_tgt = [[random.random() * 2 - 1 for _ in range(out_dim)] for _ in range(batch_size)]
        ins.append(b_in)
        tgts.append(b_tgt)
    return ins, tgts


# ---------------------------------------------------------------------------
# Network Factories (Matching Go Dimensions)
# ---------------------------------------------------------------------------

def _base_cfg(num_layers):
    return {
        "id": "bench",
        "depth": 1, "rows": 1, "cols": 1,
        "layers_per_cell": num_layers,
    }


def _l(spec, idx):
    """Inject z/y/x/l coordinates so each spec lands in the right cell slot."""
    return {"z": 0, "y": 0, "x": 0, "l": idx, **spec}


def create_large_dense_net():
    cfg = _base_cfg(3)
    cfg["layers"] = [
        _l({"type": "dense", "input_height": 128, "output_height": 512, "activation": "relu"},  0),
        _l({"type": "dense", "input_height": 512, "output_height": 512, "activation": "relu"},  1),
        _l({"type": "dense", "input_height": 512, "output_height": 8,   "activation": "linear"}, 2),
    ]
    return cfg


def create_deep_dense_net():
    cfg = _base_cfg(5)
    cfg["layers"] = [
        _l({"type": "dense", "input_height": 128, "output_height": 512, "activation": "relu"},  0),
        _l({"type": "dense", "input_height": 512, "output_height": 512, "activation": "relu"},  1),
        _l({"type": "dense", "input_height": 512, "output_height": 512, "activation": "relu"},  2),
        _l({"type": "dense", "input_height": 512, "output_height": 512, "activation": "relu"},  3),
        _l({"type": "dense", "input_height": 512, "output_height": 8,   "activation": "linear"}, 4),
    ]
    return cfg


def create_large_cnn1d_net():
    cfg = _base_cfg(3)
    cfg["layers"] = [
        _l({"type": "cnn1", "input_channels": 3,  "input_height": 128, "filters": 32, "kernel_size": 3, "stride": 1, "padding": 1, "output_height": 128, "activation": "relu"}, 0),
        _l({"type": "cnn1", "input_channels": 32, "input_height": 128, "filters": 64, "kernel_size": 3, "stride": 1, "padding": 1, "output_height": 128, "activation": "relu"}, 1),
        _l({"type": "dense", "input_height": 64 * 128, "output_height": 8, "activation": "linear"}, 2),
    ]
    return cfg


def create_large_cnn2d_net():
    cfg = _base_cfg(3)
    cfg["layers"] = [
        _l({"type": "cnn2", "input_channels": 3,  "input_height": 32, "input_width": 32, "filters": 16, "kernel_size": 3, "stride": 1, "padding": 1, "output_height": 32, "output_width": 32, "activation": "relu"}, 0),
        _l({"type": "cnn2", "input_channels": 16, "input_height": 32, "input_width": 32, "filters": 32, "kernel_size": 3, "stride": 1, "padding": 1, "output_height": 32, "output_width": 32, "activation": "relu"}, 1),
        _l({"type": "dense", "input_height": 32 * 32 * 32, "output_height": 8, "activation": "linear"}, 2),
    ]
    return cfg


def create_large_cnn3d_net():
    cfg = _base_cfg(2)
    cfg["layers"] = [
        _l({"type": "cnn3", "input_channels": 2, "input_depth": 8, "input_height": 8, "input_width": 8, "filters": 8, "kernel_size": 3, "stride": 1, "padding": 1, "output_depth": 8, "output_height": 8, "output_width": 8, "activation": "relu"}, 0),
        _l({"type": "dense", "input_height": 8 * 8 * 8 * 8, "output_height": 8, "activation": "linear"}, 1),
    ]
    return cfg


def create_rmsnorm_net():
    cfg = _base_cfg(4)
    cfg["layers"] = [
        _l({"type": "dense",   "input_height": 128, "output_height": 512, "activation": "linear"}, 0),
        _l({"type": "rmsnorm", "input_height": 512, "output_height": 512},                         1),
        _l({"type": "dense",   "input_height": 512, "output_height": 512, "activation": "relu"},   2),
        _l({"type": "dense",   "input_height": 512, "output_height": 8,   "activation": "linear"}, 3),
    ]
    return cfg


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(name, cfg_fn, num_batches, batch_size, in_dim, out_dim, gpu_ok, epochs=20):
    print(f"--- {name} ---")

    cfg = cfg_fn()
    batches_in, batches_tgt = _gen_batches(num_batches, batch_size, in_dim, out_dim)

    # --- CPU ---
    n_cpu = Network(cfg)
    t0 = time.perf_counter()
    losses_cpu = train(n_cpu, batches_in, batches_tgt,
                       epochs=epochs, learning_rate=0.01, use_gpu=False)
    cpu_time = time.perf_counter() - t0

    if not gpu_ok:
        print(f"  CPU: {_fmt_dur(cpu_time)} | GPU: skipped (no device)")
        print()
        return

    # --- GPU ---
    n_gpu = Network(cfg)
    init_wgpu(n_gpu._handle)
    sync_to_gpu(n_gpu._handle)

    t0 = time.perf_counter()
    losses_gpu = train(n_gpu, batches_in, batches_tgt,
                       epochs=epochs, learning_rate=0.01, use_gpu=True)
    gpu_time = time.perf_counter() - t0

    # --- Results (mirroring Go output) ---
    init_loss = losses_cpu[0] if losses_cpu else 0.0
    final_cpu = losses_cpu[-1] if losses_cpu else 0.0
    final_gpu = losses_gpu[-1] if losses_gpu else 0.0

    cpu_imprv = ((init_loss - final_cpu) / init_loss * 100) if init_loss > 0 else 0.0
    gpu_imprv = ((init_loss - final_gpu) / init_loss * 100) if init_loss > 0 else 0.0
    speedup   = cpu_time / gpu_time if gpu_time > 0 else 0.0

    cpu_dur = _fmt_dur(cpu_time)
    gpu_dur = _fmt_dur(gpu_time)

    print(f"  | {'Metric':<12} | {'CPU':<14} | {'GPU':<14} | {'':<8} |")
    print(f"  | {'Time':<12} | {cpu_dur:<14} | {gpu_dur:<14} | Speedup: {speedup:.2f}x |")
    print(f"  | {'Final Loss':<12} | {final_cpu:<14.6f} | {final_gpu:<14.6f} | CPU: {cpu_imprv:+.1f}% / GPU: {gpu_imprv:+.1f}% |")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== M-POLY-VTD Multi-Architecture Training Showdown ===")
    random.seed(42)

    # Pre-init GPU once to warm up device and measure init time
    gpu_ok = False
    gpu_init_time = 0.0
    try:
        dummy = Network(_base_cfg(1))
        t0 = time.perf_counter()
        init_wgpu(dummy._handle)
        gpu_init_time = time.perf_counter() - t0
        gpu_ok = True
        print(f"[OK] GPU initialised in {_fmt_dur(gpu_init_time)} (shared across all tests)")
    except Exception as e:
        print(f"[!!] GPU unavailable: {e} - running CPU-only")
    print()

    print("Layers tested: Dense MLP, CNN 1D, CNN 2D, CNN 3D, RMSNorm MLP, Deep Transformer MLP")
    print("(SwiGLU / MHA / Embedding: GPU backward not yet in DispatchBackwardLayer - skipped here)")
    print()

    epochs = 20

    run_test("Dense MLP (128->512->512->8)",
             create_large_dense_net, 8, 64, 128, 8, gpu_ok, epochs)

    run_test("CNN 1D (3ch*128->32f->64f->Dense->8)",
             create_large_cnn1d_net, 8, 32, 3 * 128, 8, gpu_ok, epochs)

    run_test("CNN 2D (3ch*32*32->16f->32f->Dense->8)",
             create_large_cnn2d_net, 8, 16, 3 * 32 * 32, 8, gpu_ok, epochs)

    run_test("CNN 3D (2ch*8*8*8->8f->Dense->8)",
             create_large_cnn3d_net, 8, 8, 2 * 8 * 8 * 8, 8, gpu_ok, epochs)

    run_test("RMSNorm MLP (128->Dense512->Norm->Dense512->8)",
             create_rmsnorm_net, 8, 64, 128, 8, gpu_ok, epochs)

    run_test("Deep Dense MLP (128->512->512->512->512->8)",
             create_deep_dense_net, 8, 64, 128, 8, gpu_ok, epochs)


if __name__ == "__main__":
    main()
