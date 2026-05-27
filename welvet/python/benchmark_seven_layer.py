#!/usr/bin/env python3
"""
Seven-layer CPU suite via welvet → Loom CABI (.so).

All Loom work goes through ctypes bindings (LoomBuildNetworkFromJSON, LoomForwardPolymorphic,
LoomBackwardPolymorphic, LoomTrain, LoomSerializeNetwork, …). Test criteria live here and in
seven_layer_spec.py — not in cabi/.
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from welvet import Network, train
from seven_layer_spec import (
    ALL_DTYPES,
    LAYER_SUITES,
    TRAINING_MODE_CPU_SC,
    TRAINING_MODE_CPU_MC,
    bench_iters_for_grid,
    train_epochs_for_grid,
    sin_target,
    max_abs_diff,
    mse_loss,
    training_ok,
    training_lr,
)


def _rows(flat, batch):
    n = len(flat) // batch
    return [list(flat[i * n : (i + 1) * n]) for i in range(batch)]


def _target_shape(in_shape, out_len, is_embedding):
    in_vol = 1
    for d in in_shape:
        in_vol *= d
    if out_len == in_vol:
        return list(in_shape)
    if is_embedding:
        return [in_shape[0], out_len // in_shape[0]]
    if len(in_shape) == 1:
        return [out_len]
    return [in_shape[0], out_len // in_shape[0]]


def _train_cabi(net, inp, tgt, in_shape, tgt_shape, batch, epochs, mode, lr):
    batches_in = [_rows(inp, batch)] if batch > 1 else [[inp]]
    batches_tgt = [_rows(tgt, batch)] if batch > 1 else [[tgt]]
    return train(
        net, batches_in, batches_tgt,
        epochs=epochs, learning_rate=lr, mode=mode, use_gpu=False, verbose=False,
        input_shape=in_shape, target_shape=tgt_shape,
    )


def _bench_forward_cabi(net, inp, shape, mode, iters):
    """Forward timing through LoomForwardPolymorphic + LoomConfigureTrainingMode."""
    net.set_training_mode(mode)
    for _ in range(3):
        net.forward_polymorphic(inp, shape)
    t0 = time.perf_counter_ns()
    last = None
    for _ in range(iters):
        last = net.forward_polymorphic(inp, shape)
    avg_ns = (time.perf_counter_ns() - t0) // max(iters, 1)
    return last, avg_ns


def _bench_backward_cabi(net, inp, in_shape, tgt, tgt_shape, mode):
    net.set_training_mode(mode)
    r = net.backward_polymorphic(inp, in_shape, tgt, tgt_shape)
    return r.get("dx", []), r.get("dw", [])




def _check_save_reload_cabi(net, inp, shape, tgt, tgt_shape, tol, after):
    out0 = net.forward_polymorphic(inp, shape)
    wire = net.serialize()
    reloaded = Network.deserialize(wire)
    out1 = reloaded.forward_polymorphic(inp, shape)
    reloaded.free()
    diff = max_abs_diff(out0, out1)
    mult = 100 if after else 1
    return diff <= tol * mult


def run_suite(suite, layer_filter=None):
    passed = failed = 0
    for g in suite.grids:
        epochs = train_epochs_for_grid(g)
        iters = bench_iters_for_grid(g)
        label = f"{suite.name} {g[0]}×{g[1]}×{g[2]}"
        print(f"\n{'═' * 70}")
        print(f"  {label} — Python → Loom CABI (CPU SC/MC · train · save/reload)")
        print(f"{'═' * 70}")

        for name, json_name, dtype, tol in ALL_DTYPES:
            print(f"  · {name:<10} ", end="", flush=True)
            try:
                net = Network(suite.build(g, json_name))
                net.morph_all(dtype)

                inp, in_shape = suite.make_input(g)
                batch = 1 if suite.is_embedding else 4

                out_sc, _ = _bench_forward_cabi(net, inp, in_shape, TRAINING_MODE_CPU_SC, iters)
                out_mc, _ = _bench_forward_cabi(net, inp, in_shape, TRAINING_MODE_CPU_MC, iters)
                fwd_scmc = max_abs_diff(out_sc, out_mc)

                tgt = sin_target(out_sc)
                tgt_shape = _target_shape(in_shape, len(tgt), suite.is_embedding)

                dx_sc, dw_sc = _bench_backward_cabi(net, inp, in_shape, tgt, tgt_shape, TRAINING_MODE_CPU_SC)
                dx_mc, dw_mc = _bench_backward_cabi(net, inp, in_shape, tgt, tgt_shape, TRAINING_MODE_CPU_MC)
                bwd_scmc = max_abs_diff(dx_sc + dw_sc, dx_mc + dw_mc)

                det_tol = max(tol, 1e-10)
                det_ok = fwd_scmc <= det_tol and bwd_scmc <= det_tol * 10

                before_ok = _check_save_reload_cabi(net, inp, in_shape, tgt, tgt_shape, tol, False)

                lr = training_lr(dtype)
                net_sc = Network(suite.build(g, json_name))
                net_sc.morph_all(dtype)
                hist_sc = _train_cabi(net_sc, inp, tgt, in_shape, tgt_shape, batch, epochs, TRAINING_MODE_CPU_SC, lr)
                net_sc.free()

                net_mc = Network(suite.build(g, json_name))
                net_mc.morph_all(dtype)
                hist_mc = _train_cabi(net_mc, inp, tgt, in_shape, tgt_shape, batch, epochs, TRAINING_MODE_CPU_MC, lr)

                loss_init = hist_mc[0] if hist_mc else 0.0
                loss_final = hist_mc[-1] if hist_mc else 0.0
                requires_learn = suite.primary != "RESIDUAL" and not suite.no_learn
                learned = training_ok(loss_init, loss_final, dtype) or not requires_learn

                after_ok = _check_save_reload_cabi(net_mc, inp, in_shape, tgt, tgt_shape, tol, True)
                overall = before_ok and after_ok and learned and det_ok

                net.free()
                net_mc.free()

                if overall:
                    passed += 1
                    print(f"PASS  loss {loss_init:.4e}→{loss_final:.4e} det={det_ok} reload={after_ok}")
                else:
                    failed += 1
                    print(f"FAIL  loss {loss_init:.4e}→{loss_final:.4e} learn={learned} save={before_ok and after_ok} det={det_ok}")
            except Exception as e:
                failed += 1
                print(f"ERR   {e}")

    return failed == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", help="Single layer type, e.g. Dense")
    args = ap.parse_args()
    print("=== welvet seven-layer suite — Python ctypes → Loom CABI ===\n")
    ok = True
    for s in LAYER_SUITES:
        if args.layer and s.name != args.layer:
            continue
        if not run_suite(s):
            ok = False
    print("\n" + ("✅ ALL PASSED" if ok else "❌ FAILURES"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
