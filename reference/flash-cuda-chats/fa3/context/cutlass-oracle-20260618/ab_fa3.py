"""A/B the pure-CUDA/PTX fused attention kernel (kernels/fused_attn) against the real FA3 baseline
(`_vllm_fa3_C.fwd`) on matching shapes: non-paged, causal, bf16, head_dim=128, MHA (q_heads==kv_heads),
q_len==k_len. Same fp32-reference correctness rule the agent uses; latency via CUDA events with warmup
+ L2 flush in a warmed process (both sides).

Usage: python -m fa3agent.cutlass_oracle.ab_fa3 [--gpu 1] [--shapes B,H,S;...]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

import torch

from .. import config as C
from .. import shapes

_THIS = os.path.dirname(os.path.abspath(__file__))
FUSED_BIN = os.path.join(_THIS, "kernels", "fused_attn")  # overridable via --bin

DEFAULT = [(8, 16, 2048), (4, 16, 4096), (2, 16, 8192), (2, 32, 4096), (1, 32, 8192)]


def _cfg(B, H, S):
    return C.Config(f"ab_hd128_bf16_B{B}_H{H}_S{S}", "prefill", "bf16", 128, H, H,
                    causal=True, paged=False, batch=B, q_len=S, k_len=S)


def time_fa3(cfg, gpu, warmup=200, iters=100, trials=5):
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)
    inp = shapes.make_inputs(cfg, seed=0, device=f"cuda:{gpu}")
    flush = torch.empty(int(C.PERF["l2_flush_mb"]) * 1024 * 1024, dtype=torch.int8, device=dev)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        shapes.call_fwd("_vllm_fa3_C", inp)
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        flush.zero_(); torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            shapes.call_fwd("_vllm_fa3_C", inp)
        e.record(); torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) / iters)
    return statistics.median(samples)


def run_mine(B, H, S, gpu, iters=100, trials=7, binary=FUSED_BIN):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    p = subprocess.run([binary, str(S), "128", "1", str(B), str(H), str(iters), str(trials)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("no JSON from fused_attn: " + p.stdout[-400:])


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--shapes", default=None, help="B,H,S;B,H,S;...")
    ap.add_argument("--bin", default=FUSED_BIN, help="path to the fused_attn binary to A/B")
    args = ap.parse_args(argv)
    torch.ops.load_library(C.BASELINE_SO)

    shps = DEFAULT
    if args.shapes:
        shps = [tuple(int(x) for x in t.split(",")) for t in args.shapes.split(";")]

    print(f"{'B,H,S':>14} {'FA3 ms':>9} {'mine ms':>9} {'FA3/mine':>9} "
          f"{'FA3 TFLOP/s':>11} {'mine TFLOP/s':>12} {'O|Δ|':>9} {'ok':>4}")
    rows = []
    for (B, H, S) in shps:
        cfg = _cfg(B, H, S)
        fa3_ms = time_fa3(cfg, args.gpu)
        mj = run_mine(B, H, S, args.gpu, binary=args.bin)
        mine_ms = mj["median_ms"]
        # causal FLOPs (QK+PV, half for causal)
        flop = B * H * 2.0 * (2.0 * S * S * 128) * 0.5
        fa3_tf = flop / (fa3_ms * 1e-3) / 1e12
        mine_tf = flop / (mine_ms * 1e-3) / 1e12
        ratio = fa3_ms / mine_ms
        ok = mj["O_max_abs"] < 5e-3
        rows.append({"B": B, "H": H, "S": S, "fa3_ms": fa3_ms, "mine_ms": mine_ms,
                     "ratio_fa3_over_mine": ratio, "fa3_tflops": fa3_tf, "mine_tflops": mine_tf,
                     "O_max_abs": mj["O_max_abs"], "correct": ok})
        print(f"{B},{H},{S:>6} {fa3_ms:>9.4f} {mine_ms:>9.4f} {ratio:>9.3f} "
              f"{fa3_tf:>11.1f} {mine_tf:>12.1f} {mj['O_max_abs']:>9.1e} {str(ok):>4}")
    with open(os.path.join(_THIS, "results", "ab_fa3.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
