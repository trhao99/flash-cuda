"""Minimal single-shape FA3 forward driver for NCU profiling. Mirrors ab_fa3's config.
Usage: python -m fa3agent.cutlass_oracle.prof_fa3 [B H S warmup iters]
"""
import sys
import torch
from .. import config as C
from .. import shapes
from .ab_fa3 import _cfg


def main(argv):
    B = int(argv[0]) if len(argv) > 0 else 1
    H = int(argv[1]) if len(argv) > 1 else 16
    S = int(argv[2]) if len(argv) > 2 else 4096
    warmup = int(argv[3]) if len(argv) > 3 else 40
    iters = int(argv[4]) if len(argv) > 4 else 3
    torch.ops.load_library(C.BASELINE_SO)
    cfg = _cfg(B, H, S)
    inp = shapes.make_inputs(cfg, seed=0, device="cuda:0")
    for _ in range(warmup):
        shapes.call_fwd("_vllm_fa3_C", inp)
    torch.cuda.synchronize()
    for _ in range(iters):
        shapes.call_fwd("_vllm_fa3_C", inp)
    torch.cuda.synchronize()
    print("done", B, H, S)


if __name__ == "__main__":
    main(sys.argv[1:])
