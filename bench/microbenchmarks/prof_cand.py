"""Minimal driver: load the already-built candidate .so and run ONE config's fwd N times.
Used under ncu/nsys to profile the candidate kernel in isolation (no oracle kernels).

    ncu --set detailed python -m bench.microbenchmarks.prof_cand attn_dense_hd128_noncausal 3
"""
import os, sys, torch
from bench import scenarios as S

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "attn_dense_hd128_noncausal"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ns = "_fa4bench_C"
    work_dir = os.path.join(_REPO_ROOT, ".fa4_trials", ns)
    so = os.path.join(work_dir, f"{ns}.so")
    torch.ops.load_library(so)
    cfg = next(c for c in S.FA3TEST_CONFIGS if c.name == name)
    inp = S.make_inputs(cfg, seed=0, device="cuda")
    for _ in range(2):
        S.call_fwd(ns, inp); torch.cuda.synchronize()
    for _ in range(iters):
        S.call_fwd(ns, inp)
    torch.cuda.synchronize()
    print("done", name)

if __name__ == "__main__":
    main()
