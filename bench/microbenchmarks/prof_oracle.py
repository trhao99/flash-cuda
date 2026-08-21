"""Run ONE config's FA4 reference (oracle) N times, for ncu/nsys profiling of the reference kernel.

    ncu --set detailed python -m bench.microbenchmarks.prof_oracle attn_dense_hd128_noncausal 3
"""
import sys, torch
from bench import scenarios as S
from bench import eval as _eval

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "attn_dense_hd128_noncausal"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    cfg = next(c for c in S.FA3TEST_CONFIGS if c.name == name)
    inp = S.make_inputs(cfg, seed=0, device="cuda")
    for _ in range(2):
        _eval.call_oracle(inp); torch.cuda.synchronize()
    for _ in range(iters):
        _eval.call_oracle(inp)
    torch.cuda.synchronize()
    print("done", name)

if __name__ == "__main__":
    main()
