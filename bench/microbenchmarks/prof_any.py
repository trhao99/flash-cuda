"""Generic NCU/nsys driver: run ONE config (from ANY set) on ours or the FA4 oracle.

Ours (loads the prebuilt .fa4_trials/<ns>/<ns>.so, does NOT rebuild):
    ncu --set detailed -o /tmp/o python -m bench.microbenchmarks.prof_any ours <cfg> [iters]
FA4 oracle:
    ncu --set detailed -o /tmp/f python -m bench.microbenchmarks.prof_any fa4  <cfg> [iters]

Env: FA4_PROF_NS (default _fa4bench_C), FA4_PROF_GPU (default 0).
"""
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from bench import eval as E          # noqa: E402
from bench import scenarios as S     # noqa: E402


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "ours"
    name = sys.argv[2] if len(sys.argv) > 2 else "attn_real_llama8b_dec_s16k"
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    gpu = int(os.environ.get("FA4_PROF_GPU", "0"))
    torch.cuda.set_device(gpu)
    dev = f"cuda:{gpu}"

    cfg = next(c for c in S.get_configs("all") if c.name == name)
    inp = S.make_inputs(cfg, seed=0, device=dev)

    if which == "ours":
        ns = os.environ.get("FA4_PROF_NS", "_fa4bench_C")
        torch.ops.load_library(os.path.join(_REPO_ROOT, ".fa4_trials", ns, f"{ns}.so"))
        run = lambda: S.call_fwd(ns, inp)                                   # noqa: E731
    else:
        tgt = E.TARGETS["blackwell"]
        run = lambda: E.call_baseline(inp, tgt, normalize_lse=False)        # noqa: E731

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    print("warmed", which, name, flush=True)
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    print("done", which, name, flush=True)


if __name__ == "__main__":
    main()
