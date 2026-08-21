"""Load the prebuilt candidate .so and run ONE config (from any set) N times, for NCU/nsys profiling.

    FA3_NS=_distill_fa3_C FA3_CFG=attn_real_llama8b_dec_s4k FA3_ITERS=6 \
        ncu -k prefill --set detailed --launch-count 2 python -m bench.microbenchmarks.prof_real
"""
import sys, os, torch
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from bench import scenarios as S

ns = os.environ.get("FA3_NS", "_distill_fa3_C")
cfg_name = os.environ.get("FA3_CFG", "attn_real_llama8b_dec_s4k")
iters = int(os.environ.get("FA3_ITERS", "6"))
gpu = int(os.environ.get("FA3_GPU", "0"))
torch.cuda.set_device(gpu)

so = os.path.join(_REPO_ROOT, ".fa3_trials", ns, f"{ns}.so")
torch.ops.load_library(so)

cfg = next(c for c in S.get_configs("all") if c.name == cfg_name)
inp = S.make_inputs(cfg, 0, device=f"cuda:{gpu}")
for _ in range(10):
    S.call_fwd(ns, inp)
torch.cuda.synchronize()
for _ in range(iters):
    S.call_fwd(ns, inp)
torch.cuda.synchronize()
print(f"ran {cfg_name} x{iters} on {ns}")
