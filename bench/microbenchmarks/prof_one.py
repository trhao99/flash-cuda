"""Load a prebuilt candidate .so and run ONE config N times (for NCU/nsys profiling).

    FA3_CFG=attn_kv_hd64_fp8_decode ncu --set detailed python -m bench.microbenchmarks.prof_one
"""
import sys, os, torch
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from bench import scenarios as S

ns = os.environ.get("FA3_NS", "_fa3v1_C")
cfg_name = os.environ.get("FA3_CFG", "attn_kv_hd64_fp8_decode")
iters = int(os.environ.get("FA3_ITERS", "50"))
gpu = int(os.environ.get("FA3_GPU", "0"))
torch.cuda.set_device(gpu)

# load prebuilt .so
so = os.path.join(_REPO_ROOT, ".fa3_trials", ns, f"{ns}.so")
torch.ops.load_library(so)

cfg = next(c for c in S.get_configs("attntest") if c.name == cfg_name)
inp = S.make_inputs(cfg, 0, device=f"cuda:{gpu}")
# warmup
for _ in range(20):
    S.call_fwd(ns, inp)
torch.cuda.synchronize()
for _ in range(iters):
    S.call_fwd(ns, inp)
torch.cuda.synchronize()
print(f"ran {cfg_name} x{iters} on {ns}")
