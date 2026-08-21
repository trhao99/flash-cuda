"""Build fa4_blackwell.cu, warm ONE config, then run a single fwd as the region ncu profiles.

    ncu --set detailed python -m bench.microbenchmarks.ncu_probe attn_dense_hd128_causal
"""
import os, sys, torch
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from bench import eval as E
from bench import scenarios as S

ns = "_fa4bench_C"
work = os.path.expanduser("~/.cache/fa4bench")
E.build_candidate(os.path.join(_REPO_ROOT, "kernels", "fa4_blackwell.cu"), ns, work,
                  E.TARGETS["blackwell"])

# pick the config named on argv (default dense causal)
name = sys.argv[1] if len(sys.argv) > 1 else "attn_dense_hd128_causal"
cfg = next(c for c in S.get_configs() if c.name == name)
inp = S.make_inputs(cfg, seed=0)
torch.cuda.synchronize()
for _ in range(5):
    S.call_fwd(ns, inp)
torch.cuda.synchronize()
print("warmed", name, flush=True)
# the region ncu profiles:
S.call_fwd(ns, inp)
torch.cuda.synchronize()
print("done", flush=True)
