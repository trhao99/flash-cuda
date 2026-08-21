"""Profile the vLLM FA3 baseline (_vllm_fa3_C) on ONE config, for NCU comparison vs our kernel.

    FA3_CFG=attn_real_llama8b_pf_s16k FA3_ITERS=3 \
        ncu -k "regex:flash_fwd|prepare" -c 1 --section SpeedOfLight \
            python -m bench.microbenchmarks.prof_fa3
"""
import sys, os, torch
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from bench import scenarios as S

cfg_name = os.environ.get("FA3_CFG", "attn_real_llama8b_pf_s16k")
iters = int(os.environ.get("FA3_ITERS", "3"))
gpu = int(os.environ.get("FA3_GPU", "0"))
torch.cuda.set_device(gpu)

import vllm.vllm_flash_attn._vllm_fa3_C  # noqa: F401  registers torch.ops._vllm_fa3_C.fwd

cfg = next(c for c in S.get_configs("all") if c.name == cfg_name)
inp = S.make_inputs(cfg, 0, device=f"cuda:{gpu}")
for _ in range(10):
    S.call_fwd(S.BASELINE_NS, inp)
torch.cuda.synchronize()
for _ in range(iters):
    S.call_fwd(S.BASELINE_NS, inp)
torch.cuda.synchronize()
print(f"ran {cfg_name} x{iters} on {S.BASELINE_NS}")
