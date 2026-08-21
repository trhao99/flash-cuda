"""NCU/nsys driver: run ONE config through ours or the FA4 oracle, N times.

  FA4_SO=.fa4_trials/ab/_fa4bench_C.so FA4_CFG=llama8b_dec_b64_s4k \
    ncu --set detailed --launch-count 4 python -m bench.microbenchmarks.prof1
  FA4_IMPL=fa4 FA4_CFG=... ncu ... python -m bench.microbenchmarks.prof1
"""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from bench import scenarios as S  # noqa: E402

cfg_name = os.environ.get("FA4_CFG", "llama8b_dec_b64_s4k")
iters = int(os.environ.get("FA4_ITERS", "3"))
gpu = int(os.environ.get("FA4_GPU", "0"))
impl = os.environ.get("FA4_IMPL", "ours")
torch.cuda.set_device(gpu)

cfg = next(c for c in S.get_configs("all") if c.name == cfg_name)
inp = S.make_inputs(cfg, 0, device=f"cuda:{gpu}")
if cfg.is_bf16q_fp8kv and inp.k.dtype == torch.float8_e4m3fn:
    if cfg.is_scalar_fp8_scales:
        inp = S.expand_fp8_descales(inp)
    inp = S._dequant_fp8_kv_to_bf16(inp)

if impl == "ours":
    ns = os.environ.get("FA4_NS", "_fa4bench_C")
    so = os.environ.get("FA4_SO") or os.path.join(_ROOT, ".fa4_trials", ns, f"{ns}.so")
    torch.ops.load_library(so)
    call = lambda: S.call_fwd(ns, inp)  # noqa: E731
else:
    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    from bench import eval as E
    E.load_baseline()
    call = lambda: E.call_baseline(inp, E.TARGETS["blackwell"], normalize_lse=False)  # noqa: E731

for _ in range(6):
    call()
torch.cuda.synchronize()
torch.cuda.profiler.start()
for _ in range(iters):
    call()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print(f"ran {cfg_name} x{iters} impl={impl}")
