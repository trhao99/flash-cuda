"""Fast A/B: time ours vs the FA4 oracle for a few configs, optionally under env sweeps.

No CSV/plot writes, no cuDNN/FlashInfer, correctness optional. Designed for tight iteration.

  python -m bench.microbenchmarks.ab llama8b_dec_b16_s4k,llama8b_dec_b64_s4k
  python -m bench.microbenchmarks.ab llama8b_dec_b16_s4k --sweep FA4_FORCE_SPLITS=1,2,4,8
  python -m bench.microbenchmarks.ab llama8b_dec_b16_s4k --check      # + correctness
  FA4_AB_GPU=1 python -m bench.microbenchmarks.ab ...
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch

from bench import eval as E
from bench import scenarios as S


def _cfgs(names: list[str]) -> list[S.Config]:
    allc = {c.name: c for c in S.get_configs("all")}
    out = []
    for n in names:
        if n not in allc:
            sys.exit(f"unknown config {n!r}")
        out.append(allc[n])
    return out


def time_one(cfg: S.Config, ns: str, gpu: int, target, check: bool, reps: int | None):
    dev = f"cuda:{gpu}"
    inp = S.make_inputs(cfg, seed=0, device=dev)
    inp_attn = inp
    if cfg.is_bf16q_fp8kv and inp.k.dtype == torch.float8_e4m3fn:
        if cfg.is_scalar_fp8_scales:
            inp_attn = S.expand_fp8_descales(inp_attn)
        inp_attn = S._dequant_fp8_kv_to_bf16(inp_attn)
    flush = torch.empty(int(S.PERF["l2_flush_mb"]) * 1024 * 1024, dtype=torch.int8,
                        device=torch.device(dev))
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def base():
        return E.call_baseline(inp_attn, target, normalize_lse=False)

    def cand():
        return S.call_fwd(ns, inp_attn)

    err = None
    if check:
        ob, _ = E.call_baseline(inp_attn, target)
        oc = cand()[0]
        d = (oc.float() - ob.float()).abs().max().item()
        r = ob.float().abs().max().item()
        err = d / max(r, 1e-6)

    for _ in range(int(S.PERF["warmup_iters"])):
        base()
        cand()
    torch.cuda.synchronize()
    if reps:
        old = dict(S.PERF)
        S.PERF["timed_iters"], S.PERF["num_trials"] = reps, 1
    b = statistics.median(E._time(base, flush, st, en))
    c = statistics.median(E._time(cand, flush, st, en))
    if reps:
        S.PERF.update(old)
    return b, c, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs")
    ap.add_argument("--sweep", default=None, help="ENV=v1,v2,... (empty string = unset)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gpu", type=int, default=int(os.environ.get("FA4_AB_GPU", 0)))
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--build", action="store_true", default=True)
    ap.add_argument("--no-build", dest="build", action="store_false")
    a = ap.parse_args()

    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    target = E.TARGETS["blackwell"]
    ns = "_fa4bench_C"
    if a.build:
        E.build_candidate(target.cand_default, ns,
                          os.path.join(os.getcwd(), target.trials_dir, ns), target)
    torch.cuda.set_device(a.gpu)
    E.load_baseline()

    cfgs = _cfgs(a.configs.split(","))
    variants = [(None, None)]
    if a.sweep:
        key, vals = a.sweep.split("=", 1)
        variants = [(key, v) for v in vals.split(",")]

    for key, val in variants:
        if key is not None:
            if val == "":
                os.environ.pop(key, None)
                tag = f"[{key} unset] "
            else:
                os.environ[key] = val
                tag = f"[{key}={val}] "
        else:
            tag = ""
        for cfg in cfgs:
            b, c, err = time_one(cfg, ns, a.gpu, target, a.check, a.reps)
            e = f"  err={err:.2e}" if err is not None else ""
            print(f"{tag}{cfg.name:34s} FA4 {b*1e3:8.2f}us  ours {c*1e3:8.2f}us  "
                  f"{b/c:6.3f}x{e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
