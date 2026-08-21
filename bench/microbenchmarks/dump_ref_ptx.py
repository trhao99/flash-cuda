"""Dump the CuteDSL SM100 forward PTX/SASS/CUBIN for chosen configs (ground truth for the port).

Monkeypatches cutlass.cute.compile to (a) force --keep-ptx/--keep-cubin and (b) copy every
produced artifact (PTX + CUBIN) into an output dir, then drives the FA4 oracle over a few configs
so it JIT-compiles the forward kernel. Also disassembles each cubin to SASS via cuobjdump.

    cd /home/yak/fa3-cuda
    CUDA_VISIBLE_DEVICES=0 python -m bench.microbenchmarks.dump_ref_ptx --out reference/ptxdump \
        --configs attn_dense_hd128_causal,attn_dense_hd128_noncausal
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DEFAULT = os.path.join(_REPO_ROOT, "reference", "ptxdump")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--configs", default="attn_dense_hd128_causal")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    # CuteDSL writes kept PTX/cubin into this dir (the ptxas hook globs it too).
    os.environ["CUTE_DSL_DUMP_DIR"] = out
    os.environ["CUTE_DSL_KEEP_PTX"] = "1"
    os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "0")  # force fresh compile

    import torch  # noqa
    import cutlass.cute as cute

    _og = cute.compile
    seen = {"n": 0}

    def patched(*a, **k):
        opt = k.get("options", "") or ""
        for flag in ("--keep-ptx", "--keep-cubin", "--generate-line-info"):
            if flag not in opt:
                opt = (opt + " " + flag).strip()
        k["options"] = opt
        t0 = time.time()
        res = _og(*a, **k)
        seen["n"] += 1
        print(f"[compile #{seen['n']}] {time.time()-t0:.1f}s options={opt!r}", flush=True)
        # Try to pull artifacts straight off the compiled object.
        try:
            arts = getattr(res, "artifacts", None)
            fn = getattr(res, "function_name", f"kernel{seen['n']}")
            if arts is not None:
                ptx = getattr(arts, "PTX", None)
                cub = getattr(arts, "CUBIN", None)
                if ptx:
                    p = os.path.join(out, f"{fn}.artifact.ptx")
                    open(p, "w").write(ptx if isinstance(ptx, str) else ptx.decode("utf-8", "replace"))
                    print(f"    wrote {p} ({len(ptx)} bytes)", flush=True)
                if cub:
                    p = os.path.join(out, f"{fn}.artifact.cubin")
                    open(p, "wb").write(cub if isinstance(cub, (bytes, bytearray)) else bytes(cub))
                    print(f"    wrote {p} ({len(cub)} bytes)", flush=True)
        except Exception as e:
            print(f"    (artifact introspection failed: {type(e).__name__}: {e})", flush=True)
        return res

    cute.compile = patched

    from .. import scenarios as S
    from .. import eval as _eval

    want = {s.strip() for s in args.configs.split(",") if s.strip()}
    cfgs = [c for c in S.get_configs("attntest") if c.name in want]
    print(f"[dump] {len(cfgs)} configs -> {out}", flush=True)
    for cfg in cfgs:
        inp = S.make_inputs(cfg, seed=0, device="cuda")
        try:
            o, l = _eval.call_oracle(inp)
            torch.cuda.synchronize()
            print(f"[ok] {cfg.name}: out {tuple(o.shape)}", flush=True)
        except Exception as e:
            print(f"[skip] {cfg.name}: {type(e).__name__}: {e}", flush=True)

    # Disassemble any cubins to SASS.
    cuobjdump = shutil.which("cuobjdump")
    for cub in glob.glob(os.path.join(out, "*.cubin")):
        if cuobjdump:
            sass = cub.rsplit(".", 1)[0] + ".sass"
            try:
                r = subprocess.run([cuobjdump, "-sass", cub], capture_output=True, text=True)
                open(sass, "w").write(r.stdout)
                print(f"[sass] {sass} ({len(r.stdout)} bytes)", flush=True)
            except Exception as e:
                print(f"[sass] failed for {cub}: {e}", flush=True)

    ptxs = glob.glob(os.path.join(out, "*.ptx"))
    print(f"[done] {len(ptxs)} ptx files in {out}", flush=True)
    for p in sorted(ptxs):
        print("   ", p, os.path.getsize(p), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
