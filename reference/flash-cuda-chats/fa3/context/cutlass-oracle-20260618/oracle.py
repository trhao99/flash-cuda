"""cutlass_oracle driver — build, benchmark, and compare the CUTLASS reference GEMM against
the pure-CUDA/PTX re-implementation on identical problems.

Workflow (the whole point of the component):
  1. build the CUTLASS oracle GEMM (`cutlass_gemm.cu`) and the pure-CUDA/PTX GEMM
     (`ptx_gemm.cu`) for a given mode (QK / PV);
  2. run both on a shape, parse their JSON result lines;
  3. cross-check the dumped fp32 outputs are aligned (max-abs diff);
  4. report the perf ratio (ptx_tflops / cutlass_tflops) so we can iterate the pure kernel
     until it matches CUTLASS while staying output-aligned.

CLI:
    python -m fa3agent.cutlass_oracle.oracle build [--mode QK|PV]
    python -m fa3agent.cutlass_oracle.oracle bench [--mode QK] [--shapes MxNxK,...] [--gpu 0]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.join(_THIS, "kernels")
RESULTS = os.path.join(_THIS, "results")
_REPO = os.path.dirname(os.path.dirname(_THIS))
CUTLASS_DIR = os.environ.get("FA3_CUTLASS_DIR") or os.path.join(_REPO, "reference", "libraries", "cutlass")

NVCC = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
GENCODE = ["-gencode", "arch=compute_90a,code=sm_90a"]
NVCC_COMMON = ["-O3", "-std=c++17", *GENCODE, "--expt-relaxed-constexpr", "--expt-extended-lambda"]

# Default shapes. QK: K=head_dim=128, M=N=seqlen tile-scale. A range probes tile-quant + scale.
DEFAULT_SHAPES = ["2048x2048x128", "4096x4096x128", "8192x8192x128", "16384x16384x128"]


def _run(cmd, timeout=600, env=None):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       timeout=timeout, env=env)
    return p.returncode, p.stdout


def build(mode: str = "QK") -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    inc = ["-I", os.path.join(CUTLASS_DIR, "include"),
           "-I", os.path.join(CUTLASS_DIR, "tools", "util", "include")]
    mode_def = "-DGEMM_MODE_PV" if mode == "PV" else "-DGEMM_MODE_QK"
    cutlass_bin = os.path.join(KERNELS, f"cutlass_gemm_{mode.lower()}")
    ptx_bin = os.path.join(KERNELS, f"ptx_gemm_{mode.lower()}")
    out = {}
    t0 = time.time()
    rc, log = _run([NVCC, *NVCC_COMMON, mode_def, *inc, "-o", cutlass_bin,
                   os.path.join(KERNELS, "cutlass_gemm.cu")])
    out["cutlass"] = {"ok": rc == 0, "bin": cutlass_bin, "secs": round(time.time() - t0, 1)}
    if rc != 0:
        out["cutlass"]["log"] = log[-1500:]
    t0 = time.time()
    rc, log = _run([NVCC, *NVCC_COMMON, mode_def, "-o", ptx_bin,
                   os.path.join(KERNELS, "ptx_gemm.cu")])
    out["ptx"] = {"ok": rc == 0, "bin": ptx_bin, "secs": round(time.time() - t0, 1)}
    if rc != 0:
        out["ptx"]["log"] = log[-1500:]
    return out


def _parse_json_line(text: str) -> dict:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:  # noqa: BLE001
                continue
    return {}


def _max_abs_diff(p1: str, p2: str) -> float:
    with open(p1, "rb") as f:
        a = f.read()
    with open(p2, "rb") as f:
        b = f.read()
    n = min(len(a), len(b)) // 4
    fa = struct.unpack(f"{n}f", a[: n * 4])
    fb = struct.unpack(f"{n}f", b[: n * 4])
    m = 0.0
    for x, y in zip(fa, fb):
        d = abs(x - y)
        if d > m:
            m = d
    return m


def bench(mode: str = "QK", shapes=None, gpu: int = 0) -> dict:
    shapes = shapes or DEFAULT_SHAPES
    os.makedirs(RESULTS, exist_ok=True)
    cutlass_bin = os.path.join(KERNELS, f"cutlass_gemm_{mode.lower()}")
    ptx_bin = os.path.join(KERNELS, f"ptx_gemm_{mode.lower()}")
    if not (os.path.exists(cutlass_bin) and os.path.exists(ptx_bin)):
        b = build(mode)
        if not (b["cutlass"]["ok"] and b["ptx"]["ok"]):
            return {"error": "build failed", "build": b}
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    rows = []
    for shp in shapes:
        M, N, K = (int(x) for x in shp.lower().split("x"))
        cout = os.path.join(RESULTS, f"c_cutlass_{mode}_{shp}.bin")
        pout = os.path.join(RESULTS, f"c_ptx_{mode}_{shp}.bin")
        _, ct = _run([cutlass_bin, str(M), str(N), str(K), cout, "200", "9"], env=env)
        _, pt = _run([ptx_bin, str(M), str(N), str(K), pout, "200", "9"], env=env)
        cj, pj = _parse_json_line(ct), _parse_json_line(pt)
        diff = _max_abs_diff(cout, pout) if (os.path.exists(cout) and os.path.exists(pout)) else None
        ratio = (pj.get("tflops", 0) / cj["tflops"]) if cj.get("tflops") else None
        rows.append({
            "shape": shp,
            "cutlass_ms": cj.get("median_ms"), "cutlass_tflops": cj.get("tflops"),
            "ptx_ms": pj.get("median_ms"), "ptx_tflops": pj.get("tflops"),
            "perf_ratio_ptx_over_cutlass": round(ratio, 4) if ratio else None,
            "output_max_abs_diff": diff,
            "ptx_correct_vs_ref": pj.get("nbad") == 0,
        })
    report = {"mode": mode, "gpu": gpu, "rows": rows, "ts": time.time()}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"bench_{mode}.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def _fmt(report: dict) -> str:
    if "error" in report:
        return f"ERROR: {report['error']}\n{json.dumps(report.get('build', {}), indent=2)}"
    lines = [f"# cutlass_oracle bench  mode={report['mode']}  gpu={report['gpu']}",
             f"{'shape':18s} {'cutlass TFLOPS':>15s} {'ptx TFLOPS':>12s} {'ratio':>7s} "
             f"{'out|Δ|':>9s} {'ok':>4s}"]
    for r in report["rows"]:
        lines.append(f"{r['shape']:18s} {r['cutlass_tflops']:>15.2f} {r['ptx_tflops']:>12.2f} "
                     f"{(r['perf_ratio_ptx_over_cutlass'] or 0):>7.3f} "
                     f"{(r['output_max_abs_diff'] if r['output_max_abs_diff'] is not None else -1):>9.2g} "
                     f"{str(r['ptx_correct_vs_ref']):>4s}")
    return "\n".join(lines)


def _main(argv):
    ap = argparse.ArgumentParser(prog="cutlass_oracle.oracle")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("build")
    pb.add_argument("--mode", default="QK", choices=["QK", "PV"])
    pn = sub.add_parser("bench")
    pn.add_argument("--mode", default="QK", choices=["QK", "PV"])
    pn.add_argument("--shapes", default=None, help="comma list MxNxK")
    pn.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        print(json.dumps(build(args.mode), indent=2))
    elif args.cmd == "bench":
        shapes = args.shapes.split(",") if args.shapes else None
        print(_fmt(bench(args.mode, shapes, args.gpu)))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
