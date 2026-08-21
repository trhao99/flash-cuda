"""Compare two snapshots of the results_blackwell_*.csv set (regression check).

  python -m bench.microbenchmarks.cmp /tmp/fa4base docs
"""
from __future__ import annotations

import csv
import glob
import math
import os
import sys


def load(d: str) -> dict[str, dict]:
    out = {}
    for f in glob.glob(os.path.join(d, "results_blackwell_*.csv")):
        for r in csv.DictReader(open(f)):
            out[r["scenario"]] = r
    return out


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main() -> int:
    a, b = sys.argv[1], sys.argv[2]
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    A, B = load(a), load(b)
    rows = []
    for k in sorted(set(A) | set(B)):
        ra, rb = A.get(k), B.get(k)
        fa = float(ra["ours_x"]) if ra and ra.get("ours_x") else None
        fb = float(rb["ours_x"]) if rb and rb.get("ours_x") else None
        sa = ra["status"] if ra else "-"
        sb = rb["status"] if rb else "-"
        rows.append((k, fa, fb, sa, sb))
    regr = [r for r in rows if r[1] and r[2] and r[2] < r[1] * (1 - thr)]
    impr = [r for r in rows if r[1] and r[2] and r[2] > r[1] * (1 + thr)]
    stat = [r for r in rows if r[3] != r[4]]
    print(f"=== REGRESSIONS (>{thr:.0%}) : {len(regr)}")
    for k, fa, fb, _, _ in sorted(regr, key=lambda r: r[2] / r[1]):
        print(f"  {fa:.3f} -> {fb:.3f}  ({fb/fa:+.1%})  {k}")
    print(f"=== IMPROVEMENTS (>{thr:.0%}) : {len(impr)}")
    for k, fa, fb, _, _ in sorted(impr, key=lambda r: -r[2] / r[1]):
        print(f"  {fa:.3f} -> {fb:.3f}  ({fb/fa:+.1%})  {k}")
    if stat:
        print("=== STATUS CHANGES")
        for k, _, _, sa, sb in stat:
            print(f"  {sa} -> {sb}  {k}")
    va = [r[1] for r in rows if r[1] and r[2]]
    vb = [r[2] for r in rows if r[1] and r[2]]
    print(f"=== paired n={len(va)}  geomean {gm(va):.4f} -> {gm(vb):.4f}   "
          f"min {min(va):.3f} -> {min(vb):.3f}   >=1.0x {sum(1 for x in va if x>=1)} -> "
          f"{sum(1 for x in vb if x>=1)}")
    allb = [float(r["ours_x"]) for r in B.values() if r.get("ours_x") and r["status"] == "OK"]
    print(f"=== NEW set: n={len(allb)} geomean {gm(allb):.4f} <1.0x: {sum(1 for x in allb if x<1.0)}")
    under = sorted([(float(r['ours_x']), k) for k, r in B.items()
                    if r.get("ours_x") and r["status"] == "OK" and float(r["ours_x"]) < 1.0])
    for v, k in under:
        print(f"    {v:.3f}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
