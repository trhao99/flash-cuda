"""Render category-aggregated bar charts — one PNG per config set evaluated.

Each bar is the geomean speedup over scenarios in that category (paper taxonomy for `claim`:
Causal d=64, Decode ≤4k, FP8 KV, …). Per-scenario numbers stay in the JSON from eval.

    python -m bench.plot --arch blackwell --set all --placeholder
        -> docs/perf_fa4_blackwell_{claim,attntest,edge,real}.png
    python -m bench.plot --arch hopper --set claim --results docs/last_results_hopper.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import bench.scenarios as S

_DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")

# Distinct, print-friendly: warm rust (ours) / stone (ref) / deep teal (cuDNN) / slate (FlashInfer).
_OURS, _REF, _CUDNN, _FI = "#c45c26", "#9a9590", "#1a6b6b", "#3d4f7c"

_ARCH = {
    "hopper": dict(tag="fa3_hopper", ref="FA3", file_prefix="perf_fa3_hopper"),
    "blackwell": dict(tag="fa4_blackwell", ref="FA4", file_prefix="perf_fa4_blackwell"),
}


def _geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def _placeholder_ratio(name: str, salt: str = "") -> float:
    """Stable pseudo-random speedup in ~[0.75, 1.35] for layout inspection."""
    h = hashlib.md5(f"{salt}:{name}".encode()).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF
    return 0.75 + 0.60 * u


def _placeholder_cudnn(name: str) -> Optional[float]:
    """Some scenarios have no cuDNN bar (homogeneous fp8 / softcap / leftpad / sink)."""
    low = name.lower()
    if any(t in low for t in ("softcap", "leftpad", "sink")):
        return None
    # Homogeneous FP8: sdpa_fp8 heuristics still segfault on this stack.
    if "fp8" in low and "bf16q" not in low:
        return None
    h = hashlib.md5(f"cudnn:{name}".encode()).hexdigest()
    if int(h[:2], 16) < 40:
        return None
    return round(0.55 + 0.70 * (int(h[2:6], 16) / 0xFFFF), 3)


def _placeholder_fi(name: str) -> Optional[float]:
    """Some scenarios have no FlashInfer bar (per-head fp8 / leftpad / hd96 / softcap)."""
    low = name.lower()
    if any(t in low for t in ("leftpad", "d96", "hd96", "softcap")):
        return None
    # Per-head FP8 unsupported; scalar (vLLM-style) FP8 is supported on trtllm-gen.
    if ("fp8" in low or "bf16q_fp8kv" in low) and "scalar" not in low:
        return None
    h = hashlib.md5(f"fi:{name}".encode()).hexdigest()
    if int(h[:2], 16) < 35:
        return None
    return round(0.40 + 0.90 * (int(h[2:6], 16) / 0xFFFF), 3)


# Per-scenario: (ours, cudnn_or_None, flashinfer_or_None)
ScenRatios = tuple[float, Optional[float], Optional[float]]


def scenario_results(set_name: str, measured: Optional[dict] = None, *,
                     placeholder: bool = False, arch: str = "blackwell"
                     ) -> dict[str, ScenRatios]:
    """name -> (ours, cudnn_or_None, fi_or_None) for scenarios with a usable ours ratio."""
    out: dict[str, ScenRatios] = {}
    for c in S.get_configs(set_name):
        if measured and c.name in measured:
            ours = measured[c.name].get("ours")
            if ours is None:
                continue
            out[c.name] = (
                float(ours),
                measured[c.name].get("cudnn"),
                measured[c.name].get("flashinfer"),
            )
        elif placeholder:
            ours = round(_placeholder_ratio(c.name, arch), 3)
            out[c.name] = (ours, _placeholder_cudnn(c.name), _placeholder_fi(c.name))
    return out


def aggregate_bars(set_name: str, per_scen: dict[str, ScenRatios]
                   ) -> list[tuple[str, float, Optional[float], Optional[float], int]]:
    """Geomean within each plot_bar.

    Returns (bar_label, ours_gm, cudnn_gm_or_None, fi_gm_or_None, n_members).
    Bars keep taxonomy order (`bar_order`); empty bars (no measured members) are dropped.
    """
    buckets: dict[str, list[ScenRatios]] = defaultdict(list)
    for c in S.get_configs(set_name):
        if c.name not in per_scen:
            continue
        bar = S.plot_bar_for(c, set_name)
        if not bar:
            continue
        buckets[bar].append(per_scen[c.name])

    rows = []
    order = S.bar_order(set_name) or tuple(buckets.keys())
    for bar in order:
        members = buckets.get(bar) or []
        if not members:
            continue
        ours_gm = _geomean([o for o, _, _ in members])
        cud_vals = [c for _, c, _ in members if c is not None]
        fi_vals = [f for _, _, f in members if f is not None]
        cud_gm = _geomean(cud_vals) if cud_vals else None
        fi_gm = _geomean(fi_vals) if fi_vals else None
        rows.append((
            bar,
            round(ours_gm, 3),
            round(cud_gm, 3) if cud_gm else None,
            round(fi_gm, 3) if fi_gm else None,
            len(members),
        ))
    return rows


def _render(results, *, out, ref_label, title, xlabel, xlim=None, figsize=None,
            n_labels: Optional[list[str]] = None):
    """Grouped horizontal bars: ours / reference / cuDNN / FlashInfer per category bar.

    `results` entries are (label, ours, cudnn_or_None, fi_or_None). Taxonomy order is preserved
    (not re-sorted by speedup) so Fig1/Fig2 paper bars read top-to-bottom as defined.
    """
    if not results:
        print(f"skip empty plot -> {out}")
        return None
    # Paper order: reverse so first taxonomy bar is at the TOP of a barh chart.
    rows = list(reversed(results))
    names = [r[0] for r in rows]
    ours = [r[1] for r in rows]
    cudnn = [r[2] for r in rows]
    fi = [r[3] if len(r) > 3 else None for r in rows]
    n = len(rows)

    if figsize is None:
        figsize = (10.5, max(3.8, 0.65 * n + 1.8))
    if xlim is None:
        extras = [c for c in cudnn if c is not None] + [f for f in fi if f is not None]
        hi = max([1.0] + ours + extras)
        xlim = (0.0, max(1.55, hi * 1.15))

    fig, ax = plt.subplots(figsize=figsize)
    y = list(range(n))
    # Four series: ours / ref / cuDNN / FlashInfer stacked vertically within each category.
    hgt = 0.20
    off_ours, off_ref, off_cud, off_fi = 1.5 * hgt, 0.5 * hgt, -0.5 * hgt, -1.5 * hgt
    ax.barh([i + off_ours for i in y], ours, color=_OURS, height=hgt, zorder=3, label="ours")
    ax.barh([i + off_ref for i in y], [1.0] * n, color=_REF, height=hgt, zorder=3,
            label=f"{ref_label} (reference)")
    have_c = [(i + off_cud, c) for i, c in zip(y, cudnn) if c is not None]
    if have_c:
        ax.barh([i for i, _ in have_c], [c for _, c in have_c], color=_CUDNN, height=hgt,
                zorder=3, label="cuDNN")
    have_f = [(i + off_fi, f) for i, f in zip(y, fi) if f is not None]
    if have_f:
        ax.barh([i for i, _ in have_f], [f for _, f in have_f], color=_FI, height=hgt,
                zorder=3, label="FlashInfer")

    pad = (xlim[1] - xlim[0]) * 0.008
    for i, v in zip(y, ours):
        ax.text(v + pad, i + off_ours, f"{v:.2f}×", va="center", ha="left", fontsize=7.5,
                color="#333")
    for i, c in have_c:
        ax.text(c + pad, i, f"{c:.2f}×", va="center", ha="left", fontsize=7.5, color="#333")
    for i, f in have_f:
        ax.text(f + pad, i, f"{f:.2f}×", va="center", ha="left", fontsize=7.5, color="#333")

    ax.axvline(1.0, color="#444", linestyle="--", linewidth=1.1, zorder=2)
    ax.text(1.0, n - 0.35, " parity", color="#555", fontsize=8, va="top", ha="left")
    ax.set_yticks(y)
    if n_labels:
        labels = list(reversed(n_labels))
        ax.set_yticklabels([f"{nm}  (n={lab})" for nm, lab in zip(names, labels)], fontsize=9)
    else:
        ax.set_yticklabels(names, fontsize=9)
    ax.set_ylim(-0.95, n - 0.05)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=12)
    ax.grid(axis="x", color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def plot_set(arch: str, set_name: str, bars: list[tuple], *, placeholder: bool = False,
             n_scen: int = 0) -> Optional[str]:
    meta = _ARCH[arch]
    ours = [r[1] for r in bars]
    cud = [r[2] for r in bars if r[2] is not None]
    fi = [r[3] for r in bars if len(r) > 3 and r[3] is not None]
    gm = _geomean(ours)
    n_ge1 = sum(1 for v in ours if v >= 1.0)
    ph = " [PLACEHOLDER]" if placeholder else ""
    n_members = sum(r[-1] for r in bars)
    cud_txt = (f" · cuDNN geomean {_geomean(cud):.3f}× on {len(cud)} bars" if cud
               else " · cuDNN not measured")
    fi_txt = (f" · FlashInfer geomean {_geomean(fi):.3f}× on {len(fi)} bars" if fi
              else " · FlashInfer not measured")
    out = os.path.join(_DOCS, f"{meta['file_prefix']}_{set_name}.png")
    # Strip member counts for the render tuple; pass as y-tick annotations.
    render_rows = [(r[0], r[1], r[2], r[3] if len(r) > 3 else None) for r in bars]
    n_labels = [str(r[-1]) for r in bars]
    return _render(
        render_rows, out=out, ref_label=meta["ref"], n_labels=n_labels,
        title=(f"{meta['tag']}.cu vs {meta['ref']} vs cuDNN vs FlashInfer — set `{set_name}` "
               f"({len(bars)} bars · {n_members or n_scen} scenarios, geomean within bar){ph}\n"
               f"ours geomean {gm:.3f}× · {n_ge1}/{len(bars)} bars ≥1.0×{cud_txt}{fi_txt}"),
        xlabel=(f"Speedup vs {meta['ref']}  ({meta['ref']} time / impl time;  >1 = faster)"),
    )


def load_measured(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "rows" in data:
        return {
            r["name"]: {
                "ours": r.get("ours"),
                "cudnn": r.get("cudnn"),
                "flashinfer": r.get("flashinfer"),
            }
            for r in data["rows"]
        }
    return data


def write_measured(path: str, rows: list, *, arch: str, cfgset: str) -> str:
    """Serialize eval results for `bench.plot --results` (per-scenario; plot aggregates)."""
    payload = {
        "arch": arch,
        "cfgset": cfgset,
        "rows": [
            {"name": r.name, "ours": r.ratio, "cudnn": r.cudnn_ratio,
             "flashinfer": r.fi_ratio,
             "correct": r.correct, "unsup": r.unsup}
            for r in rows
        ],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


_CSV_FIELDS = (
    "scenario", "bar", "phase", "dtype", "head_dim", "status",
    "ours_x", "cudnn_x", "fi_x", "base_us", "cand_us", "cudnn_us", "fi_us",
    "err_ratio", "note",
)


def _row_status(r, *, ref_label: str = "ref", cudnn_enabled: bool = True,
                fi_enabled: bool = True) -> str:
    if hasattr(r, "status_label"):
        return r.status_label(ref_label, cudnn_enabled=cudnn_enabled, fi_enabled=fi_enabled)
    if r.unsup:
        return "UNSUP"
    if r.correct:
        return "OK"
    return "FAIL"


def _breakdown_records(set_name: str, rows: list, *, ref_label: str = "ref",
                       cudnn_enabled: bool = True, fi_enabled: bool = True) -> list[dict]:
    """Per-scenario dicts for one named set, in config order, annotated with plot bar."""
    by_name = {r.name: r for r in rows}
    out = []
    for c in S.get_configs(set_name):
        r = by_name.get(c.name)
        if r is None:
            continue
        out.append({
            "scenario": c.name,
            "bar": S.plot_bar_for(c, set_name) or "",
            "phase": c.phase,
            "dtype": c.dtype,
            "head_dim": c.head_dim,
            "status": _row_status(r, ref_label=ref_label, cudnn_enabled=cudnn_enabled,
                                  fi_enabled=fi_enabled),
            "ours_x": "" if r.ratio is None else f"{r.ratio:.4f}",
            "cudnn_x": "" if r.cudnn_ratio is None else f"{r.cudnn_ratio:.4f}",
            "fi_x": "" if getattr(r, "fi_ratio", None) is None else f"{r.fi_ratio:.4f}",
            "base_us": "" if not r.base_ms else f"{r.base_ms * 1e3:.2f}",
            "cand_us": "" if not r.cand_ms else f"{r.cand_ms * 1e3:.2f}",
            "cudnn_us": "" if not r.cudnn_ms else f"{r.cudnn_ms * 1e3:.2f}",
            "fi_us": "" if not getattr(r, "fi_ms", None) else f"{r.fi_ms * 1e3:.2f}",
            "err_ratio": f"{r.err_ratio:.4g}" if r.err_ratio else "",
            "note": (r.note or "").replace("\n", " "),
        })
    return out


def write_breakdown_csvs(rows: list, *, arch: str, cfgset: str,
                         docs_dir: Optional[str] = None,
                         ref_label: str = "ref", cudnn_enabled: bool = True,
                         fi_enabled: bool = True) -> list[str]:
    """One CSV per member set (`claim` / `attntest` / …) with the per-scenario breakdown."""
    import csv
    docs = docs_dir or _DOCS
    os.makedirs(docs, exist_ok=True)
    written = []
    for set_name in S.plot_sets_for(cfgset):
        recs = _breakdown_records(set_name, rows, ref_label=ref_label,
                                  cudnn_enabled=cudnn_enabled, fi_enabled=fi_enabled)
        if not recs:
            continue
        path = os.path.join(docs, f"results_{arch}_{set_name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            w.writeheader()
            w.writerows(recs)
        written.append(path)
    return written


def format_breakdown_markdown(rows: list, *, arch: str, cfgset: str,
                              ref_label: str = "ref", cudnn_enabled: bool = True,
                              fi_enabled: bool = True) -> str:
    """Markdown tables (one per set) with the per-scenario result breakdown."""
    sections = []
    for set_name in S.plot_sets_for(cfgset):
        recs = _breakdown_records(set_name, rows, ref_label=ref_label,
                                  cudnn_enabled=cudnn_enabled, fi_enabled=fi_enabled)
        if not recs:
            continue
        lines = [
            f"### `{set_name}` ({len(recs)} scenarios · {arch} vs {ref_label})",
            "",
            "| scenario | bar | status | ours× | cuDNN× | FI× | base µs | cand µs | note |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in recs:
            note = r["note"]
            if len(note) > 48:
                note = note[:45] + "…"
            lines.append(
                f"| `{r['scenario']}` | {r['bar']} | {r['status']} "
                f"| {r['ours_x'] or '—'} | {r['cudnn_x'] or '—'} | {r['fi_x'] or '—'} "
                f"| {r['base_us'] or '—'} | {r['cand_us'] or '—'} | {note} |"
            )
        # Category geomean footer for bars that have ours×.
        by_bar: dict[str, list[float]] = defaultdict(list)
        for r in recs:
            if r["ours_x"] and r["status"] == "OK":
                by_bar[r["bar"]].append(float(r["ours_x"]))
        if by_bar:
            lines += ["", f"**Bar geomeans (ours vs {ref_label}):**"]
            order = S.bar_order(set_name) or tuple(by_bar.keys())
            for bar in order:
                xs = by_bar.get(bar) or []
                if not xs:
                    continue
                lines.append(f"- {bar} (n={len(xs)}): **{_geomean(xs):.3f}×**")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def plot_cfgset(arch: str, cfgset: str, measured: Optional[dict] = None, *,
                placeholder: bool = False) -> list[str]:
    """Emit one PNG per member set of `cfgset` (`all` → claim/attntest/edge/real)."""
    written = []
    for set_name in S.plot_sets_for(cfgset):
        per = scenario_results(set_name, measured, placeholder=placeholder, arch=arch)
        bars = aggregate_bars(set_name, per)
        path = plot_set(arch, set_name, bars, placeholder=placeholder, n_scen=len(per))
        if path:
            written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=["hopper", "blackwell", "both"], default="both")
    ap.add_argument("--set", dest="cfgset", default="all",
                    help="claim | attntest | edge | real | all (one plot per member set)")
    ap.add_argument("--placeholder", action="store_true",
                    help="fill every scenario with stable fake speedups (layout inspection)")
    ap.add_argument("--results", default=None,
                    help="JSON from eval (docs/last_results_<arch>.json); ignored with --placeholder")
    args = ap.parse_args()

    arches = ["hopper", "blackwell"] if args.arch == "both" else [args.arch]
    for arch in arches:
        measured = None
        if args.results and not args.placeholder:
            measured = load_measured(args.results)
        elif not args.placeholder:
            default = os.path.join(_DOCS, f"last_results_{arch}.json")
            if os.path.isfile(default):
                measured = load_measured(default)
            else:
                print(f"[warn] no results for {arch}; use --placeholder or --results")
                continue
        plot_cfgset(arch, args.cfgset, measured, placeholder=args.placeholder)


if __name__ == "__main__":
    main()
