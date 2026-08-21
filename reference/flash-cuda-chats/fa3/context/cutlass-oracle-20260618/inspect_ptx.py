"""Inspect CUTLASS source / generated PTX & SASS for the oracle.

The cutlass-oracle workflow needs to *see* exactly which Hopper primitives a high-performance
CUTLASS GEMM issues, in order, so we can mirror them in pure CUDA/PTX. This module:

  * compiles a `.cu` to PTX (`nvcc -ptx`) and/or extracts SASS from a built binary
    (`cuobjdump -sass`);
  * pulls out the ORDERED, de-duplicated sequence of the Hopper instruction forms that
    matter (wgmma, TMA bulk-tensor copy, mbarrier, fences, warpgroup reg (de)alloc,
    ldmatrix/stmatrix, exp2), with their operand *shapes* (e.g. m64n128k16);
  * greps the CUTLASS headers for the definition of a named atom (source inspection).

CLI:
    python -m fa3agent.cutlass_oracle.inspect_ptx ptx   <file.ptx>
    python -m fa3agent.cutlass_oracle.inspect_ptx sass  <file.so|.cubin|.o>
    python -m fa3agent.cutlass_oracle.inspect_ptx build <file.cu> [extra nvcc args...]
    python -m fa3agent.cutlass_oracle.inspect_ptx source <atom_name>   # grep cutlass headers
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

# ----------------------------------------------------------------------------------------
# Paths / toolchain
# ----------------------------------------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
# reference/libraries lives at <repo>/reference/libraries; this pkg at <repo>/fa3agent/cutlass_oracle
_REPO = os.path.dirname(os.path.dirname(_THIS))
REFLIB = os.path.join(_REPO, "reference", "libraries")
CUTLASS_DIR = os.environ.get("FA3_CUTLASS_DIR") or os.path.join(REFLIB, "cutlass")
CUTLASS_INC = os.path.join(CUTLASS_DIR, "include")
CUTLASS_UTIL_INC = os.path.join(CUTLASS_DIR, "tools", "util", "include")
FA3_HOPPER = os.path.join(REFLIB, "vllm-flash-attention", "hopper")

NVCC = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
CUOBJDUMP = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
GPU_ARCH = os.environ.get("FA3_GPU_ARCH", "sm_90a")

# Instruction forms worth tracking, longest/most-specific first so the regex picks the
# fullest token (e.g. the whole wgmma mnemonic incl. shape).
_FORM_PATTERNS = [
    ("wgmma_mma",     re.compile(r"wgmma\.mma_async\.sync\.aligned\.[\w.]+")),
    ("wgmma_fence",   re.compile(r"wgmma\.fence\.sync\.aligned")),
    ("wgmma_commit",  re.compile(r"wgmma\.commit_group\.sync\.aligned")),
    ("wgmma_wait",    re.compile(r"wgmma\.wait_group\.sync\.aligned(?:\s+\d+)?")),
    ("tma_load",      re.compile(r"cp\.async\.bulk\.tensor\.[\w.:]+")),
    ("tma_store",     re.compile(r"cp\.async\.bulk\.[\w.:]*tensor[\w.:]*\bglobal\b[\w.:]*")),
    ("cp_async",      re.compile(r"cp\.async\.(?:ca|cg)\.shared[\w.:]+")),
    ("mbarrier",      re.compile(r"mbarrier\.[\w.:]+")),
    ("fence_proxy",   re.compile(r"fence\.proxy\.[\w.:]+")),
    ("barrier",       re.compile(r"barrier\.(?:cluster|sync)[\w.:]*")),
    ("elect",         re.compile(r"elect\.sync[\w.:]*")),
    ("setmaxnreg",    re.compile(r"setmaxnreg\.[\w.:]+")),
    ("ldmatrix",      re.compile(r"ldmatrix\.sync\.aligned\.[\w.]+")),
    ("stmatrix",      re.compile(r"stmatrix\.sync\.aligned\.[\w.]+")),
    ("ex2",           re.compile(r"ex2\.approx\.[\w.]+")),
    ("tensormap",     re.compile(r"tensormap\.[\w.:]+")),
]


@dataclass
class InspectResult:
    source: str                       # the inspected file
    counts: dict = field(default_factory=dict)        # form -> total count
    shapes: dict = field(default_factory=dict)        # form -> {full_token: count}
    sequence: list = field(default_factory=list)      # ordered, de-duplicated full tokens
    note: str = ""

    def summary(self) -> str:
        lines = [f"# inspect: {os.path.basename(self.source)}"]
        if self.note:
            lines.append(f"# {self.note}")
        if not self.counts:
            lines.append("(no Hopper primitives found)")
            return "\n".join(lines)
        lines.append("\n## instruction-form counts")
        for form, n in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {form:14s} {n}")
        lines.append("\n## distinct forms (token -> count)")
        for form in self.counts:
            for tok, n in sorted(self.shapes.get(form, {}).items(), key=lambda kv: -kv[1]):
                lines.append(f"  {tok}   x{n}")
        lines.append("\n## ordered de-duplicated sequence (mirror this in pure PTX)")
        lines.extend(f"  {i:2d}. {tok}" for i, tok in enumerate(self.sequence))
        return "\n".join(lines)


def _scan_text(text: str, src: str) -> InspectResult:
    res = InspectResult(source=src)
    last = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        for form, pat in _FORM_PATTERNS:
            m = pat.search(line)
            if not m:
                continue
            tok = m.group(0)
            # normalize whitespace inside the token (e.g. "wait_group  0")
            tok = re.sub(r"\s+", " ", tok)
            res.counts[form] = res.counts.get(form, 0) + 1
            res.shapes.setdefault(form, {})
            res.shapes[form][tok] = res.shapes[form].get(tok, 0) + 1
            if tok != last:                     # collapse runs into a sequence
                res.sequence.append(tok)
                last = tok
            break
    return res


def inspect_ptx_text(text: str, src: str = "<ptx>") -> InspectResult:
    return _scan_text(text, src)


def inspect_ptx_file(path: str) -> InspectResult:
    with open(path, "r", errors="replace") as fh:
        return _scan_text(fh.read(), path)


def inspect_sass(binary: str, timeout: int = 120) -> InspectResult:
    """Disassemble a .so/.cubin/.o and scan the SASS (HGMMA/UTMALDG/LDGSTS forms)."""
    try:
        p = subprocess.run([CUOBJDUMP, "-sass", binary], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        r = InspectResult(source=binary, note=f"cuobjdump failed: {e}")
        return r
    text = p.stdout
    # SASS uses different mnemonics; scan for the Hopper-relevant ones.
    sass_pat = [
        ("HGMMA", re.compile(r"\bHGMMA\b[\w.]*")),
        ("QGMMA", re.compile(r"\bQGMMA\b[\w.]*")),
        ("IGMMA", re.compile(r"\bIGMMA\b[\w.]*")),
        ("UTMALDG", re.compile(r"\bUTMALDG\b[\w.]*")),
        ("UTMASTG", re.compile(r"\bUTMASTG\b[\w.]*")),
        ("LDGSTS", re.compile(r"\bLDGSTS\b[\w.]*")),
        ("LDSM", re.compile(r"\bLDSM\b[\w.]*")),
        ("STSM", re.compile(r"\bSTSM\b[\w.]*")),
        ("MUFU", re.compile(r"\bMUFU\b[\w.]*")),
        ("LDL", re.compile(r"\bLDL\b[\w.]*")),   # spills
        ("STL", re.compile(r"\bSTL\b[\w.]*")),
    ]
    res = InspectResult(source=binary)
    last = None
    for raw in text.splitlines():
        for form, pat in sass_pat:
            m = pat.search(raw)
            if not m:
                continue
            tok = m.group(0)
            res.counts[form] = res.counts.get(form, 0) + 1
            res.shapes.setdefault(form, {})
            res.shapes[form][tok] = res.shapes[form].get(tok, 0) + 1
            if tok != last:
                res.sequence.append(tok)
                last = tok
            break
    return res


def build_ptx(cu_path: str, out_ptx: Optional[str] = None, extra: Optional[list] = None,
              fa3_includes: bool = False, timeout: int = 600) -> tuple[int, str, str]:
    """Compile `cu_path` to PTX. Returns (rc, ptx_path, log). `fa3_includes` adds the FA3
    hopper headers + the FA3 forward compile defines so a single instantiation TU builds."""
    out_ptx = out_ptx or os.path.splitext(cu_path)[0] + ".ptx"
    cmd = [NVCC, "-ptx", "-O3", "-std=c++17", "-arch=" + GPU_ARCH,
           "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo",
           "-I", CUTLASS_INC, "-I", CUTLASS_UTIL_INC]
    if fa3_includes:
        cmd += ["-I", FA3_HOPPER,
                "-DFLASHATTENTION_DISABLE_BACKWARD", "-DFLASHATTENTION_DISABLE_DROPOUT",
                "-DFLASHATTENTION_DISABLE_APPENDKV", "-DFLASHATTENTION_DISABLE_SM8x",
                "-DFLASHATTENTION_DISABLE_CLUSTER", "-DFLASHATTENTION_DISABLE_UNEVEN_K",
                "-DFLASHATTENTION_VARLEN_ONLY", "-DFLASHATTENTION_PACKGQA_ONLY"]
    cmd += (extra or [])
    cmd += ["-o", out_ptx, cu_path]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
        return p.returncode, out_ptx, p.stdout
    except subprocess.TimeoutExpired:
        return 124, out_ptx, "<timeout>"


def grep_source(atom: str, max_hits: int = 40) -> str:
    """Find where an atom / template (e.g. SM90_64x128x16_F32BF16BF16_SS) is defined in the
    CUTLASS headers (source inspection)."""
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "-n", "--no-heading", atom, CUTLASS_INC]
    else:
        cmd = ["grep", "-rn", atom, CUTLASS_INC]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           text=True, timeout=60)
        lines = p.stdout.splitlines()[:max_hits]
        return "\n".join(lines) if lines else f"(no hits for {atom!r} in {CUTLASS_INC})"
    except Exception as e:  # noqa: BLE001
        return f"<grep failed: {e}>"


def _main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    cmd = argv[1]
    if cmd == "ptx":
        print(inspect_ptx_file(argv[2]).summary())
    elif cmd == "sass":
        print(inspect_sass(argv[2]).summary())
    elif cmd == "build":
        rc, ptx, log = build_ptx(argv[2], extra=argv[3:] or None,
                                 fa3_includes="--fa3" in argv)
        if rc != 0:
            print(f"build failed rc={rc}\n{log[-2000:]}")
            return rc
        print(f"PTX -> {ptx}\n")
        print(inspect_ptx_file(ptx).summary())
    elif cmd == "source":
        print(grep_source(argv[2]))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
