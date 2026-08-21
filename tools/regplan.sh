#!/usr/bin/env bash
# Compile fa4_blackwell.cu with a given setmaxnreg register plan and report per-kernel
# LDL/STL (spill) counts for the fa4-family kernels. Compile-only: no benchmark, no torch op.
#   tools/regplan.sh 192 64 64
set -euo pipefail
cd "$(dirname "$0")/.."
SM=${1:-192}; CORR=${2:-48}; PROD=${3:-80}
TI=$(python -c "import torch,os;print(os.path.dirname(torch.__file__)+'/include')")
PI=$(python -c "import sysconfig;print(sysconfig.get_paths()['include'])")
OUT=/tmp/regplan_${SM}_${CORR}_${PROD}.o
nvcc -O3 -std=c++17 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda \
  -gencode arch=compute_100a,code=sm_100a -lineinfo -DFA3_LIB_NAME=_x \
  -DFA4REG_SM=$SM -DFA4REG_CORR=$CORR -DFA4REG_PROD=$PROD \
  -I"$TI" -I"$TI/torch/csrc/api/include" -I"$PI" \
  -c kernels/fa4_blackwell.cu -o "$OUT" 2>&1 | grep -E "error|Error" && exit 1 || true
python3 - "$OUT" "$SM/$CORR/$PROD" <<'PY'
import re, subprocess, sys
o, tag = sys.argv[1], sys.argv[2]
sass = subprocess.run(['cuobjdump', '-sass', o], capture_output=True, text=True).stdout
tot = {}
for f in re.split(r"\n\t*Function : ", sass)[1:]:
    name = f.split('\n')[0].strip()
    d = subprocess.run(['c++filt', name], capture_output=True, text=True).stdout.split('(')[0]
    if 'bfloat16' not in d:
        continue
    n = len(re.findall(r'\bLDL\b', f)) + len(re.findall(r'\bSTL\b', f))
    for k in ('fwd_kernel_fa4_paged', 'fwd_kernel_fa4dk', 'fwd_kernel_fa4<'):
        if k in d + '<':
            tot.setdefault(k, []).append((d.split('fa4::ws::')[-1], n))
            break
print(f"--- plan {tag}")
for k, v in sorted(tot.items()):
    for d, n in v:
        print(f"    spills={n:5d}  {d}")
PY
