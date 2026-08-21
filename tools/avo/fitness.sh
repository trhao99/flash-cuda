#!/usr/bin/env bash
# AVO fitness function (portable). Build a candidate kernel .cu, correctness-gate it,
# then measure geomean speed vs the FA4 oracle. This is the single "score(x)" used by
# the evolutionary search; it is pure bash + the repo's bench harness, no Claude needed.
#
#   usage:  tools/avo/fitness.sh <variant.cu> <unique_namespace> "<env_flags>"
#   prints: RESULT ns=<ns> geomean=<X>x        (on success)
#           RESULT ns=<ns> FAIL stage=build|correctness
#
# env knobs:  AVO_REPO   (default: git toplevel)   AVO_WORK (default: /tmp/avo)
#             AVO_CONFIGS (default: the 3 hd128 dense short configs)
set -uo pipefail
REPO="${AVO_REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
WORK="${AVO_WORK:-/tmp/avo}"; mkdir -p "$WORK"
CFG="${AVO_CONFIGS:-fa3_dense_hd128_causal,fa3_dense_hd128_noncausal,fa3_dense_hd128_gqa}"
CU="$1"; NS="$2"; ENVF="${3:-}"
cd "$REPO" || exit 99
rm -f ".fa4_trials/$NS/lock" 2>/dev/null

# Phase 1: build + correctness. No GPU lock -> many candidates build in parallel (nvcc is CPU-bound).
# --no-cudnn: the fitness score is ours-vs-FA4, and cuDNN would add a graph build + 800 iterations
# per config, which does not fit the timeouts below.
env $ENVF timeout 260 python -m bench.eval --target blackwell --no-cudnn --cand "$CU" --namespace "$NS" --configs "$CFG" --no-perf \
    > "$WORK/build_$NS.log" 2>&1 \
  || { echo "RESULT ns=$NS FAIL stage=build"; tail -8 "$WORK/build_$NS.log"; exit 1; }
grep -qE "wrong: [1-9]|WRONG" "$WORK/build_$NS.log" \
  && { echo "RESULT ns=$NS FAIL stage=correctness"; grep -E "WRONG|wrong:" "$WORK/build_$NS.log" | tail -4; exit 2; }

# Phase 2: perf. flock serializes GPU timing across candidates (parallel timings corrupt each other);
# the .so is already cached from Phase 1 so this is fast.
env $ENVF timeout 200 flock "$WORK/gpu.lock" \
    python -m bench.eval --target blackwell --no-cudnn --cand "$CU" --namespace "$NS" --configs "$CFG" > "$WORK/perf_$NS.log" 2>&1
GM=$(grep -oE "geomean=[0-9.]+x" "$WORK/perf_$NS.log" | head -1)
echo "RESULT ns=$NS ${GM:-geomean=NA}"
grep -E "OK |WRONG|geomean=" "$WORK/perf_$NS.log" | tail -6
