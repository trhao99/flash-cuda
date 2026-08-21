---
name: optimize-fa4-blackwell
description: >-
  Optimize kernels/fa4_blackwell.cu (CUDA-only, no CUTLASS/cuBLAS/cuDNN deps) to
  match then beat FA4/cuDNN/FlashInfer/TRT-LLM on user-selected Blackwell
  scenarios. Use when tuning FA4 parity, NCU-profiling attention, rewriting
  fa4_blackwell.cu, or running AVO-style kernel search.
---

# Optimize FA4 Blackwell (CUDA-only)

## Goal

Iterate on `kernels/fa4_blackwell.cu` until it **matches, then exceeds**, the fastest of FA4 / cuDNN / FlashInfer (TRT-LLM cubin) on the scenario subset the user names. NVIDIA has already beaten FA4 with CUDA-only — treat parity as achievable. Aggressive rewrites OK.

## Hard rules

- **No** CUTLASS / CuTe / CuteDSL / cuBLAS / cuDNN includes or link deps in `fa4_blackwell.cu` (prose mentions OK).
- Correctness gate every change vs FA4 + fp32 ref (`bench.eval`). Never ship a faster wrong kernel.
- **Separate kernels** for decode vs prefill, long vs short, paged vs dense, etc. Unifying later is fine; one mega-kernel is too restrictive.
- Fan out **subagents** for research/rewrites; schedule benches/NCU across the **8 GPUs**.
- Prefer monotone lineage: keep only correct improvements on the target subset (see `tools/avo/`).

## Loop

1. **Scope** — which configs? (user subset, or losers from `docs/results_blackwell_*.csv` / latest eval).
2. **Baseline** — time ours vs FA4 vs cuDNN vs FlashInfer on those configs; target = max of the three refs.
3. **Classify** — host/launch vs wrong-route vs true kernel gap (`docs/questions.md`).
4. **NCU** — ours **and** the winning ref on the same scenario (locked clocks when possible). Compare stalls, MMA/XU %, spills, instr counts, occupancy, barriers.
5. **Steal** — distill FA4 CuteDSL / FlashInfer / TRT-LLM cubins; dump PTX/SASS; fix what we do differently.
6. **Rewrite** — one coherent change or a specialized kernel; use Blackwell features that matter (tcgen05, TMA, ST matrix, 2SM MMA, deep pipeline / double-buffer out, CLC persistent).
7. **Gate** — `bench.eval --target blackwell --configs …`; keep only correct wins. Repeat past parity.

## Ask yourself

- Can we distill / reverse engineer anything useful from the flash-attention 4 repository, the cutlass/cutedsl source code, or the flashinfer repository?
- Can we look into the SASS from cudnn/flashinfer (trtllm)/FA4 and figure out what we are doing differently, and fix that?
- Can we take a look at the generated PTX and reuse anything useful in our code?
- Should we use some specialized kernels for the scenarios where we are slower (e.g. decoding vs prefill, long vs short context etc)? Don't try to do everything in a single kernel, it's too restrictive. Better to write separate kernels for the various scenarios and unify them later if needed.
- Are we hitting the max available bandwidth and tflops, or not? If not, what parts of the kernels are holding us back? Profile with NCU both our kernel and the reference kernels (cudnn, flashinfer/trt-llm/fa4) and compare the bottlenecks, register spills, number of instructions launched, other relevant overheads, etc. Then use that information to try to rewrite our kernel to be as efficient as theirs.
- Are we doing some simple/naive warp specialization or other naive implementation instead of porting a full pipeline that is known to work well in cutlass/cutedsl/etc?
- Should we try an approach like AVO: https://arxiv.org/html/2603.24517v1 ?
- Is our kernel using all the optimized CUDA and PTX features that are needed to reach optimal performance on Blackwell? E.g. tcgen05, TMA / ST matrix / 2sm MMA / pipelining / double buffering out / clc persistent?

## Commands

```bash
# A/B (all GPUs by default)
python -m bench.eval --target blackwell --configs <csv> --gpus 0,1,2,3,4,5,6,7
python -m bench.eval --target blackwell --set real|edge|claim|all

# Candidate copy (parallel trials)
python -m bench.eval --target blackwell --cand /tmp/fa4_try.cu --namespace _try1 --configs <csv>

# AVO fitness (correctness then geomean)
tools/avo/fitness.sh /tmp/avo/var.cu _avo_ns "FA4_REBUILD=1"

# NCU (warm + one fwd)
ncu --set detailed -o /tmp/ncu_ours  python -m bench.microbenchmarks.ncu_probe <cfg>
# also: prof_cand.py / prof_fa4.py / prof_oracle.py / prof_real.py

# PTX/SASS of refs
python -m bench.microbenchmarks.dump_ref_ptx
tools/avo/dump_sass.sh <so> '<kernel_regex>' /tmp/sass.txt
```

## Pointers (read on demand)

| What | Where |
|------|--------|
| Host vs kernel checklist | `docs/questions.md` |
| CUTLASS FMHA | `reference/cutlass` (ex. `77_blackwell_fmha`) |
| FlashInfer / TRT cubins | `reference/flashinfer` (+ disassemble downloaded cubins) |
| CUDA / PTX / NCU docs | `reference/cuda-programming-guide.md`, `parallel-thread-execution.md`, `ncu.md` |
| AVO loop | `tools/avo/README.md`, `tools/avo/search.js` |
