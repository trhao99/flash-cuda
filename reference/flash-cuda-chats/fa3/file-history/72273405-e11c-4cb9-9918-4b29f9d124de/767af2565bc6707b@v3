# FA3 Hopper: perf work on worst configs + "reproduce the CUTLASS pipeline?" assessment

## Context

Two intertwined asks:
1. Improve perf on the configs where `kernels/fa3_hopper.cu` does worst vs the FA3 baseline (from the
   real-shapes perf run: paged decode/prefill of the SOL-ExecBench GQA shapes at 0.53–0.73×, and fp8
   hd256 at 0.20/0.34×).
2. Decide whether to do for FA3 what was done for FA4: **reproduce the CUTLASS/CuteDSL pipeline in pure
   CUDA** (not import it) — the move that took FA4 dense prefill from ~0.6× to ~1.0×.

Three independent explorations (FA4 rework, current FA3 kernel, reference CUTLASS FA3) were done. Findings
below drive the recommendation.

## Assessment: should we reproduce the CUTLASS FA3 pipeline? → No (it already is one)

FA4's win came because its old `fwd_kernel_ws2` had a **hand-designed warp role split that diverged from
the CUTLASS FA4 collective**; reproducing the collective's exact roles/stages/tmem/MMA-schedule closed a
real structural gap (~0.6→0.78×), and a separate launch-overhead fix took it to ~1.0×. FA4 also kept `ws2`
for sub-wave varlen, and its single-tile paged/decode shapes "don't map onto the pipeline structure" and
stayed laggards.

`fa3_hopper.cu` is **already a faithful reproduction of the CUTLASS FA3 collective**. Mapping the reference
checklist (`reference/vllm-flash-attention/hopper/`) to the current kernel, 1:1:
- 1 producer + 1–2 consumer warpgroups, `setmaxnreg` dealloc/alloc split — present (`pf::`, lines ~164, 345–354).
- Multi-stage TMA smem ring + mbarrier `full[]`/`empty[]` handshake (STAGES 3–5) — present.
- Intra-WG QK(n) ∥ PV(n−1) ∥ softmax(n) software pipeline — present (`qk_rs_pv_g`, ~562).
- 2-WG named-barrier pingpong for tensor-core arbitration — present (`PP_BASE`, ~1121).
- Persistent + dynamic `atomicAdd` work-stealing scheduler — present (`prefill_persist_kernel`, ~1243).
- GQA q*rep M-packing, split-KV + separate `combine_kernel` — present.
- Paged TMA fast path (page_size % tile == 0) + cp.async gather fallback — present (~2210).

So the structural rework that helped FA4 is a no-op for FA3. A CUTLASS-style change would help **only hd256
prefill** (register allocator — the kernel itself flags this at line ~2263), and FA4's experience shows the
residual there is a ptxas/CuTe **codegen** ceiling pure CUDA can't cross. **Recommendation: do not
re-architect the pipeline.** The slow configs are feed/policy-bound, addressable inside the existing structure.

Key confirmation: the reference CUTLASS FA3 **also cannot TMA at page_size=1** (its TMA paged path requires
`page_size % kBlockN == 0`); it uses the same cp.async gather. So beating it at page_size=1 is about a
**better gather + split policy**, not a different pipeline.

## Root causes of the slow configs (all feed/policy, not structure)

1. **page_size=1 → no TMA → per-row dependent-address cp.async gather.** The `use_tma` gate (~2211) requires
   `block_size % 8 == 0`, which fails at 1. The fallback `gather_row_k/v` (~760/783, called ~1086–1109)
   loads one key row per producer thread, each behind a dependent `page_table[]` lookup (long-scoreboard).
   This starves **every** real config (both decode and prefill). Highest-leverage single fix.
2. **`num_splits=1` collapse for large-base long-context decode.** `launch_decode_split` (~2771–2788):
   `base = tiles_m*hkv*b`; for h32 + batch 16–64, `base ≥ SMs` → `want=1` → no split, so each CTA streams
   the whole 4k–32k context on **one consumer WG** at ~8% occupancy (latency-exposed). The wave-quant
   heuristic was tuned on the attntest decode family (h6–h16, short ctx); it under-splits the real family.
3. **Missing native fp8 hd256.** `launch_prefill_fp8` only instantiates `<128,128>` (~2390); hd256 fp8 takes
   the bf16-convert path (1× tensor throughput), not scalar. Rare corner (no SOL model uses hd256).

Dead code to ignore: `attn_kernel` (~73) and the whole `dec::decode_kernel`/`launch_decode` path (~2422/2661)
are not on any live dispatch edge.

## Plan (prioritized by value / effort; each phase gated on no attntest regression)

### Phase 1 — page_size=1 paged K/V feed (highest value: hits all 9 real configs; low structural risk)
Port FA4's proven fix (`fwd_kernel_fa4_paged`, commit `bdd7aa9`, "off-critical-path page-row resolve"):
- Resolve `page_table[]` → physical K/V row base pointers **once**, ahead of the load loop, into smem
  (a `prow[]` array), breaking the per-row dependent-address chain in `gather_row_k/v`.
- Issue **coalesced** cp.async from the prefetched pointers (contiguous 16B per lane), instead of one
  dependent-address row per thread.
- Files: the `pf::` gather producer in `kernels/fa3_hopper.cu` (`gather_row_k`/`gather_row_v` ~760/783 and
  their call sites in `prefill_kernel` ~1086–1109 and the persist/decode-split paths). Reference pattern:
  `kernels/fa4_blackwell.cu:8107+` (`prow_c[FA4KVS][WTN]` prefetch).
- Regression guard: the attntest paged configs (`attn_kv_hd128_page1/page4/page128`, all decode/gather) must
  stay ≥ their current ratios; TMA-eligible paged configs are untouched (this only changes the non-TMA path).

### Phase 2 — decode split/occupancy policy for large-base long-context (medium value; needs NCU)
- Add a **context-length-aware split floor** to `launch_decode_split` (~2771–2788) and
  `launch_decode_split_fp8` (~2858–2870): when per-CTA K-tiles are large (long ctx) and `base ≥ SMs`, allow
  splitting past one wave up to a bounded per-split K-tile count, trading a partial-wave tail for latency
  hiding. Re-derive the threshold from an NCU split sweep that **includes the real h32 family** (the current
  heuristic was tuned only on h6–h16).
- Regression guard: re-run the full attntest decode family; the wave-quant behavior for short-ctx / small-base
  decodes must not change (gate the new branch on long ctx + large base only).

### Phase 3 — native fp8 hd256 (low value, optional; defer unless requested)
- Add `launch_prefill_fp8_t<256,256>` + `launch_decode_split_fp8<256,256>` instantiations, mirroring the
  <128,128> native fp8 path (V-transpose, P→fp8 round-trip). Rare corner; recommend deferring — the convert
  path is correct, just slow, and no SOL-ExecBench model uses hd256.
- (Also low-value: `attn_dense_hd256_causal` bf16 at 0.848× is register-allocator-bound — the one spot a
  CUTLASS codegen edge would help; pure-CUDA ceiling is low. Skip.)

## Verification (per phase)

- Correctness first: `python -m bench.eval --no-perf --namespace _distill_fa3_C --set real` and `--set attntest`
  (both must stay N/N correct vs `_vllm_fa3_C`).
- Perf: `python -m bench.eval --namespace _distill_fa3_C --set real` (target: lift the 0.53–0.73× cluster) and
  `--set attntest` (guard: attntest geomean stays ≥ 1.04×, no config regresses materially).
- Profiling for Phase 1/2 root-cause confirmation + tuning: `bench/ncu_probe.py` / `bench/prof_one.py` on
  `attn_real_llama8b_dec_s4k` (decode) and `attn_real_llama8b_pf_s16k` (prefill).
- On success, refresh `bench/plot.py` `RESULTS` (Hopper) and regenerate `docs/per_scenario_speedup.png`
  (dev-env python has matplotlib).

## Scope decision needed
Recommend Phase 1 first (biggest, safest win — fixes the paged feed for all real configs). Phase 2 after, with
profiling. Phase 3 deferred unless you want the fp8-hd256 corner closed.

---

## OUTCOME (implemented, committed f69aaf1)

The profiling redirected the plan. What actually worked and what didn't:

**SHIPPED — coalesced cooperative K/V gather (real geomean 0.651 -> 0.748, no regressions):**
The real bottleneck was the non-TMA gather's *layout*: "1 thread = 1 row" issues 16 serial 16B cp.async, so
each warp instruction reads one 16B chunk from 32 different rows -> 32 uncoalesced L1TEX sectors (74.9% mem
pipe on paged prefill; ~47% BW decode). Fixed by TPR=HD/8 lanes cooperating on one row (1 coalesced 16B chunk
each) + a decoupled smem page-row prefetch (producer-WG bar.sync, id 1) so the page_table LDG is off the load
dependency chain. `if constexpr (COOP)` guards it to bf16 hd128/64/256; writes stay koffR/voffR (byte-exact).
Short-ctx decode now beats FA3 (dec_s512 1.06-1.11x); long-ctx decode + paged prefill 0.53-0.73 -> 0.62-0.84x.
attntest 1.043 -> 1.058 (page1/page4 gather path also improved). All correct (real 10/10, attntest 64/64, edge 31/31).

**REFUTED (profiled, reverted, documented inline):**
- More num_splits: non-persistent split path adds a multi-wave tail + combine overhead -> strictly regresses.
- 2 CTA/SM (env FA3_DEC2CTA): neutral -- halving STAGES (5->3) to fit a 2nd CTA cancels the extra-warps benefit;
  and once the gather was coalesced the decode became DRAM-latency-bound, where more warps still didn't help.
- Per-row TMA for block_size=1 (env FA3_PS1_TMA): 1.6x SLOWER -- 128 tiny [64,1] boxes/tile, TMA descriptor
  overhead dominates a 128B transfer (this is why the block_size%8==0 gate exists).

**NOT PURSUED (clear next levers, higher effort / uncertain / low value):**
- Persistent scheduler for paged (wave-collapse 8->1 for prefill; the decode is now DRAM-latency-bound at ~40%
  vs FA3's ~55%). Big change (add the coalesced gather into prefill_persist_kernel's continuous ring; deadlock
  risk). The most plausible remaining win for the long-ctx family.
- Native fp8 hd256 (attn_dense_hd256_fp8 0.20x / kv_hd256_fp8_decode 0.34x): worst configs, but a rare corner
  (no SOL-ExecBench model uses hd256 fp8) and needs a new prefill_fp8_kernel<256,256> instantiation.
- dense hd256 causal 0.848x: register-allocator bound (the kernel itself flags this as CUTLASS-codegen-only).
