# CUTLASS trace — FA3 prefill, head_dim=128, bf16

This file records **which CUTLASS / CuTe functions and templates the vLLM-FlashAttention-3
forward kernel instantiates** for the worst-performing prefill config in
`run_20260616_011948`, so the oracle can re-implement each one in pure CUDA/PTX.

## 1. Target config (chosen from the full config matrix)

Picked the worst prefill configs from the champion (step035) per-config ratios
(`reports/metrics_all.json`, `cand_latency / base_latency`):

| config | ratio (base/cand) | cand ms | base ms | head_dim | dtype | causal | paged |
|---|---|---|---|---|---|---|---|
| **prefill_hd128_bf16_chunk16k** | **0.241** (4.15x slower) | 7.17 | 1.73 | 128 | bf16 | yes | yes |
| prefill_hd128_fp8_causal | 0.246 | 0.208 | 0.051 | 128 | fp8e4m3 | yes | yes |
| mixed_chunked_hd128_bf16 | 0.286 | 0.497 | 0.142 | 128 | bf16 | yes | yes |
| prefill_hd256_bf16_causal | 0.357 | 0.162 | 0.058 | 256 | bf16 | yes | yes |

**Chosen: `prefill_hd128_bf16` (causal, paged).** Rationale: worst absolute gap, the
production-critical head_dim, and bf16 (simplest dtype → isolates the GEMM problem before
adding fp8 dequant). `chunk16k` and `mixed_chunked` run the *same* hd128-bf16 device kernel;
they differ only in problem shape (q_len/k_len), so the CUTLASS templates instantiated are
identical.

Device template instantiated (from `hopper/instantiations/flash_fwd_hdim128_bf16_paged_sm90.cu`):

```cpp
run_mha_fwd_<90, cutlass::bfloat16_t, /*kHeadDim=*/128, /*kHeadDimV=*/128,
             /*Split=*/false, /*PagedKVNonTMA=*/true, /*Has_softcap=*/false, /*PackGQA=*/true>
```

## 2. Tile / kernel shape (resolved at compile time)

From `tile_size_fwd_sm90(headdim=128, headdim_v=128, is_causal, is_local, element_size=2, ...)`
(`hopper/tile_size.h:40-45`), with `Use_one_mma_wg=false` (prefill, q_len>1):

```
kBlockM = 128
kBlockN = 128       (is_causal || is_local || paged_kv_non_TMA ? 128 : 160)
kHeadDim = 128
kHeadDimV = 128
MmaPV_is_RS = true  (P fed to the 2nd GEMM from registers)
IntraWGOverlap = true
kStages = 2         (Arch >= 90)
ClusterShape = <1,1,1>   (Varlen ⇒ Enable_cluster=false)
NumMmaWarpGroups = 2     (AtomLayout M = kBlockM/64 = 2 ⇒ 256 MMA threads)
NumProducerThreads = 32  (1 warp issues TMA)
```

Tile shapes:
- `TileShape_MNK    = <128, 128, 128>`   (QK GEMM:  M=kBlockM, N=kBlockN, K=kHeadDim)
- `TileShape_MNK_PV = <128, 128, 128>`   (PV GEMM:  M=kBlockM, N=kHeadDimV, K=kBlockN)

## 3. CUTLASS functions / templates used (the list to re-implement)

Source: `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`, `epilogue_fwd.hpp`,
`flash_fwd_kernel_sm90.h`.

### 3.1 GEMM #0 — S = Q·Kᵀ  (QK^T)
- Selector: `cute::GMMA::ss_op_selector<bf16, bf16, float, Shape<128,128,128>>()`
  (`mainloop...:93-99`, MmaQK_is_RS=false → **SS**: both operands from shared memory).
- Resolved atom: **`SM90_64x128x16_F32BF16BF16_SS`**
  → PTX **`wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16`**
  (confirmed in `cute/arch/mma_sm90.hpp:ss_op_selector`, Tile_N=128 ⇒ n128 atom;
   `cute/arch/mma_sm90_gmma.hpp` for the inline PTX).
- TiledMMA: `make_tiled_mma(atom, AtomLayout<Shape<2,1,1>>)` → 2 warpgroups stacked in M
  (each does m64), 256 threads. K=128 ⇒ **8** `wgmma` k16 steps per QK GEMM, accumulator fp32.

### 3.2 GEMM #1 — O = P·V  (PV)
- Selector: `cute::GMMA::rs_op_selector<bf16, bf16, float, Shape<128,128,128>,
  GMMA::Major::K, GMMA::Major::MN>()` (`mainloop...:105-113`, MmaPV_is_RS=true → **RS**:
  A=P from registers, B=V from smem; V is MN-major for bf16).
- Resolved atom: **`SM90_64x128x16_F32BF16BF16_RS`**
  → PTX **`wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16`** (A from registers).
- P is the softmax(S) result: fp32 accumulator → bf16 via `cutlass::NumericArrayConverter`,
  re-laid-out to the `wgmma` A-register fragment layout (the "P permute"), then consumed RS.
- TiledMMA AtomLayout `<2,1,1>`. K(=kBlockN)=128 ⇒ **8** `wgmma` k16 steps, accumulator fp32.

### 3.3 TMA loads (global → shared)
- `GmemTiledCopyQ = cute::SM90_TMA_LOAD`; `TMA_Q = make_tma_copy_A_sm90(...)`
  → PTX `cp.async.bulk.tensor.<rank>d.shared::cluster.global.mbarrier::complete_tx::bytes`.
  Tile [128(M) × 128(K)] bf16, swizzled smem.
- `GmemTiledCopyKV = sm90_cluster_shape_to_tma_atom(1) = SM90_TMA_LOAD`;
  `TMA_K = make_tma_copy_B_sm90(...)` — K tile [128(N) × 128(K)] per stage.
- `TMA_V = make_tma_copy(SM90_TMA_LOAD, ... select<1,0,2,3>(StrideV) ...)` — V loaded
  transposed (Vt) into smem; tile [128(headdimV) × 128(N)] per stage.
- NOTE paged path: when `PagedKVNonTMA=true` (paged KV without TMA) K & V are loaded with
  **`cp.async` (`LDGSTS` / `cp.async.cg`)** via `PagedKVManager`, NOT TMA. Q still uses TMA.
  The TMA-paged variant (`get_pagedkv_tma` heuristic true) uses TMA for paged K/V instead.

### 3.4 Shared-memory layouts / swizzle
- `cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, bf16, M, K>()`
  → 128-byte-K swizzle atom `GMMA::Layout_K_SW128_Atom<bf16>` = `Swizzle<3,4,3>` over an
  `8×64` bf16 core. Applied to Q, K, V(t), and P smem via `tile_to_shape`.

### 3.5 Pipeline / synchronization
- `cutlass::PipelineTmaAsyncNoCluster<kStages=2>` (cluster size 1) for K/V (TMA) producer-
  consumer → PTX `mbarrier.init`, `mbarrier.arrive.expect_tx`, `mbarrier.try_wait`,
  `mbarrier.arrive`. `cutlass::PipelineAsync<2>` when K/V come via cp.async (paged-non-TMA).
- Warp specialization: 1 producer warpgroup (issues TMA / cp.async, `setmaxnreg.dec`) +
  2 consumer (MMA) warpgroups (`setmaxnreg.inc`), ordered by `NamedBarrier`
  (`hopper/named_barrier.hpp`), `cutlass::arch::warpgroup_reg_alloc/dealloc`.
- `wgmma.fence` / `wgmma.commit_group` / `wgmma.wait_group` around each MMA group.

### 3.6 Softmax + epilogue (supporting, not GEMM)
- `flash::Softmax` (`softmax.h`): online softmax in log2 domain, `exp2` via `MUFU.EX2`,
  row-max / row-sum reductions over the MMA accumulator fragment, rescale of O.
- `flash::CollectiveEpilogueFwd` (`epilogue_fwd.hpp`): O write-out (regs → smem via
  `Copy_Atom<SM90_U32x4_STSM_N>` = `stmatrix`, then TMA store or vectorized `STG`),
  `softmax_lse` write.

## 4. Reduction to GEMM building blocks (oracle scope)

The oracle re-implements the two **wgmma + TMA GEMM building blocks** in pure CUDA/PTX and
proves they match CUTLASS performance + output on the same problem:

1. **QK GEMM (TN, SS)**: `C[M,N] = A[M,K] · B[N,K]ᵀ`, A=Q row-major, B=K row-major,
   K=head_dim=128, M=N=seqlen. bf16 in, fp32 accum. wgmma `m64n128k16` (operands SS).
2. **PV GEMM (NN, RS)**: `O[M,N] = P[M,K] · V[K,N]`, P row-major (regs), V row-major,
   N=head_dim_v=128, K=seqlen. bf16 in, fp32 accum. wgmma `m64n128k16` (A in regs).

See `kernels/` for the CUTLASS reference and the pure-CUDA/PTX implementations, and
`inspect_ptx.py` for the PTX/SASS extraction used to mirror the instruction sequence.

## 5. Confirmation (PTX inspected, not just source-read)

- Standalone CUTLASS oracle GEMM `kernels/cutlass_gemm.cu` (`-DGEMM_MODE_QK`) compiles to a
  kernel whose mangled name + PTX contain exactly `MMA_64x128x16_F32BF16BF16_SS`,
  `SM90_TMA_LOAD`, `Swizzle<3,4,3>`, `KernelTmaWarpSpecializedCooperative`.
  `inspect_ptx.py ptx` extracts the ordered sequence:
  `mbarrier.init → setmaxnreg.dec/inc → mbarrier.arrive.expect_tx →
   cp.async.bulk.tensor.Nd...mbarrier::complete_tx → wgmma.fence →
   wgmma.mma_async.m64n128k16.f32.bf16.bf16 ×8 → commit_group → wait_group`.
- Authoritative FA3 TU `flash_fwd_hdim128_bf16_paged_sm90.cu` (built against FA3's pinned
  CUTLASS 3.8.0 at `vllm-flash-attention/csrc/cutlass`, NOT the standalone 4.5.2 — its
  epilogue/SM100 API drifted) → PTX saved at
  `flash_fwd_hdim128_bf16_paged_sm90_cut38.ptx`. `inspect_ptx.py` confirms:
  `wgmma.mma_async.m64n128k16.f32.bf16.bf16` ×656, `cp.async.bulk.tensor.5d` ×18 (Q via TMA),
  `cp.async.cg.shared.global.L2::128B` ×960 (paged K/V via cp.async — the PagedKVNonTMA path),
  `stmatrix.x4.m8n8.b16` ×112 (O epilogue), `ex2.approx.f32` ×2706 (softmax), mbarrier +
  setmaxnreg warp-specialization. The matmul building block is identical to the oracle GEMM.

## 6. Pure-CUDA/PTX re-implementation — results (QK)

`kernels/ptx_gemm.cu`: **persistent** kernel (grid = `occupancy × #SMs`, each CTA strides over the
`tiles_m × tiles_n` output grid), 2 warpgroups (256 threads) / 128×128 tile, hand-built GMMA
descriptor (no-swizzle K-major), `wgmma.mma_async.m64n128k16.f32.bf16.bf16`, vectorized 128-bit
(float4) smem fill, and a **float2 (64-bit) accumulator→global epilogue** writing adjacent column
pairs directly from registers. Bit-identical to CUTLASS output; **exceeds** CUTLASS perf (median
of 9 trials, CUDA-event timed, 50-iter warmup, L2 flush between trials — `oracle.py bench`):

| shape (M×N×K) | CUTLASS TFLOPS | pure-PTX TFLOPS | ratio | output max\|Δ\| |
|---|---|---|---|---|
| 2048²×128            | 54.8 | 82.1  | **1.50** | 0.0 |
| 4096²×128            | 61.9 | 96.1  | **1.55** | 0.0 |
| 8192²×128            | 67.7 | 107.4 | **1.59** | 0.0 |
| 16384²×128           | 68.2 | 112.8 | **1.65** | 0.0 |
| 2048×16384×128 (real chunk16k QK) | 63.2 | 102.5 | **1.62** | 0.0 |

### Why pure-PTX beats CUTLASS here (ncu, H200, 4096²×128 / 8192²×128)
The standalone QK GEMM has **fp32 global output**, so it is *store-bound*, not compute-bound:
at 8192² it writes 244 MB but reads only 4.9 MB (A+B), arithmetic intensity ≈64 FLOP/byte vs the
H200 balance point ≈206 — so neither kernel is tensor-bound (tensor pipe only **6–12% active**).
The race is won on the memory/epilogue path:
- **CUTLASS** (`GemmUniversal`, already persistent: grid=132, 1 wave) is **L2-bound at 73%** and
  caps at **~68 TFLOPS for every tile shape** tried (128×128, 128×256, 256×128, 128×128×128;
  256×256 is worse) — its fp32 epilogue stages accumulators smem→L2→DRAM.
- **Old non-persistent ptx** (v2): 2 blocks/SM, 3.9 waves at 4096, scattered 4-byte stores →
  L1TEX 57% / L2 60%, tensor 6% → only 0.82–0.96× (load-at-start + store-at-end latency exposed,
  too few warps to hide it).
- **New persistent ptx**: float2 stores straight from registers (no smem epilogue staging) →
  L1TEX **85%**, L2 only 49%, DRAM-write 26%, 1 wave. Writing output directly is more
  bandwidth-efficient than CUTLASS's smem-staged fp32 epilogue ⇒ 1.5–1.65×.

The two changes that flipped 0.9× → 1.6×: **(1) persistent + linear tile scheduler** (kills
wave-quantization, biggest win at small shapes — 2048² 0.82→1.50) and **(2) float2 register→global
epilogue** (halves store transactions, biggest win at large shapes).

> Caveat: in the *real* FA3 kernel the QK output S stays on-chip (feeds softmax → PV), so this
> standalone store-bound win does not translate 1:1 into a faster end-to-end attention kernel; it
> proves the wgmma GEMM engine + memory path match/beat CUTLASS on the traced building block with
> identical I/O. `cp.async` loads were tried and *regressed* (113→82 TFLOPS): single-buffer with
> immediate `wait_group` exposes latency with no overlap, and a double-buffer pipeline would halve
> occupancy (128 KB smem ⇒ 1 block/SM). The float4 LDG→STS load is the empirical winner here.

### Lessons (carry into the deliverable kernel)
- **Persistent grid + tile scheduler** is essential: a one-tile-per-CTA launch loses ~2× to
  wave-quantization and exposed per-tile load/store latency. Size grid via
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor × multiProcessorCount`, capped at tile count.
- **Epilogue store width matters**: emit the widest aligned store the accumulator fragment allows
  (here adjacent column pairs ⇒ float2) directly from registers; don't scatter 4-byte stores.
- `wgmma` requires the `compute_90a` *virtual* arch: build with
  `-gencode arch=compute_90a,code=sm_90a` (plain `-arch=sm_90a` resolved to `compute_90` here
  and ptxas rejected `wgmma`).
- `wgmma` reads smem through the **async proxy**; smem written by ordinary `STS` must be made
  visible with `fence.proxy.async.shared::cta` (after `__syncthreads`) before the wgmma, or you
  get an intermittent, concurrency-dependent ~0.1% wrong-element race. (FA3 emits this fence.)
  Also `__syncthreads()` at the end of each persistent tile so the next tile's STS does not clobber
  smem still being read by another warp's wgmma.
- The GMMA SS descriptor for no-swizzle K-major bf16: layout_type=0, LBO=128B, SBO=256B,
  per-k16-step address advance = MN·16·2 bytes; smem element offset
  `(k/16)·MN·16 + (mn/8)·128 + ((k%16)/8)·64 + (mn%8)·8 + (k%8)`.
- fp32 accumulator fragment (m64nNk16): thread→ rows `{16·warp+lane/4, +8}`, cols
  `8·c + 2·(lane%4) + {0,1}`, registers ordered `[rlo c0, rlo c1, rhi c0, rhi c1]` per c-block.

## 7. PV GEMM results (RS operand)

`kernels/ptx_gemm.cu -DGEMM_MODE_PV`: `O[M,N] = P[M,K]·V[K,N]`, K=n-block keys=128, N=head_dim_v.
wgmma **RS** (`MMA_64x128x16_F32BF16BF16_RS`): A=P in **4×b32 registers** (K-major required), B=V^T
in smem. Same persistent grid + float2 epilogue as QK. Bit-identical to CUTLASS:

| shape (M×N×K) | CUTLASS TFLOPS | pure-PTX TFLOPS | ratio | output max\|Δ\| |
|---|---|---|---|---|
| 2048²×128  | 54.6 | 57.6 | 1.06 | 0.0 |
| 4096²×128  | 61.9 | 64.2 | 1.04 | 0.0 |
| 8192²×128  | 67.7 | 67.7 | 1.00 | 0.0 |
| 16384²×128 | 68.2 | 71.1 | 1.04 | 0.0 |

Key facts for the RS path:
- A operand is **4 `uint32_t` registers** (not an smem descriptor); the RS atom requires A **K-major**
  (`static_assert tnspA==Major::K`). Only `tnspB` exists (B in smem).
- **A-fragment register packing** (m64n128k16, lane `l`, warp `w` in WG, `tg=l%4`, `group=l/4`,
  thread row `r0 = wg·64 + w·16 + group`), per k-step `kb` with `k0 = 16·kb + 2·tg`:
  `a0={P[r0][k0],P[r0][k0+1]}`, `a1={P[r0+8][k0],…}`, `a2={P[r0][k0+8],…}`, `a3={P[r0+8][k0+8],…}`
  (low 16 bits = lower k index). This is exactly the **accumulator** layout with N→K — i.e. the QK
  output S, after softmax→bf16, drops straight into the PV A operand (the point of RS). Worked first try.
- B = V^T: V is `[K,N]` row-major; stored as `sVt[gmma_k_off(n,k,CTA_N)] = V[k·N + n]` (same K-major
  layout as QK's B). Loaded by a coalesced float4 read of 8 consecutive `n` scattered to 8 k-strided
  smem slots. *This scatter (smem bank conflicts) is why PV sits at ~1.0× vs QK's ~1.6×*; an
  `ldmatrix.trans`/`stmatrix` smem transpose (what FA3 uses) would lift it. Still ≥1.0× and bit-exact.

## 8. Component oracles (each validated standalone vs CUTLASS / fp32 ref)

Built one component at a time (`kernels/`), each with its own correctness + perf check:

| component | file | validation | result |
|---|---|---|---|
| QK GEMM (SS) | `ptx_gemm.cu -DGEMM_MODE_QK` | vs CUTLASS GEMM | bit-identical, **1.5–1.65×** |
| PV GEMM (RS) | `ptx_gemm.cu -DGEMM_MODE_PV` | vs CUTLASS GEMM | bit-identical, **1.0–1.06×** |
| **Memory movement (TMA)** | `tma_load.cu` | round-trip vs input; BW vs cp.async | `nbad=0`; load BW **≥ cp.async** (2019 vs 1981 GB/s @32k×128). `cuTensorMapEncodeTiled` + `cp.async.bulk.tensor.2d…mbarrier::complete_tx` + `mbarrier.init/arrive.expect_tx/try_wait.parity` all exercised. (SWIZZLE_NONE; 128B-swizzle+wgmma integration deferred — paged FA3 loads K/V via cp.async anyway.) |
| **Synchronization (mbarrier)** | `tma_load.cu` | (as above) | mbarrier phase-tracked pipeline works; full mainloop pipeline → §10 |
| **Math/numerics (softmax)** | `softmax_test.cu` | online log2 softmax vs 2-pass fp32 ref | O max\|Δ\| **1.0e-7**, LSE **9.5e-7** (causal & non-causal). `exp2f`(→`ex2.approx.f32`), per-block running max/sum rescale, `LSE=scale·max+ln(sum)` (`__logf`). |

## 9. Fused forward kernel (NO CUTLASS) — `kernels/fused_attn.cu`

One CTA = (m_block, batch·head); 2 MMA warpgroups (256 thr) each own 64 of the 128 query rows.
Per n-block: `qk_gemm` (SS) → causal mask → `softmax_row` (online, ×2 owned rows) → P fp32-frag
→ bf16 **register repack == PV RS A-operand** (the linear pack proven in §7, no smem round-trip) →
`pv_gemm` (RS) → epilogue (`O/=Σ`, `LSE=scale·max+ln Σ`, bf16 O + fp32 LSE store). Reuses the
oracle's GMMA descriptor / K-major layout / accumulator-fragment math verbatim.

**Correctness** vs fp32 CPU ref (`attn_ref.cuh`): O max\|Δ\|=**2.7e-3** (= bf16 rounding of the bf16
output, as expected), LSE\|Δ\|≤1.9e-6 — causal & non-causal, every size. This is the headline: a
self-contained attention forward with zero CUTLASS, bit-faithful to FA3's numerics.

There are three mainloop variants in `fused_attn.cu`, selected at compile time:
- **default** (`fused_attn_fwd`): single-buffered, synchronous loads — the simple correct baseline.
- **`-DPIPE`** (`fused_attn_fwd_pipe`): cp.async double-buffered K + V load overlapped with QK.
- **`-DOVL`** (`fused_attn_fwd_ovl`): the fast path (see §11) — cp.async K **and** V + intra-WG
  software pipeline. This is the canonical `fused_attn` binary and what `ab_fa3.py` runs.

## 10. Closing the gap: software pipeline + cp.async V (the 2× win this iteration)
NCU on the lockstep baseline (H200, 4096 B1H16): tensor pipe **10–12.8%**, occupancy **12.5%** (1
CTA/SM, 197 regs/thr), dominant stall **long-scoreboard 6.3/issue** (global-load latency). The two
fixes that moved the needle, each validated against the fp32 ref before integrating:

1. **Intra-warpgroup software pipeline (FA3's core overlap).** Lag PV by one n-block so
   `softmax(nb)` (FFMA/SFU pipes) runs while `PV(nb−1)` (tensor cores) is still in flight —
   `wgmma.wait_group 1` keeps the PV group pending across the softmax. The O-rescale is hoisted out
   of softmax (`rescale_o`, before both wgmma issues) so nothing writes `acc_o` between the QK and PV
   issues. Softmax-stats (`softmax_stats`) is split from the O-rescale so it never touches `acc_o`
   while PV accumulates into it.
2. **cp.async V via `wgmma` imm-trans-b=1.** The old `load_vt` did an LDG→8×STS *scatter* into the
   K-major (V^T) layout — a synchronous LDG that was the remaining dominant stall. Instead V is now
   `cp.async`'d in **natural (dv-contiguous) layout** (`gmma_mn_off`, the no-swizzle MN-major
   core-matrix tiling derived from CUTLASS `Layout_MN_INTER_Atom`: 8×8 atom, MN contiguous; LBO=2048B
   / SBO=128B / k-block-stride=4096B) and read by the PV wgmma with **imm-trans-b=1**. 16B coalesced
   copies, no scatter, and fully prefetchable. Both K and V are double-buffered cp.async (one group
   per iter holds {K(nb+1),V(nb)}; `wait_group 1` keeps one in flight).

Result: long-scoreboard **6.3→1.7**, tensor pipe **10→17.7%**, throughput **~doubled**.

### A/B vs real FA3 (`ab_fa3.py`, `_vllm_fa3_C.fwd`, non-paged causal bf16 hd128, H200)
| B,H,S | FA3 ms | mine ms | FA3 TFLOP/s | mine TFLOP/s | ratio | O\|Δ\| |
|---|---|---|---|---|---|---|
| 8,16,2048 | 0.280 | 0.719 | 490 | 191 | **0.39** | 2.7e-3 |
| 4,16,4096 | 0.511 | 1.326 | 538 | 207 | **0.39** | 2.7e-3 |
| 2,16,8192 | 0.929 | 2.537 | 592 | 217 | **0.37** | 2.7e-3 |
| 2,32,4096 | 0.514 | 1.326 | 535 | 207 | **0.39** | 2.7e-3 |
| 1,32,8192 | 0.932 | 2.537 | 590 | 217 | **0.37** | 2.7e-3 |

Was ~0.2×; now **~0.37–0.39× FA3**, correctness unchanged (O\|Δ\|=2.7e-3 = bf16 rounding, all ok).

## 11. Remaining gap to 1.0× (NCU after the §10 fixes, 4096 B1H16)
tensor pipe **17.7%**, SM throughput **21.4%**, **warps active 9.4%**; stalls now balanced and small:
barrier **1.6**, long-scoreboard **1.7**, mio 0.8, wait 0.7. The wall is now **occupancy**: 256
threads = 8 warps = 12.5% theoretical, and the math warps spend their cycles in softmax/exp so there
aren't enough warps to also keep the tensor pipe saturated (the intra-WG overlap fills *some* of it
with PV, hence 17.7%). Remaining FA3 levers, in ROI order — each needs decoupled load sync (the
cooperative `__syncthreads` loads block desyncing the warpgroups), i.e. a producer warpgroup:
- **(1) Producer (3rd) warpgroup + mbarrier loads** → 384 thr (18.75% occ ceiling) and frees the 2
  consumers from load issue; replaces `__syncthreads` with per-buffer mbarrier so consumers desync.
- **(2) 2-consumer ping-pong** (`NamedBarrier` scheduler, needs (1)): WG0 softmax overlaps WG1 MMA so
  the tensor pipe is fed during *both* phases (not just by the intra-WG PV).
- **(3) TMA** (`cp.async.bulk.tensor`) for K/V — single-thread issue, frees address math.
- **(4) 2-CTA cluster** for M=256 tiles (FA3's hd128 config) to amortize K/V.
These are a structural warp-specialized rewrite of the mainloop, not local tweaks.

## 12. Warp-specialized rewrite (levers 1+2+intra-WG overlap) — `0.37→0.5× FA3`
Implemented the producer-warpgroup + 2-consumer ping-pong, then layered the intra-WG overlap on top.
Two compile-time variants (both 3 warpgroups = 384 thr, `__launch_bounds__(384,1)`, `setmaxnreg`
dec=40 producer / inc=232 consumer — exactly FA3's reg split):
- **`-DWS`** (`fused_attn_fwd_ws`): producer + ping-pong, NO intra-WG overlap. 2-stage K+V ring.
- **`-DWSO`** (`fused_attn_fwd_ws_ovl`): + intra-WG PV-lag overlap. 3-stage K+V ring (so blocks `nb`
  (QK), `nb−1` (PV) and the producer's prefetch `nb+1` are all live). **The fast path.**

**Producer WG** (`wg==0`): `cp.async`-loads K (`gmma_k_off`) and V (`gmma_mn_off`, dv-contiguous) into
the smem ring and signals readiness with `cp.async.mbarrier.arrive.noinc` on a per-stage `full[]`
mbarrier (init count = 128 producer thr; the `.noinc` deferred arrive flips the barrier when the
stage's cp.async complete). Waits `empty[]` (consumers' `mbarrier.arrive`, count 256) before reuse.
Q loaded once → `barQ`. (TMA not used: SWIZZLE_NONE TMA lands row-major, not the wgmma core-matrix
layout — would need the deferred swizzle integration.)

**Consumer ping-pong** (transcribed from `named_barrier.hpp` + the no-overlap mainloop scheduler
protocol in `mainloop_fwd_sm90_tma_gmma_ws.hpp`): 2 named barriers (`bar.sync`/`bar.arrive`, 256 thr).
Per iter the token wraps both GEMM issues — `sync(self); QK; PV; arrive(other)` — so each WG's softmax
(token released) overlaps the *peer* WG's GEMMs, and (WSO) its own in-flight `PV(nb−1)`. Kickoff =
cwg0 `arrive(self)` (FA3's `mma_init`). `qkpv_issue` issues QK+PV under a single `wgmma.fence` with all
descriptors / `s` / `acc_o` written beforehand and `scale-d≡1` (s pre-zeroed) — removes C7513/C7520
(though wgmma serialization turned out not to be the binding limit here).

### A/B vs real FA3 (`ab_fa3.py --bin fused_attn_wso`, H200)
| B,H,S | FA3 ms | mine ms | FA3 TFLOP/s | mine TFLOP/s | ratio | O\|Δ\| |
|---|---|---|---|---|---|---|
| 8,16,2048 | 0.299 | 0.577 | 460 | 238 | **0.52** | 2.7e-3 |
| 4,16,4096 | 0.515 | 1.014 | 534 | 271 | **0.51** | 2.7e-3 |
| 2,16,8192 | 0.944 | 1.872 | 582 | 294 | **0.50** | 2.7e-3 |
| 1,32,8192 | 0.948 | 1.873 | 580 | 294 | **0.51** | 2.7e-3 |

Progression: lockstep **0.2×** → cp.async+intra-WG (`OVL`) **0.39×** → producer+ping-pong (`WS`)
**0.39–0.45×** → +intra-WG overlap (`WSO`) **0.50–0.52×**. Correctness unchanged.

### NCU on WSO (4096 B1H16): the wall is latency-hiding at 1 CTA/SM
tensor pipe **34.7%** (OVL was 17.7%), warps active **18.0%** (3 WG = 1 block/SM — *the same
occupancy FA3 runs at*; smem 224 KB + 64.5 K regs both cap at 1 block, so 2 blocks/SM is impossible).
L2 **62.7%**, DRAM 3.5% (K/V mostly L2-resident). Active warps/scheduler **2.87**, eligible **0.64**,
**no-eligible 58%**. Stalls/issue: long-scoreboard **1.62**, barrier **1.53** (ping-pong peer-wait),
wgmma-wait **1.19**. With only ~3 warps/scheduler the SM cannot fully hide these — this is the
structural ceiling of the approach, and the remaining 2× to FA3 lives in *pipeline/L2 efficiency*.

## 13. Next levers toward 1.0× (from FA3/CUTLASS, not from scratch)
- **TMA + 128B-swizzle for K/V** (FA3's actual load path): 1 issue/tile vs ~4096 cp.async (cuts
  producer instruction pressure), `L2_PROMOTION_128B` (helps the 62.7% L2), and lets the producer
  prefetch deeper. Needs the swizzle↔wgmma-descriptor integration deferred in §8 (`tma_load.cu`).
- **NCU-guided micro-tuning vs FA3's own kernel**: profile `_vllm_fa3_C.fwd` with the same metrics to
  pinpoint where its 3 warps/scheduler stay busier (softmax scheduling, deeper V stages).
- **2-CTA cluster** (M=256, FA3's hd128 config) to amortize K/V loads across SMs.
- Wire the component into `fa3agent` (replace `sass/cutlass_oracle.py`).
See `FA3_REPLICATION_PLAN.md` for the full CUTLASS→raw-PTX checklist.

## 14. NCU-profiled FA3 itself + code/PTX diff → root cause of the 2× gap
Profiled the *real* `_vllm_fa3_C.fwd` (`prof_fa3.py`, B1 H16 S4096 causal, H200) with the same metrics,
read its kernel/scheduler source, and diffed its PTX against ours. **Same metric, side by side:**

| metric | **FA3** | **mine (WSO)** | read |
|---|---|---|---|
| tensor pipe (hmma active) | **68.6%** | 34.7% | FA3 ~2× — it is *compute*-bound, we are not |
| duration (this shape) | 154 µs | ~360 µs | the 2× |
| grid | **(132,1,1)** = #SMs | (tiles_m·BH) = 100s | **FA3 is persistent** |
| regs/thread, blocks/SM | 168, 1 | 168, 1 | identical occupancy |
| smem/block | **167 KB (2-stage K/V)** | 224 KB (3-stage) | depth is *not* our problem |
| L2 throughput | 36.8% | 62.7% | we hammer L2 ~2× harder |
| stall long-scoreboard | **0.41** | 1.62 | **we wait on memory; FA3 doesn't** |
| stall barrier | 2.25 | 1.53 | FA3 waits on mbarrier (TMA) — "good" stall |
| stall wgmma-wait | 1.58 | 1.19 | FA3 waits on a *busy* tensor pipe |

**Code diff** (`flash_fwd_kernel_sm90.h` 318–468): FA3 is a **persistent kernel** — grid = #SMs, each CTA
loops `scheduler.get_next_work()` over work tiles; the K/V pipeline state (`smem_pipe_read/write`) is
declared *outside* the loop so it **never drains between tiles**, and `get_next_work` is called *before*
the epilogue ("so the next tile is ready to go"). The producer is a **single warp**
(`SingleProducerWarp`, warps 1–3 `return`) because TMA only needs one thread to issue a full-tile load.

**PTX diff** (`flash_fwd_*_cut38.ptx`): the wgmma itself is **identical** to ours —
`wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16`, predicate scale-d, 8 wgmma per
`commit_group`/`wait_group` (656 wgmma / 82 groups). So our instruction issue is already optimal; the
gap is **not** in the GEMM. The non-paged kernel loads K/V via **`cp.async.bulk.tensor` (TMA)**; only
the *paged* variant uses `cp.async.cg` (960×, like us).

**Source-level SASS on our kernel** (`--page source`): stall samples concentrate on **mbarrier
spin-wait branches `@!P0 BRA` (~14%)** and **`WARPGROUP.DEPBAR` wgmma-waits (~12%)**. The ratio to FA3
is **constant (~0.5×) across S** ⇒ steady-state, not prologue/wave-quant.

**Root cause:** our producer is a *full warpgroup* issuing ~32 `cp.async`/thread/tile (≈4096
LDGSTS/tile). It is **instruction-issue bound on the loads**, so it can't keep the ring full → consumers
spin on `full[]` (the 14% `@!P0 BRA`) → the tensor pipe idles at 34.7%. FA3's long-scoreboard is 0.41
*because* its loads are TMA (one `cp.async.bulk.tensor` per tile, latency hidden behind the mbarrier —
which is why its **barrier** stall is high, 2.25, but harmless).

### Ablation: `-DABLATE_NOLOAD` (producer signals `full[]` with NO load; garbage data, same tensor work)
| variant | tensor pipe | long-scoreboard | barrier | duration |
|---|---|---|---|---|
| WSO (real loads) | 34.7% | 1.62 | 1.53 | ~360 µs |
| WSO no-load (free data) | **46.8%** | 2.08 | 2.19 | 271 µs |
| FA3 (real TMA loads) | **68.6%** | 0.41 | 2.25 | 154 µs |

⇒ the gap is **two-part**: (a) producer load-issue costs ~12 tensor-pts (34.7→46.8) — **TMA fixes this**;
(b) even with *free* loads we cap at 46.8% vs FA3's 68.6% — that residual is **consumer scheduling +
per-tile epilogue drain** (non-persistent CTAs idle the tensor pipe during each tile's
normalize+store), which **persistent + cross-tile epilogue overlap** fixes. long-scoreboard staying 2.08
with zero global loads proves it is *not* the K/V loads — it tracks the mbarrier/epilogue path. So
**TMA alone ≈0.68×; TMA + persistent are both required for ~1.0×.**

### Re-prioritized plan (highest leverage first)
1. **TMA for K/V** (the steady-state fix): replace the producer's per-element `cp.async` with
   `cp.async.bulk.tensor.2d`, one issue per tile from a single warp. Converts long-scoreboard→hidden
   async, removes the producer instruction wall, drops L2 pressure. Requires a host `CUtensorMap` and a
   TMA-compatible smem layout (start SWIZZLE_NONE matching the current `gmma_k_off`/`gmma_mn_off`
   core-matrix layout so the wgmma descriptors are unchanged; add 128B swizzle only if still L2-bound).
2. **Persistent scheduler** (boundary/L2 fix): grid = #SMs, static-strided work loop with the K/V ring
   carried across tiles (single token kickoff stays valid; Q single-buffer + full/empty handshake like
   FA3's `QueryEmpty`). Removes wave-quant and warms the pipeline across tiles.
3. 2-CTA cluster (M=256) last.

## 15. TMA + 128B-swizzle: ground-truth layouts/descriptors + K-major validated bit-exact
Executing lever 1. Derived the exact swizzle/descriptors from cute (`swz_probe.cu`, host-compiled
against the vendored CUTLASS) and validated the K-major load+wgmma path end-to-end (`tma_swz_test.cu`).

**cute ground truth (`GMMA::Layout_{K,MN}_SW128_Atom<bf16>` tiled to [128,128]):**
- **K-major** (`Q`,`K` operands): `off(mn,k) = (k>>6)*8192 + swz(mn*64 + (k&63))`, `swz(o)=o^(((o>>7)&7)<<4)`.
  Verified **exact for all 16384 elements** vs cute (`mismatches=0`). Descriptor `make_gmma_desc<K>`:
  LBO=16B, SBO=1024B, base_off=0, **lt=1 (B128)**; per-k-step element start `(kb&3)*16 + (kb>>2)*8192`.
- **MN-major** (`V` operand, `imm-trans-b=1`): must use FA3's **exact** `SmemLayoutVt =
  tile_to_shape(Layout_MN_SW128_Atom, (dv,keys,stages), Step<_2,_1,_3>)` (the `Step` reorder matters —
  a plain `tile_to_shape` gives a *different*, interleaved layout). Real layout (cute dump):
  `Sw<3,4,3> o ((64,2),(8,16)):((1,8192),(64,512))`. ⇒ the **two dv-halves are at clean stride 8192**
  (`dv=63→63`, `dv=64→8192`), just like K. keys-step = `kb*1024` elems = **`kb*2048` B**.
  `make_gmma_desc<MN>` on the real layout = **LBO field=1024 (→16384B), SBO field=64 (→1024B)**, lt=B128.
  (My first probe used a non-`Step` tiling and got the wrong LBO/SBO=1024/2048 — corrected here.)

**K-major TMA path — bit-exact (`tma_swz_test.cu`, H200):** TMA-load Q,K with `CU_TENSOR_MAP_SWIZZLE_128B`
(box `{64 inner=128B, 128 rows}`, two boxes per operand at smem `0`/`8192` = the 2 k-blocks) then the QK
wgmma with the K-major B128 descriptor → `max_abs_diff=0.0000` vs fp32 CPU ref. **Key insight:** the
TMA-write swizzle and the wgmma-read swizzle are the *same hardware swizzle*, so they round-trip even
though the physical bytes do **not** match cute's logical `layK` offsets (a manual-store "pathA" using
the cute offsets fails; TMA "pathB" passes). ⇒ I never need to reproduce cute's byte layout — only make
TMA-write and wgmma-read self-consistent. For K that means {2 boxes at 0/8192} ↔ {descriptor k-step
+8192 for kb≥4}.

**FA3's own V-load code (`mainloop_fwd_sm90_tma_gmma_ws.hpp`, the "compare the code" step):**
- bf16 (`!Is_FP8`, `!V_colmajor`): `Transpose_V=false` — **V is NOT transposed**; loaded directly.
- `MmaMajorV = TmaMajorV = GMMA::Major::MN` — V is TMA-loaded MN-major into `SmemLayoutVt` and the PV
  wgmma reads that *same* layout (`SmemLayoutVtMma==SmemLayoutVt`), `TensorStorageNoTranspose` (no `vt`).
- `NumProducerThreads = NumThreadsPerWarp` under TMA → **single-warp producer** (matches §14 PTX/code).
- `SmemLayoutAtomVt = ss_smem_selector<MN, bf16, dv, N>`, tiled with `Step<_2,_1,_3>` (vs K's
  `Step<_1,_2,_3>`) — this reorder is exactly why the dv-blocks interleave. The V TMA is built by one
  `make_tma_copy(SM90_TMA_LOAD, mV, take<0,2>(SmemLayoutVt), select<1,2>(TileShape_MNK_PV))`.

**V path — bit-exact (`tma_swz_test_v.cu`, H200):** SS PV GEMM `O[128,128]=P@V`, P=A K-major TMA (box
`{64,128}` ×2 @ 0/8192), V=B MN-major TMA (box `{64 dv-inner=128B, 128 keys}` ×2 @ dv0=0/64 → smem
0/8192), PV wgmma with `imm-trans-b=1` and the MN B128 descriptor (LBO=16384B, SBO=1024B, keys-step
`kb*2048`B) → **`max_abs_diff=0.0000`** vs fp32 CPU ref. So both operand majornesses are now validated
end-to-end under TMA+128B-swizzle. SWIZZLE_128B requires the box inner ≤128B (256B is rejected by
`cuTensorMapEncodeTiled`), hence the 2-box split for the 128-wide contiguous dim — which is fine because
both layouts place their two halves at clean smem stride 8192.

**Integration recipe (both validated):** K-major (Q,K): box `{64,128}`×2, desc LBO=16/SBO=1024/lt=1,
k-step `(kb&3)*16+(kb>>2)*8192` elems, trans=0. MN-major (V): box `{64,128}`×2 @ dv0=0/64, desc
LBO=16384/SBO=1024/lt=1, keys-step `kb*2048`B, trans-b=1.

## 16. `-DWST`: TMA + 128B-swizzle integrated → **0.50→0.71–0.76× FA3** (the predicted TMA win)
Forked `fused_attn_fwd_ws_ovl` into `fused_attn_fwd_wst` (`fused_attn.cu`): **identical** consumer
schedule (producer + 2-consumer ping-pong + intra-WG PV overlap, 3-stage ring), but the load path is now
FA3's: a **single producer thread** issues `cp.async.bulk.tensor.2d` (TMA) per box with
`mbarrier.arrive.expect_tx` complete-tx (init count 1, the TMA byte-transaction flips the phase), and the
consumer GEMMs read **128B-swizzled** smem via the B128 descriptors (`qk_gemm_sw`/`qkpv_issue_sw`/
`pv_gemm_t_sw`). Crucial code-trace simplification: **K and V are the *same* TMA op** — both are
`[rows, HD=128]` row-major in gmem → box `{64,128}`×2; only the wgmma descriptor (K-major vs MN-major)
differs. One `CUtensorMap` shape over the `[BH·S, HD]` tensor serves Q/K/V (`cuTensorMapEncodeTiled`,
`SWIZZLE_128B`, box `{64,128}`); the kernel passes Q/K/V coords `{x=HD-half·64, y=bh·S+row}`. smem must
be ≥1024B-aligned (`extern __shared__ __align__(1024)`); SWIZZLE_128B rejects a 256B inner box, hence the
2-box split.

### A/B vs real FA3 (`ab_fa3.py --bin fused_attn_wst`, H200)
| B,H,S | FA3 ms | mine ms | FA3 TFLOP/s | mine TFLOP/s | ratio | O\|Δ\| |
|---|---|---|---|---|---|---|
| 8,16,2048 | 0.296 | 0.388 | 464 | 354 | **0.76** | 2.7e-3 |
| 4,16,4096 | 0.513 | 0.702 | 536 | 392 | **0.73** | 2.7e-3 |
| 2,16,8192 | 0.933 | 1.317 | 589 | 417 | **0.71** | 2.7e-3 |
| 1,32,8192 | 0.935 | 1.318 | 588 | 417 | **0.71** | 2.7e-3 |

Matches the §14 ablation prediction (TMA alone ≈0.68×; here 0.71–0.76×). Correctness unchanged.

## 17. Persistent scheduler (lever 2) — NOT the bottleneck; the ping-pong barriers are
Implemented FA3's `StaticPersistentTileScheduler` as `fused_attn_fwd_wstp` (`-DWSTP`): grid=#SMs, each CTA
strides `work += gridDim.x` over `tiles_m·BH` tiles (`work → m_block=work%tiles_m, bh=work/tiles_m`), and
the **K/V smem ring phase is carried continuously across tiles** via a global block counter so the
pipeline never drains (matching `flash_fwd_kernel_sm90.h` keeping `smem_pipe_*` outside the work loop). Q
is single-buffered with a producer↔consumer handshake (`barQ` ready / `qEmpty` free = FA3's `QueryEmpty`
named barrier). **Gotcha (deadlock):** the per-tile ping-pong kickoff (`cwg0` arrive `SB_self`) leaves
exactly 128 pending arrivals on named-barrier 1 every tile; in a single-tile kernel that's harmless, but
persistent accumulates it → deadlock. Fixed with a `cwg1` balancing arrive + a 256-thread consumer
rendezvous (id 3) per tile.

**Result (clean, H200; an earlier run was contaminated by a stray GPU process):** persistent is
**neutral for causal** (391 vs 391 TFLOP/s @ 4096 B4H16) and **+3% for noncausal** (459 vs 445). Reason:
non-persistent already gets the **hardware block scheduler's dynamic load-balancing** for free, which the
*static* persistent schedule gives up (round-robin can't rebalance causal's triangular tile costs). The
A/B ratio is also **flat ~0.73× across S=2k→8k**, i.e. **not wave-quant-bound** — so persistent targets the
wrong thing. (A *dynamic* work-stealing scheduler would recover causal balance but, per the NCU below,
overlap—not scheduling—is the limit.)

### NCU on `-DWST` (post-TMA) → the real bottleneck is the inter-WG ping-pong
Causal 4096 B4H16 steady state: **tensor 42.6%, barrier stall 2.22 (dominant), long-scoreboard 0.68**
(TMA fixed it), wgmma-wait 0.70, active-warps 13.6% (1 CTA/SM, reg+smem bound — same as FA3). The 2-WG
ping-pong makes one WG's 128 threads sit on a named barrier while the peer runs its GEMM → barrier-bound,
tensor pipe starved. **Fix `-DNOPP`:** drop the ping-pong token (intra-WG PV-lag overlap already keeps each
WG's own tensor busy; the mbarrier/wgmma data-deps preserve correctness). **Universal +6-7%.**

### Final A/B vs FA3 (`ab_fa3.py`, causal prefill shapes, H200)
| B,H,S | FA3 ms | WST+NOPP | ratio | WSTP+NOPP | ratio | O\|Δ\| |
|---|---|---|---|---|---|---|
| 8,16,2048 | 0.297 | 0.372 | **0.80** | 0.386 | 0.77 | 2.7e-3 |
| 4,16,4096 | 0.508 | 0.657 | **0.77** | 0.661 | 0.78 | 2.7e-3 |
| 2,16,8192 | 0.933 | 1.230 | 0.76 | 1.220 | **0.77** | 2.7e-3 |
| 1,32,8192 | 0.930 | 1.231 | 0.76 | 1.218 | **0.77** | 2.7e-3 |

Net: **0.50× (WSO) → 0.73× (TMA) → ~0.77× (TMA + drop ping-pong)**, all bit-exact-correct, no CUTLASS.
Best all-rounder: `-DWSTP -DNOPP` (ties causal, +3% noncausal). Remaining gap to 1.0× is pure overlap
quality (FA3 42.6%→68.6% tensor): next levers are a tighter QK/PV issue schedule (fewer fences/waits) and
a 2-CTA/SM-friendly register/smem budget — both overlap, not memory or scheduling.

## 18. `-DSEQB`: async wgmma batches (the wgmma-serialization root cause) → **~0.88× FA3**
The §17 "barrier stall 2.22 (dominant)" was misattributed to the ping-pong. NCU on the **steady state**
(`--launch-skip 60`) of `WST+NOPP` showed barrier **1.94** even with ping-pong *off*. SASS root cause:
the warp-specialized mainloop's `qkpv_issue_sw` (QK group + PV group both in flight, deferred
`wait_group<1>/<0>`) compiles to **`WARPGROUP.DEPBAR.LE gsb0,0x0` after *every* HGMMA** (DEPBAR=32 for 32
HGMMA, ptxas warning C7513) — i.e. the 8-wgmma batches were **fully serialized**, and each `.sync.aligned`
DEPBAR is a *warpgroup barrier* → that *was* the "barrier stall."

**Key experiment that isolated it:** `fused_attn_fwd_ws` (warp-spec, single-group `commit+wait`) is
**async (DEPBAR=2)**; `ws_ovl`/`wst` (deferred 2-groups-in-flight) are **serialized (DEPBAR=32)**. The
identical `qk_batch` body is async in the basic kernel and serialized in the WS kernel ⇒ the trigger is
**keeping two wgmma groups in flight**, which ptxas serializes here. Disproven as the cause (all still 32
DEPBAR): hoisted/precomputed descriptors, single-asm-block 16-wgmma (`*_batch_blk`), `cute`-style 32-bit
`desc_add`, `__shfl` uniform bases, FA3's `warpgroup_fence_operand` before+after, removing all fences. It
is **register-pressure-driven**: with ~200 live regs (64 `s` + 64 `acc_o` + 32 `P` + 16 descriptors)
ptxas recomputes the PV descriptors *between* the two pending groups (the C7513 "non-wgmma inst defines a
wgmma input mid-pipeline"). FA3 avoids it only because `cute`'s descriptor *iterators* live in **uniform
registers** (separate file) — unreachable from hand-written wgmma.

**Fix — give up intra-WG overlap, keep async batches (`-DSEQB`):** each consumer WG runs **sequential**
`QK batch (commit+wait0) → softmax → PV batch (commit+wait0)`. One group in flight at a time ⇒ **DEPBAR
32→2**, no C7513. Then re-add **inter-WG ping-pong** (each WG's softmax overlaps the *peer* WG's GEMM;
token wraps both GEMM phases, 2 `bar.sync`+2 `bar.arrive`/iter). This is exactly FA3's per-WG structure;
only the *intra*-WG PV-lag overlap is dropped.

NCU progression (causal 4096 B4H16, steady state): serialized `WST` tensor **50%** / barrier 1.94 →
`SEQB` (no PP) tensor **57.9%** / barrier 1.23 → `SEQB`+ping-pong tensor **61.2%** / barrier 1.08.

### Final A/B vs FA3 (`ab_fa3.py`, `-DWST -DSEQB`, causal prefill, H200)
| B,H,S | FA3 ms | mine ms | FA3 TFLOP/s | mine TFLOP/s | ratio | O\|Δ\| |
|---|---|---|---|---|---|---|
| 8,16,2048 | 0.297 | 0.328 | 462 | 419 | **0.91** | 2.7e-3 |
| 4,16,4096 | 0.509 | 0.580 | 540 | 474 | **0.88** | 2.7e-3 |
| 2,16,8192 | 0.931 | 1.075 | 590 | 511 | **0.87** | 2.7e-3 |
| 2,32,4096 | 0.513 | 0.586 | 536 | 470 | **0.88** | 2.7e-3 |
| 1,32,8192 | 0.939 | 1.079 | 586 | 510 | **0.87** | 2.7e-3 |

Net this session: **0.77× → ~0.88× FA3** (min-of-trials hits 0.94×), bit-exact, no CUTLASS, **async
wgmma**. Production build = `-DWST -DSEQB`. Persistent (`-DWSTP -DSEQB`) and QK-only ping-pong both tested
**slightly slower**. The last ~12% is the **intra-WG PV-lag overlap** that ptxas refuses to keep async for
hand-written wgmma (FA3 reaches it only via `cute`'s uniform-register descriptor iterators) — a genuine
ptxas codegen ceiling for the no-CUTLASS constraint, documented above.

## 19. The §18 "ptxas ceiling" was WRONG — intra-WG overlap is a WAR hazard, now fixed → **~0.91× FA3**
§18's "register-pressure / uniform-register ceiling" conclusion was a **misdiagnosis**. The intra-WG
2-groups-in-flight overlap (QK(nb) ∥ PV(nb-1)) is fully reachable from hand-written inline-asm wgmma. Three
pieces of evidence, gathered by reading FA3/cute source + the real `.so` SASS:

1. **FA3's own SASS is async (proof cute does NOT need a special trick we can't reach).** Disassembling
   the shipped `_vllm_fa3_C.abi3.so` (192 sm_90a cubins) and counting per-kernel: the fwd kernels run
   **HGMMA:DEPBAR ≈ 12:1** (e.g. 168 HGMMA / 14 DEPBAR, ratio 0.08); 616/784 fwd kernels are ≤0.1. Its
   `HGMMA … gdesc[UR16], R24` reads the descriptor from a **uniform register**, with descriptors
   precomputed by `UIADD3.64 UR8,UR16,0x2` — exactly cute's `DescriptorIterator::operator+` (32-bit add to
   the low half, high half = compile-time-const layout/swizzle; `mma_traits_sm90_gmma.hpp:323`).
2. **An isolated 2-group test is async with PLAIN inline asm** (no cute): two deferred 8-wgmma groups
   (QK SS + PV RS) + `wait_group 0` → **HGMMA 16 / DEPBAR 1**, even reusing a single `gdesc[UR4]`. So
   reusing one UR is fine and register pressure (~186 regs) is NOT the trigger.
3. **Growing that test into a loop reproduced the serialization** (DEPBAR 16/16, C7513) — and the trigger
   was a single line: writing `P` (the next PV's A-operand) from softmax **while the in-flight PV(nb-1)
   still reads `P`**. That is a genuine **WAR hazard**, so ptxas *correctly* inserts a `WARPGROUP.DEPBAR`
   before every HGMMA to protect `P`. Moving the `P` write to **after** `wgmma_wait0()` (PV done) →
   **DEPBAR 16→2**, no C7513.

This is precisely what FA3 does: in `mainloop_fwd_sm90` it overwrites `tOrP` via `convert_type_out(...)`
only **after** `warpgroup_wait<0>` (`mainloop_fwd_sm90_tma_gmma_ws.hpp:1272-1276`). The fix in our mainloop
is one reorder:
```
wgmma_wait1();  softmax_stats(...);        // softmax overlaps in-flight PV(nb-1)  (reads s, NOT P)
wgmma_wait0();                             // PV(nb-1) finished -> it no longer reads P
for i: P[i] = pack_bf16(s[2i], s[2i+1]);   // NOW safe to overwrite P for next iter's PV
```
Applied to `fused_attn_fwd_wst` (and `wstp`). Net SASS: `wst14` **DEPBAR 32→4** (= 1 per 8-wgmma group,
matching FA3's 0.08 ratio), C7513 gone. `-DCUTE` (uniform-`cwg` via `__shfl` like cute
`canonical_warp_group_idx` + base-once `make_smem_desc` + `desc_add`) shaves another ~3%.

NCU (causal 4096 B4H16, steady state) — **mine vs FA3**: tensor **64.5% vs 71.3%**, SM throughput
**59.6% vs 70.2%**, barrier stall **0.71 vs 2.22** (we stall *less*; FA3 hides more behind active warps),
active-warps **13.7% vs 18.8%**. Remaining gap = pipeline fill/drain + causal tile imbalance (FA3's
persistent+dynamic scheduler amortizes it; our static `wstp` is neutral here).

### Final A/B vs FA3 (`ab_fa3.py`, `-DWST -DCUTE`, causal prefill, H200)
| B,H,S | FA3 ms | mine ms | FA3 TFLOP/s | mine TFLOP/s | ratio | O\|Δ\| |
|---|---|---|---|---|---|---|
| 8,16,2048 | 0.299 | 0.325 | 459 | 423 | **0.92** | 2.7e-3 |
| 4,16,4096 | 0.515 | 0.567 | 534 | 485 | **0.91** | 2.7e-3 |
| 2,16,8192 | 0.939 | 1.043 | 585 | 527 | **0.90** | 2.7e-3 |
| 2,32,4096 | 0.514 | 0.566 | 535 | 486 | **0.91** | 2.7e-3 |
| 1,32,8192 | 0.938 | 1.044 | 586 | 527 | **0.90** | 2.7e-3 |

Net: **0.88× → ~0.91× FA3**, bit-exact, no CUTLASS, **intra-WG QK∥PV overlap async** (the thing §18
wrongly declared unreachable). Production build = **`-DWST -DCUTE`**. The last ~9% is fill/drain + causal
load-balance (a dynamic persistent scheduler), not wgmma serialization.

