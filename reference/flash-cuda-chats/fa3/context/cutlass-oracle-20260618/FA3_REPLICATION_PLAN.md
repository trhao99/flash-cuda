# FA3 → self-contained `.cu` replication plan (hd128 bf16 prefill)

Goal: a single CUTLASS-free `.cu` that reproduces FA3's forward output (ideally bit-exact) and
matches/beats its performance. CUTLASS/CuTe is FA3's *only* third-party compute dependency, but it
is the substrate of the whole kernel — not a swappable GEMM call — so this is a from-scratch rewrite
informed by the oracle (the wgmma/descriptor/epilogue recipes verified in `ptx_gemm.cu`).

## 1. What FA3 does (forward, one warp-specialized persistent kernel; 384 threads = 1 producer WG + 2 consumer/MMA WGs)

Per output tile `(batch, head, query-block m)`:
1. **Schedule & setup** — persistent tile scheduler assigns the work tile; compute varlen/seqlen
   offsets and the n-block range (causal/local masking prunes blocks); init mbarriers; prefetch TMA
   descriptors.
2. **Producer WG** (`setmaxnreg.dec`) — TMA-loads Q once and K,V per n-block into swizzled smem
   (`cp.async.bulk.tensor`, or `cp.async` for paged-KV), signaling mbarriers w/ expected bytes
   (double-buffered, 2 stages).
3. **Consumer WGs** (`setmaxnreg.inc`), loop n-blocks:
   - **GEMM0 `S = Q·Kᵀ`** — wgmma SS, fp32 accumulate (m64n128k16 ×8).
   - **Online softmax (log2)** — row-max, `S = exp2(S·scale − max)` via `ex2.approx.f32`, row-sum,
     rescale running `O` by `exp2(old_max − new_max)`, update running max/sum; apply mask.
   - **S→P** cast fp32→bf16 into registers.
   - **GEMM1 `O += P·V`** — wgmma RS (P in regs, V transposed in smem), fp32 accumulate.
   - intra-WG overlap: GEMM0(n) ∥ GEMM1(n−1); release K/V buffers to producer via mbarrier.
4. **Epilogue** — `O /= rowsum`; `LSE = max + log(rowsum)`; write O (regs→smem `stmatrix`→TMA/STG)
   and `softmax_lse`.
5. **Combine kernel** — only for split-KV (`num_splits>1`): reduce partial O/LSE across splits.

## 2. CUTLASS functionalities to replicate (→ raw CUDA/PTX), with status

### A. GEMM / tensor cores
| CUTLASS/CuTe | Raw replacement | Status |
|---|---|---|
| `GMMA::ss_op_selector`+`cute::gemm` (QK) | `wgmma.mma_async.m64n128k16.f32.bf16.bf16` SS + hand smem descriptor | ✅ `ptx_gemm.cu` QK |
| `GMMA::rs_op_selector`+`cute::gemm` (PV) | same wgmma, **A operand in 4×b32 registers (K-major)**, V^T in smem | ⏳ in progress |
| `warpgroup_fence/arrive/commit/wait` | `wgmma.fence`/`commit_group`/`wait_group N` | ✅ |
| acc fp32 fragment ↔ thread map | hand-derived index math (rows `{16w+l/4,+8}`, cols `8c+2(l%4)+{0,1}`) | ✅ QK; reuse PV |
| RS A-fragment packing | 4×b32: `{(r0,k0),(r0,k0+1)} {(r0+8,k0),…} {(r0,k0+8),…} {(r0+8,k0+8),…}`, k0=2·(l%4) | ⏳ |

### B. Memory movement
| CUTLASS/CuTe | Raw replacement | Status |
|---|---|---|
| `SM90_TMA_LOAD`, `make_tma_copy_*`, `cute::copy(tma)` | `cp.async.bulk.tensor.{2..5}d…mbarrier::complete_tx::bytes` | ⛔ |
| host TMA descriptor build | `cuTensorMapEncodeTiled` (CUDA driver — allowed) | ⛔ |
| `prefetch_tma_descriptor` | `prefetch.tensormap` | ⛔ |
| paged-KV `cp.async` (`PagedKVManager`) | `cp.async.cg.shared.global` + page-table gather | ⛔ |
| smem swizzle `Swizzle<3,4,3>`/`ss_smem_selector` | K-SW128 XOR swizzle (or current no-swizzle layout) | ⚠ no-swizzle works; need true 128B swizzle for TMA |
| `Copy_Atom<SM90_U32x4_STSM_N>` (O) | `stmatrix.x4.m8n8.b16` (or vectorized STG) | ⚠ STG done; stmatrix optional |
| `SM75_U16x8_LDSM_T` (V transpose) | `ldmatrix.x4.trans.m8n8.b16` (or transposed global load) | ⏳ PV |

### C. Synchronization / scheduling
| CUTLASS/CuTe | Raw replacement | Status |
|---|---|---|
| `PipelineTmaAsync` (2-stage K/V) | `mbarrier.init/arrive.expect_tx/try_wait` + phase tracking | ⛔ |
| `NamedBarrier` (producer/consumer) | `barrier.sync N, cnt` / `bar.arrive` | ⛔ |
| `warpgroup_reg_alloc/dealloc` | `setmaxnreg.inc/dec.sync.aligned.u32` | ⛔ |
| `VarlenDynamicPersistentTileScheduler` | persistent grid + work-tile index math | ⚠ basic linear scheduler done; need varlen/causal-LPT |
| `grid_dependency_control` (GDC/PDL) | `griddepcontrol.*` | optional |

### D. Math / numerics — governs *exact* output
| CUTLASS/CuTe | Raw replacement | Status |
|---|---|---|
| `flash::Softmax` (online, log2) | hand row-max/sum + rescale, **`ex2.approx.f32`** (NOT `expf`) | ⛔ |
| `NumericArrayConverter` fp32→bf16 (S→P) | `cvt.rn.bf16.f32` (round-nearest-even) | ⛔ |
| `flash::Mask` (causal/local/seqlen) | predicated index compares | ⛔ |
| `cutlass::fast_tanh` (softcap) | `tanh.approx.f32` | optional |
| `cutlass::FastDivmod` | magic-number divmod | ⛔ |
| `LSE = max + log(sum)` | `lg2`/scale | ⛔ |

### E. Launch / glue
`device_kernel`/`kernel_launch` → plain `<<<>>>`; torch op (37-arg `fwd` schema; already in
fa3agent `seed_kernel.cu`); split-KV combine kernel.

## 3. Bit-exactness requirements
fp32 wgmma accumulation is deterministic and already matches. To match FA3 *to the bit* also requires:
1. **exp2 via `ex2.approx.f32`** (the MUFU.EX2 instruction), not `expf`/`exp2f` library calls;
2. the **exact op order** of the online-softmax rescale (fp32 add is non-associative);
3. **round-to-nearest-even** fp32→bf16 for S→P and the O write (`cvt.rn.bf16.f32`).
Miss any of these ⇒ "correct within tolerance" (fa3agent's 2×-fp32-ref rule) but not bit-exact.

## 4. Build order
1. **PV GEMM oracle** (RS) ← current. 2. TMA + mbarrier pipeline. 3. warp-specialization
(setmaxnreg/named barriers). 4. softmax/mask/LSE/epilogue numerics (group D — determines exact
output). 5. assemble fused forward for one config; A/B vs `_vllm_fa3_C.fwd` (correctness + latency).
