# Blackwell (SM100) FA4 gap analysis — 2026-08 pass

State after this pass: **154/154 runnable scenarios numerically correct**, geomean **1.480×** vs the
upstream Dao-AILab FA4 CuteDSL oracle, min **0.776×**, **94/118** paired scenarios at or above parity.
(Start of pass: geomean 1.425×, min 0.593×, 86/118.)

On the `real` subset (all `page_size=1` paged, the set this pass targeted):

| bar | before | after |
|---|---|---|
| Decode (n=5) | 0.877× | **1.184×** |
| Prefill (n=3) | 0.618× | **0.975×** |
| both (n=8)   | 0.749× | **1.101×** |

Per scenario, final 4-GPU sweep: `dec_s512` 1.500, `dec_s4k` 1.169, `dec_s16k` 1.012,
`dec_b64_s4k` 0.876, `qwen3moe_dec_s512` 1.494, `pf_s2k` 1.040, `pf_s16k` 0.906,
`qwen3moe_pf_s4k` 0.984. Only `dec_b64_s4k` and `pf_s16k` are under parity; both are analysed under
"Other open items".

**Read the measurement note at the bottom before quoting any of these.** The 4-GPU sweep has a wide
spread on this set: in dedicated single-GPU `bench.microbenchmarks.ab` runs the same three prefill
scenarios measure 1.034 / 1.060 / 1.041 and `dec_b64_s4k` 0.904. One sweep also produced a single
`qwen3moe_pf_s4k` = 0.311 sample that did **not** reproduce in six subsequent measurements (0.984-1.060)
and has no identified mechanism — treat isolated 4-GPU outliers on this set with suspicion.
The one config at the low end of the tree, `mixed_1pf8k_64dec4k` (0.776 in the sweep, 0.803 clean
against 0.828 before), is a ~3% drift inside its historical noise band (0.776-0.888) and is not in the
targeted subset.

Regenerate with `python -m bench.eval --target blackwell --set all`; compare two snapshots with
`python -m bench.microbenchmarks.cmp <old-dir> docs`.

## What closed the `page_size=1` family

The previous pass concluded that this family was starved of K/V ring slots and that the only way
forward was freeing ~64KB of shared memory. **That was the wrong diagnosis.** Ring depth is worth
almost nothing here (an isolated loader benchmark puts depth 2→3 at 2.4% and depth 3→4→6 at 0%);
the limiter was the *number and size of memory transactions the producer issues*, and the cost of
issuing them at all.

1. **De-page + dense route for large-q paged prefill** (`depage_kv`, `depage_cuk`, `depage_pool`).
   A prefill re-reads every KV row ~`rep*seqlen_q/(2*WTM)` times — 256× on `attn_real_llama8b_pf_s16k`
   — so the per-tile page walk was paid 256 times. Materializing the cache once into the contiguous
   `[total,hkv,hd]` layout costs 2 passes over the *unique* bytes (<1% of the attention's own KV
   traffic) and hands the shape to the fully TMA-fed persistent `fwd_kernel_fa4`.
   `pf_s2k` 0.707→0.994, `pf_s16k` 0.570→0.898, `qwen3moe_pf_s4k` 0.593→0.943 at the time; item 7 then
   took all three past parity. Keeping the paged route instead (`FA4_NODEPAGE=1`) still measures
   0.700/0.566/0.584 today, so the routing decision holds independently of the other fixes.
   Gated to `page_size % 8 != 0`, i.e. only caches the page-aligned TMA box cannot express. The first
   version was not, and hijacking the already-fast page-TMA shapes cost `mistral_pf_q4k` 1.03→**0.17**,
   `llama8b_pf_q8k` 0.92→0.64, `llama8b_ext_q1k_k64k` 0.93→0.52. Fully general over page order and
   ragged seqlens (it is a real gather, not a contiguity assumption); scratch is pooled across calls
   because a per-call `at::empty` is a ~100µs hazard on a 125µs kernel when the pool is fragmented.

2. **Warp-uniform `tile::gather4`** (`gather4_kv_warp`). `cp.async.bulk.tensor` takes its operands in
   UNIFORM registers, so a gather4 issued from divergent lanes makes ptxas emit an
   `ELECT` / `R2UR.BROADCAST` / `BRA.U.ANY` replay loop that runs **once per active lane**. The old
   `op = lid` mapping therefore put 32 serial issues on 4 of the 11 load warps and left 7 idle:
   `UTMALDG` `inst_executed` was 1048576 = 32× the warp-parallel count, ~165 cycles per issue, and the
   gather4 issue region alone was **29.5% of every instruction the kernel executed**. Now each load
   warp owns a contiguous slice of the `WTN/4` key-groups and its lane 0 issues, staging its own page
   rows with `__syncwarp()` — which also deletes the per-tile CTA-wide `bar.sync` that was the single
   hottest PC (32.6% of stall samples, `barrier` 8.42 cycles/issue against FA4's 2.61).

3. **Run-detecting full-width TMA boxes** (`tma_full_kv_warp`, `s_contig`). The decisive measurement is
   a controlled pair on *identical* shapes: `llama8b_dec_b16_s16k` (`page_size=64`, 8 multi-row boxes
   per tile) ran **1.03×** while `attn_real_llama8b_dec_s16k` (`page_size=1`, 128 gather4 per tile) ran
   **0.64×** — same bytes, same tile, same ring, zero over-fetch on both sides (1.0001× of useful DRAM
   bytes). A `page_size=1` cache cannot be expressed as a multi-row box at *launch* time, but a tile
   whose `WTN` physical rows happen to be consecutive can use one at *run* time. Each CTA scans its own
   page-table range once (CTA-uniform result, so no per-tile decision and no extra rendezvous) and
   full tiles then cost `KCB+VCB` = 4 boxes of 16KB instead of 128 gather4 of 512B. A fragmented page
   table keeps the gather4 producer, unchanged. **NB this upgrade's benefit is conditional on the
   input:** `bench/scenarios.make_inputs` builds page tables with `arange`, so every `real` scenario
   satisfies it. FA4 has no such fast path (verified in `flash_fwd_sm100.py`/`paged_kv.py`). With the
   upgrade disabled (`FA4_NOPTRUN=1`) `dec_s16k` is **0.771** and `dec_b64_s4k` **0.783**, against 1.013
   and 0.904 with it — those are the numbers to quote for a genuinely fragmented cache. (Both are well
   above the 0.635 this upgrade was first measured against, because item 7 lifted the gather4 path too.)

4. **Parking the surplus load warps.** A contiguous tile needs 4 issues, so the other 7 load warps were
   spinning on `kv_empty`/`kv_full` and stealing issue slots from the softmax. Sizing `kv_full`'s
   arrive count to the participating warps (`NWU`) and returning early was worth **0.779→0.955** on
   `dec_s16k` — and it is free, because when the page table breaks all 11 warps participate again.
   This, not register spills, was the reason a 256-thread CTA used to look better: with the surplus
   warps parked, 512 threads now beats 256 (0.955 vs 0.924).

5. **Split Q staging.** A GQA-packed decode tile owns `rep` rows (4 or 8) of `WTM=128`, so the Q loop
   ran 32× its useful work through the per-element (div, mod, `koff`, predicate, store) path. Now the
   owned rows are gathered and the rest bulk-zeroed with `uint4` stores (`koff` keeps each row inside
   contiguous 64-element blocks). The unowned rows must still be **zeroed, not left as garbage**: a
   garbage row's row-max makes its `corr != 1`, which defeats the correction WG's warp-ballot skip of
   the 128-f32 tmem O rescale and costs far more than the fill saves (measured 0.955→0.874).

6. **`cp.async.mbarrier.arrive.noinc`** for the software-gather producer. It previously ran
   `cp.async.commit_group; cp.async.wait_group 0` *inside* the producer loop, which pinned the ring at
   ONE tile in flight no matter how deep it was — that, not the instruction count, is why the cp.async
   path used to lose to gather4. Fixed, it is still slower than gather4 here (0.466 vs 0.635), which
   the isolated benchmark confirms is inherent: 5.2 TB/s for cp.async vs 6.0 TB/s for warp-uniform
   gather4 at equal issuing warps.

7. **Ring state out of local memory in the MMA and Load warps** (`rsel`/`rflip`). This turned out to be
   the largest single win of the pass and it applies to *every* warp-specialised kernel here, not just
   the paged ones. `KF[s] VF[s] KVE[s] kd0a[s] vd0a[s] sK[s] sV[s] pkf[s] pvf[s] pe[s]` are all indexed
   by the runtime ring slot `s = i % KVS` inside the per-block loop, so ptxas puts the whole table in
   **local memory** — exactly the pathology documented at the top of this file and fixed for the
   correction WG's C-channel in the previous pass, but never applied to the MMA/Load warps. ncu on
   `attn_dense_hd128_seq16k`: **865,568 local load requests against FA4's 16,384**, with an `LDL.64` of
   the MMA descriptor immediately ahead of every `UTCHMMA` and an `LDL` of the barrier address+phase
   ahead of every `mbar_wait`. ptxas reports **zero spills**, which is why this hid for so long: these
   are not spills, they are runtime-indexed arrays. Applied to `fwd_kernel_fa4`, `fwd_kernel_fa4dk`,
   `fwd_kernel_fa4_paged` (descriptors only; its barriers were already done) and `fwd_kernel_ws_paged`
   (which also rebuilt `make_desc(sK[s])` on *every* block — now hoisted):

   | | before | after |
   |---|---|---|
   | `attn_dense_hd128_seq16k` | 0.843 | **0.884** |
   | `dense_c_d128_s8k` | 0.855 | **0.923** |
   | `attn_real_llama8b_pf_s16k` | 0.929 | **1.034** (ab) / 0.943 (eval) |
   | `attn_real_qwen3moe_pf_s4k` | 0.943 | **1.060** (ab) / 0.991 (eval) |
   | `attn_real_llama8b_dec_s16k` | 0.978 | **1.013** |
   | `attn_real_llama8b_dec_s4k` | 1.076 | **1.165** |
   | `attn_real_llama8b_dec_b64_s4k` | 0.816 | **0.908** |
   | `llama8b_pf_q8k` | 0.895 | **0.960** |
   | `attn_varlen_hd192_dv128` | 1.36 | **1.469** |

   Measured secondary effects (ncu, seq16k): local ld requests 865,568 → 2,368; SASS instructions
   503.9M → 433.6M; `sm__cycles_active` 1,565K → 1,474K; TMEM pipe 62.7% → 66.2%; `long_scoreboard`
   5.70 → 5.24 cycles/issue. **`fwd_kernel_ws2_cont` still has the pathology** (`vd0a[s]`,
   `KVE[s]`, `sK[s]`, `pe[s]`) and is the obvious next target if that route is ever re-enabled.

### Where the loaders actually top out (isolated, `/tmp/gbench.cu`, 128 CTAs, L2-flushed)

| loader | GB/s (arange) | GB/s (random pages) |
|---|---|---|
| contiguous TMA box `[64,128]`, 4 desc/tile | **6326** | n/a |
| contiguous TMA box `[64,4]`, 128 desc/tile | 6233 | n/a |
| **gather4 warp-uniform, 15 load warps** | **5987** | **5781** |
| gather4 warp-uniform, 11 load warps | 5851 | 5649 |
| gather4 divergent lanes (the old code) | 4811 | 4813 |
| gather4 warp-uniform, 4 load warps | 4491 | 4471 |
| `cp.async.cg` 16B, 352 threads | 5167 | 4845 |
| `cp.async.bulk` non-tensor, 256B slabs | 2807 | 2809 |
| `LDG.128` → registers, no smem | 4026 | 3924 |
| contiguous `uint4` stream at 128 CTAs (reference) | 5198 | — |
| contiguous `uint4` stream at 1184 CTAs (roof) | 6663 | — |

Two conclusions. **gather4 is not rate-limited by the TMA engine** — a *contiguous* box of the
identical 512B size and identical 128 issues/tile is only 3.8% faster at 15 warps, and the
descriptor-size curve is flat from 512B to 16KB. What it is limited by is **issuing warps**
(4→8→11→15 gives 4491→5759→5851→5987) against a ~19KB-in-flight-per-SM ceiling. And a random page
table costs only 3.4%, so the harness's identity page table is not hiding a bandwidth cliff — it is
hiding a *transaction-count* win, which is why item 3 above matters in the kernel but only 3.8% in
isolation.

## Are we running FA4's pipeline?

Yes for the core — `fwd_kernel_fa4` is a faithful replica of `FlashAttentionForwardSm100`:
persistent CTA, 256-row Q tile as two stacked 128-row halves, two softmax warpgroups with an
ordered ping-pong, a correction warpgroup, one MMA warp driving `tcgen05` into tmem, TMA-Q, and an
mbarrier K/V ring. Five differences remain, in rough order of how much they still cost us:

| # | FA4 | ours | where it bites |
|---|---|---|---|
| 1 | `kv_stage = (224KB − smem_q_o) / max(K,V)_per_stage` → **5 tile slots** at hd128 (2.5 K/V pairs); K and V are separate pipeline stages in one ring | `FA4KVS = 2` → 4 slots (2 pairs). A 3rd full pair needs 256KB > the 227KB cap | latency-bound paged prefill and dense long context. The decoupled ring already exists as `fwd_kernel_fa4dk<KVS,VST>` (used at hd192 with 2/1); hd128 wants K-depth 3 / V-depth 2 = 160KB + 64KB Q = 224KB, which fits |
| 2 | a fraction of `ex2` is evaluated by a packed-f32x2 degree-3 polynomial on the FMA pipe (`ex2_emu_freq`, `e2e_asm2`) | all `ex2.approx` on the XU pipe | dense long context. NCU on `dense_c_d128_s8k`: **XU 56.3%** of peak vs tensor **55.2%** and FMA **9.8%** — the softmax's transcendentals are co-limiting with the MMAs. `tools/ex2_emu.py` implements the patch; it compiles but **hangs** (see below) |
| 3 | separate **epilogue warp** (w13); load on w14, w15 idle | correction WG also does the epilogue; load w13, Q-producer + scheduler w14–15 | frees a warp for a wider *and* persistent load warpgroup, which is what page_size=1 needs |
| 4 | paged KV: TMA when `page_size == tile_n`, else a software cp.async gather with page rows in **registers** + warp-shuffle broadcast; load warps widened to 2 (q_stage=2) or 4 (q_stage=1) | page-aligned TMA boxes when `page_size % 8 == 0`, else `tile::gather4`; page rows staged through smem | page_size=1 (the `real` set) |
| 5 | `LdRed32x32bOp` (`tcgen05.ld.red`) returns each 32-column tile's max in an extra register | software `max` reduction over the loaded fragment | minor: instruction count in the softmax |

## What landed this pass

1. **Page-aligned TMA paged-KV producer.** A WTN=128-key tile spans WTN/segrows *contiguous* physical
   page segments whenever `page_size % 8 == 0` (the 128B-swizzle atom is 8 rows), so K+V becomes
   `nseg*(HD/64 + HDV/64)` `cp.async.bulk.tensor.2d` boxes — **8 at hd128/page 64, against ~4096
   `cp.async` issues**. Instructions per launch dropped 138.6M → 42.3M.
2. **256-thread paged CTA.** The page-TMA producer needs ~8 issues per tile, so eleven load warps are
   pointless — and `__launch_bounds__(512,1)` caps ptxas at 128 regs/thread, which the 1-pass
   softmax's `f[128]` overflows. At 256 threads ptxas gets 256 regs and no `setmaxnreg` juggling.
3. **K/V ring depth 3** when a CTA has ≥24 K blocks (below that the extra ring fill costs more).
4. **Degenerate sliding-window normalization.** `window_left == max_ctx` + `window_right == 0` under
   causal *is* plain causal, but every fast route gates on `win_l<0 && win_r<0`. `mistral_pf_q4k`
   0.276 → 1.03.
5. **Ragged empty-tile skip.** The persistent scheduler handed out the rectangle `gx*hq*nB` sized by
   the longest sequence; `varlen_c_d128_realmix` ran 3072 tiles for 1088 tiles of real work.
6. **Mixed-batch split by query length.** One launch has to be sized by the longest sequence, forcing
   `pack=false`, so decode sequences read K/V once per *query* head instead of per KV head (4× the
   traffic at rep=4). Now two launches with a CTA-uniform ownership predicate — no D2H, no metadata.
7. **Paged prefill on the persistent kernel** via a `PAGED` template parameter on `fwd_kernel_fa4`
   (page descriptors + `seqused_k` + page-table base row; everything else shared with dense).
8. **Fold the Q scale into the softmax.** Pre-scaling Q rounded `q*scale*log2e` back to bf16, adding
   ~2^-9 relative error per element before the QK dot product — that alone was both remaining FAILs.
9. **De-spill the ring state.** `a[s]` with a runtime `s` puts the array in *local* memory; every
   pipelined kernel indexed its ring state that way inside the per-block loop. `rsel`/`rflip` keep it
   in registers: local-memory load sectors on `attn_real_llama8b_pf_s16k` **38.6M → 0.72M**.

## RESOLVED: page_size=1 (`attn_real_*`) — superseded, kept for the falsified reasoning

The analysis in this section (PC sampling showing ~61% mbarrier spins, the conclusion that the paged
producer "has run out of ring slots", and the ranked next step "free ~64KB of shared memory so the
ring can hold a third K/V pair") **was wrong about the cause**. The spins were real, but the producer
was not waiting for slots — it was waiting for a lane-serialized `UTMALDG` replay loop and a per-tile
CTA-wide `bar.sync` on a dependent page-table load. See "What closed the `page_size=1` family" at the
top. The ring-depth experiments (K3/V1 vs K2/V2, the 3/2 plumbing that does not fit) remain valid
*measurements* and are still in the tree behind `FA4_PGKD`; they simply attacked the wrong limiter.
An isolated loader benchmark now puts the marginal value of ring depth beyond 3 slots at 0%.

## FALSIFIED: `ex2` emulation is not worth doing on this kernel

The claim in the table above — that a fraction of `ex2` on the XU pipe co-limits the dense kernel, and
that FA4's packed-f32x2 degree-3 polynomial (`ex2_emu_freq=16`, 12.5% of pairs) is the fix — **does not
hold.** Direct test: mechanically replace *all 1024* per-element `ex2.approx.ftz.f32` in the generated
inline-PTX softmax with an FMA-pipe op, i.e. remove 100% of the transcendental work:

```
perl -pi -e 's/ex2\.approx\.ftz\.f32 (f\d+), (d\d+);/mul.f32 $1, $2, $2;/g' kernels/fa4_blackwell.cu
```

Result: `dense_c_d128_s8k` 0.841 → **0.855**, `attn_dense_hd128_seq16k` and
`attn_real_llama8b_pf_s16k` did not move. Removing the entire XU load is worth ~1.7%, so emulating
12.5% of it is worth ~0.2%. `tools/ex2_emu.py` and the reserved PTX registers (`mgc`, `pc0..pc3`,
`e7..e10`, `ei0/ei1`, `ef0/ef1`) are dead weight; do not spend time on the hang.

For the record, FA4's exact settings for bf16/hd128/causal are `ex2_emu_freq=16`, `ex2_emu_res=4`,
`ex2_emu_start_frg=1` → emulation only for `k in {12,14,28,30}` and fragments `j in {1,2}`, i.e. 8 of
64 pairs = 12.5% of elements; `POLY_EX2[3] = (1.0, 0.695146143436, 0.227564394474, 0.077119089663)`
evaluated by Horner in `fma.rn.ftz.f32x2`, with `add.rm` against `2^23+2^22` for the integer split and
a `shl.b32 23` + `add.s32` splice for the exponent.

## FALSIFIED: FA4's `split_P_arrive = 96` does not help here

Implemented faithfully in `fwd_kernel_fa4`: a `p_part` mbarrier, a mid-block
`tcgen05.wait::st` + arrive inserted into the generated softmax PTX after the 3rd of 4
`tcgen05.st...x16` P stores (P words 0..47 == key columns 0..95), and the MMA warp split into
`pv_k(...,0,PVA)` / `pv_k(...,PVA,NKT_PV)` with `PVA = 6` of 8 k-steps. Numerically identical
(`err` unchanged), and a net **loss**: `attn_dense_hd128_seq16k` 0.850→0.820,
`attn_real_llama8b_pf_s16k` 0.917→0.889, `dense_c_d128_s8k` flat. The 128 extra arrivals per half per
block plus the split store batch cost more than the overlap buys — the same conclusion the `ws2` path
comment already records for its own CORR-only `split_P`. Reverted; not kept behind a flag.

Two other FA4 tricks from the same distillation were checked and found **already present**: the
`rescale_threshold = 8.0` row-max hysteresis (generated PTX: `setp.gt.f32 pth, jmp, 0f41000000` +
`selp.f32 mm, cand, %3, pth`) and the ballot-gated O-rescale skip
(`if(!__all_sync(0xffffffffu, sc0==1.0f)) tmem_o_rescale<HDV>(...)`).

## Other open items

* **Dense long context** `dense_c_d128_s8k` 0.923, `attn_dense_hd128_seq16k` 0.884,
  `dense_nc_d128_s8k` 0.871, `attn_dense_hd128_seq8k` 0.911 — and this ceiling also caps the de-paged
  prefill (`attn_real_llama8b_pf_s16k` 0.943, which routes here). **Fully characterised by ncu now**, so
  do not re-diagnose it:
  - **The absolute tensor work is IDENTICAL to FA4's.** `sm__ops_path_tensor_op_utchmma_src_bf16_dst_fp32`
    = 920,800 cycles/SM for both, `sm__mem_tensor_cycles_active` = 981K for both (0.006% apart), and
    `UTCHMMA` / `LDTM.x32` / `STTM.x16` / `F2FP…PACK_AB` per 128-key block are 32.0 / 32.8 / 32.0 / 519.8
    for us against 32.0 / 32.0 / 32.5 / 519.9 for FA4. We simply spread it over 33% more cycles
    (TMEM pipe 62.7% vs 83.2%). Every cycle of the gap is **idle tensor pipe** — not memory (DRAM 5.7%
    vs FA4's 2.9%), not L2, not occupancy, not registers, not instruction algebra.
  - Ruled out quantitatively: **XU** (our XU demand is 2083 cycles/block, *below* the 2181 TMEM-pipe
    floor, so deleting 100% of the transcendentals can win at most 1.6% — matching the measured +1.7%);
    **split_P** (implemented, measured, a loss); **tmem read latency** (`short_scoreboard` 0.41 vs 0.37);
    **P conversion / O store / epilogue** (identical counters); **scheduler imbalance** (max/avg
    `sm__cycles_active` 1.044 at 16k, better than FA4's 1.093); **2-SM clusters** (FA4 runs
    `launch__cluster_dim_x = 0` here and issues the same 32 `UTCHMMA` per block — parity does *not*
    need clusters).
  - What remains: `long_scoreboard` 5.24 vs FA4's 2.23 cycles/issue, and **twice the mbarrier hops** —
    47.7 waits per block against FA4's 24.5, with 37.6% of our SASS being spin against FA4's 18.3%
    (all 4 warps of each warpgroup independently spin on the same barrier; 40 of our 47 waits are that).
    Structurally, tmem is 512/512 columns (S0@0 S1@128 O0@256 O1@384, P aliased on S), so each half's
    softmax window is exactly 16 MMA = 1024 cycles while its measured non-wait work is ~2039 — a 2×
    deficit that *is* the 41% idle tensor pipe. The two candidate fixes, in order: (a) have only the
    warpgroup-lead warp spin and fan out with a named barrier (~30 of 47.7 waits, ~1.5%); (b) narrow the
    K tile to `WTN=64` so S0/S1 need 64 columns each, freeing 128 tmem columns for a genuinely
    double-buffered P and doubling the ping-pong stages per 128 keys at identical MMA/TMEM instruction
    counts. (b) is the one sized to close the remaining gap. NB FA4 buys its win partly with **clock**:
    at the same 992 W cap it runs 1477 MHz against our 1620 MHz, so the *cycle* gap is 1.31× where the
    wall gap is 1.19× — cycle parity is wall parity, and beating it likely does need `cta_group::2`.
* **`attn_real_llama8b_dec_b64_s4k` 0.908** — the only `real` decode scenario still under parity, and
  the cause is scheduling, not the loader. 64 batches × 8 kv-heads = 512 CTAs on 148 SMs = **3.46
  waves**, and each CTA pays a full cold start (prologue, tmem alloc, 3 `__syncthreads`, ring fill)
  amortized over only 32 K-blocks. Splitting K to smooth the waves makes it *worse* (splits 1/2/4 =
  0.831/0.770/0.608 with ring depth 3 forced) because it multiplies the cold starts. FA4 runs a
  persistent 148-CTA grid here (ncu: waves 1.00) and overlaps item *i*'s epilogue with item *i+1*'s
  K/V prefetch. The fix is a persistent paged-decode kernel with cross-item overlap — the same
  structure `fwd_kernel_fa4` already has for dense. Note the wave quantization itself is irreducible
  (512 indivisible items on 148 SMs is 86.5% however you schedule them); what persistence buys is the
  cold start, worth the remaining ~14%.
* **hd96** `dense_c_d96_s2k` 0.789 — runs the padded-128 `PADQ` path, so it does 128-wide QK/PV for
  96 useful columns. Needs native-96 tiles.
* **Register plan.** `2*SM + CORR + PROD = 512` at 512 threads. The generated inline-PTX softmax peaks
  near 140 live registers, and sweeping SM downward makes *it* spill (176 → 113 static spill sites vs
  62 at 192), so 192/64/64 is a local optimum; `tools/regplan.sh <SM> <CORR> <PROD>` compiles a plan
  and reports per-kernel spill counts. NB on the paged-decode kernel the 7.55M local-load sectors this
  causes are **not** the limiter — parking the surplus load warps beat removing the spills (item 4).
* **FA4 facts worth keeping** (from a full read of `flash_attn/cute/`): FA4 uses **no** `tile::gather4`
  and no `tcgen05.cp` anywhere; its paged `page_size != tile_n` path is 4096 `cp.async.cg` 16B copies
  per (K,V) pair with page rows in registers broadcast by `shfl.sync` (width 8, 2 shuffles per i64),
  no smem staging and no next-block page-table prefetch (a genuine gap to exploit). Decode
  (`q_stage=1`) repurposes softmax1 warps 4-7 as the load warpgroup (128 load threads) and leaves
  **104 registers/thread of the CTA budget unused** (184+80+64+80 = 408 of 512). 2SM MMA is off for
  causal; CLC is off unless `FA_CLC=1` and never for non-TMA paged. `num_splits` for
  batch16/hkv8/q=1 on 148 SMs is **1**, so FA4 does not split our decode scenarios either.

## Measurement note

Scenarios are scheduled across all four GPUs, so which device a short kernel lands on moves its ratio
run to run. Repeating four small configs in one process on one GPU gives ±1.5%; the same configs vary
±5% across sweeps. Trust absolute µs from `bench.microbenchmarks.ab` (single GPU, one process) for
anything under ~10%, and treat scattered few-percent "regressions" in a `cmp` diff as sampling.
