export const meta = {
  name: 'fa3-paged-persist-recipe',
  description: 'Reverse-engineer FA3s exact 2-stage paged persist producer + Q buffering, cross-check my kernels smem, and emit a precise verified diff to un-starve my persist',
  phases: [
    { title: 'Research', detail: 'FA3 producer 2-stage schedule, Q/smem budget, my-kernel diff, smem math' },
    { title: 'Synthesize', detail: 'reconcile into the target config + a ready-to-apply, deadlock-checked diff' },
  ],
}

const REF = '/home/yak/fa3-cuda/reference/vllm-flash-attention/hopper'
const MYFILE = '/home/yak/fa3-cuda/kernels/fa3_hopper.cu'

phase('Research')

const A = agent(
  `Read FA3 (${REF}, files mainloop_fwd_sm90_tma_gmma_ws.hpp + pipeline/PipelineAsync). For the bf16, head_dim=128, causal, PAGED (page_size=1, PagedKVNonTMA=true, Use_TMA_KV=false, kStages=2) forward case, extract the PRODUCER (load warpgroup) schedule EXACTLY, with file:line citations and quoted code:
  1. How does the producer issue K and V cp.async loads across the 2 pipeline stages? Are K and V on SEPARATE pipelines/mbarriers (pipeline_k vs pipeline_v) with independent producer_commit, or one combined barrier? Quote the load loop.
  2. IntraWGOverlap at the LOAD level: is K for n_block loaded one step AHEAD of V (V lags by one n_block)? Quote the exact interleave (the load_K/load_V calls and their smem_pipe_write indices).
  3. At only 2 stages, how does the producer avoid starving the consumer given the cp.async gather latency at 1 CTA/SM? Does it issue BOTH stages' loads upfront in a prologue before the consumer starts? Is there any extra prefetch depth beyond 2? Quote the prologue load code.
  4. Exactly how many producer threads issue the cp.async (32 or 128)? How is the page table resolved per key (PagedKVManager: kGmemThreadsPerRow, __shfl broadcast) and is the page-table load decoupled/prefetched ahead of the K/V gather? Cite paged_kv.h.
  Give concrete quoted code, not paraphrase.`,
  { label: 'fa3-producer-2stage', phase: 'Research', agentType: 'Explore' }
)

const B = agent(
  `Read FA3 (${REF}, mainloop_fwd_sm90_tma_gmma_ws.hpp + flash_fwd_kernel_sm90.h + the SharedStorage struct + tile_size.h). For bf16 head_dim=128 causal PAGED persistent forward, answer with file:line citations + quoted code:
  1. Does FA3 DOUBLE-BUFFER Q across the persistent work-tile loop, or single-buffer it? How many Q smem buffers? Quote the Q smem allocation in SharedStorage.
  2. How is Q (re)loaded per work-tile in the persistent kernel — via TMA (one async box) or cp.async gather? Is the Q reload for the NEXT work tile overlapped with the CURRENT tile's mainloop, or does the work-tile boundary serialize on Q (consumer must free Q before producer reloads)? Quote the Q load + the barrier/pipeline that gates it.
  3. Compute the EXACT persistent-kernel dynamic smem budget in bytes for hd128 paged: Q buffer(s) + K stages + V stages + all mbarriers. Show the arithmetic (kBlockM=128, kBlockN=128, kStages=2, element 2 bytes, MmaPV_is_RS so no P smem). How much of the 228KB Hopper smem is used, and how much headroom remains?
  4. Does FA3 use a SEPARATE dedicated kernel/config for paged vs dense prefill, or the same kernel with runtime flags? (I share one persist kernel for dense TMA + paged COOP and it forces compromises — I want to know if FA3 specializes.)
  Concrete quoted code.`,
  { label: 'fa3-q-smem-budget', phase: 'Research', agentType: 'Explore' }
)

const C = agent(
  `Read my kernel ${MYFILE}. I need a PRECISE, ready-to-apply edit spec (exact before/after text blocks with enough surrounding context to be unique) to modify prefill_persist_kernel (the persistent paged prefill kernel, template <typename TIn, typename TC, int HD, int HDV, bool USE_TMA, bool EXTRA, int STAGES, int CN, int CM, int MINB, bool TMA_Q>) to implement "K-ahead-of-V separate barriers", matching how FA3 does IntraWGOverlap at the load level:
  1. Locate the barrier layout in prefill_persist_kernel: currently full[STAGES], empty[STAGES], barQf[QBUF], barQe[QBUF], schF[SDEPTH], schE[SDEPTH], schW[SDEPTH]. Give the exact bytes layout and how to add fullK[STAGES]+fullV[STAGES] replacing full[STAGES] (the launcher smem_p calc must grow by STAGES uint64_t).
  2. Locate the producer's COOP gather (load_kv_coop lambda) which currently loads K and V interleaved then PF_PUBLISH(&full[st]). Give the exact edit to: load all K first -> PF_PUBLISH(&fullK[st]); then all V -> PF_PUBLISH(&fullV[st]). Also the TMA load_kv lambda (non-paged dense path) must split its expect_tx/tma_box into fullK/fullV. And the init loop mbar_init(&full[s],...) -> fullK/fullV.
  3. Locate the consumer mainloop (the qk_pv_g loop and the prologue qk_gemm_g). Give the exact edit so QK waits fullK[st] and PV waits fullV[pst] (with the correct phases: fp=(gg/STAGES)&1 for fullK[st]; the PV uses fullV at stage pst=(gg-1)%STAGES with phase ((gg-1)/STAGES)&1). Handle the prologue (first tile, fullK only for QK, then fullV for the first PV) and the final pv_final_g (needs fullV of the last tile waited).
  Report the exact line numbers and before/after text for EVERY edit. Also report the launcher branch (launch_prefill_t paged-persist branch, currently gated && false with PPS constant) and the exact edit to (a) set PPS=2, (b) set QBUF=2 in prefill_persist_kernel + the launcher smem_p, (c) remove the && false. Be exhaustive and exact — I will apply these edits verbatim.`,
  { label: 'my-kernel-diff-spec', phase: 'Research', agentType: 'Explore' }
)

const D = agent(
  `Compute smem budgets for my kernel ${MYFILE} prefill_persist_kernel, hd128 (HD=HDV=128, CM=CN=128, HDsm=HDVsm=128), element 2 bytes. The dynamic smem is (QBUF*CM*HDsm + STAGES*CN*HDsm + STAGES*CN*HDVsm)*2 + barriers. Barriers currently: (2*STAGES + 2*QBUF + 2*SDEPTH) uint64 + SDEPTH int, with SDEPTH=2. If I split full->fullK+fullV, barriers become (3*STAGES + 2*QBUF + 2*SDEPTH) uint64 + SDEPTH int. Hopper max dynamic smem = 227KB (232448 bytes) on H200 (cudaFuncAttributeMaxDynamicSharedMemorySize). Tabulate total bytes for these configs and say which FIT under 232448 and how much headroom: (a) STAGES=2,QBUF=1; (b) STAGES=2,QBUF=2; (c) STAGES=3,QBUF=1; (d) STAGES=3,QBUF=2; (e) STAGES=2,QBUF=2 with fullK/fullV split; (f) STAGES=3,QBUF=1 with fullK/fullV split. Show the arithmetic for each.`,
  { label: 'smem-budget-math', phase: 'Research', agentType: 'Explore' }
)

const [fa3prod, fa3q, mydiff, smem] = await parallel([() => A, () => B, () => C, () => D])

phase('Synthesize')

const plan = await agent(
  `You are synthesizing an implementation plan to un-starve my hand-written FlashAttention-3 paged persistent prefill kernel so it reaches FA3 parity. Context that is ALREADY ESTABLISHED (do not re-derive): my dense-causal hd128 mainloop already BEATS FA3 (1.05x) with the identical schedule, so the mainloop is fine. The paged residual (0.84-0.89x per-tile) is the 128-thread cp.async COOP producer's per-light-causal-m-tile pipeline-fill prologue; FA3 amortizes it with a persistent kernel. My persist LOSES: STAGES=2 starves the ring (0.62), STAGES=3 fits but the QBUF=1 Q single-buffer serializes each work-tile boundary (0.80), a prefetched scheduler regressed (0.72). Measured persist bottleneck: ncu barrier stall 1.46 vs FA3 0.44.

Here are four research findings:

FA3 PRODUCER 2-STAGE SCHEDULE:
${fa3prod}

FA3 Q BUFFERING + SMEM BUDGET:
${fa3q}

MY KERNEL DIFF SPEC (K-ahead + STAGES/QBUF changes):
${mydiff}

MY SMEM BUDGET MATH:
${smem}

Produce a decisive plan:
1. What EXACT config does FA3 use (kStages, Q buffers, K/V separate barriers + K-ahead yes/no)? 
2. Given my smem budget, which of my candidate configs are FEASIBLE (fit <232448 bytes) and which best matches FA3? Rank them.
3. Recommend the SINGLE best variant to implement first, and a fallback. For it, is the "K-ahead-of-V separate fullK/fullV" the key un-starvation mechanism (letting the consumer start QK on K[i] while V[i] streams in, so a 2-stage ring behaves like a deeper one)? Explain WHY it should beat my STAGES=3/QBUF=1 persist.
4. Flag any DEADLOCK or correctness risk in the my-kernel-diff-spec (barrier init counts, phase parities, the empty[] release gating both fullK and fullV, the pv_final fullV wait). List concrete things to double-check.
Be decisive and concrete. Output the recommended variant, the exact config numbers, and the ordered risk checklist.`,
  { label: 'synthesize-plan', phase: 'Synthesize' }
)

return { plan }
