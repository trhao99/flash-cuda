export const meta = {
  name: 'parity-lever-hunt',
  description: 'Find concrete faithful optimizations for fa3_hopper.cu configs still below 0.95x vs FA3',
  phases: [
    { title: 'Hunt', detail: 'parallel deep-read FA3/cutlass vs our kernel per target cluster' },
    { title: 'Synthesize', detail: 'dedupe + cross-rank candidates by EV, flag already-tried' },
  ],
}

const KERNEL = '/home/yak/fa3-cuda/kernels/fa3_hopper.cu'
const FA3 = '/home/yak/fa3-cuda/reference/vllm-flash-attention/hopper/'
const CUTLASS = '/home/yak/fa3-cuda/reference/cutlass'
const MEM = '/home/yak/.claude/projects/-home-yak-fa3-cuda/memory/fa3-perf-gap-closing.md'

const CAND_SCHEMA = {
  type: 'object',
  properties: {
    candidates: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          target: { type: 'string', description: 'config(s) this helps, e.g. attn_real_llama8b_pf_s16k' },
          lever: { type: 'string', description: 'concise name of the optimization' },
          mechanism: { type: 'string', description: 'WHY it helps: the specific inefficiency in our code vs what FA3 does, with file:line for both' },
          expected_impact_pp: { type: 'number', description: 'rough expected +pp gain, honest' },
          faithful: { type: 'boolean', description: 'true if it matches FA3/cutlass behavior (not benchmark-gaming)' },
          already_tried: { type: 'boolean', description: 'true if the memory notes this exact lever was tested' },
          risk: { type: 'string', description: 'correctness/regression risk + which other configs it touches' },
          concrete_change: { type: 'string', description: 'the specific code edit to make (function, lines, what to change)' },
        },
        required: ['target', 'lever', 'mechanism', 'expected_impact_pp', 'faithful', 'already_tried', 'concrete_change'],
      },
    },
  },
  required: ['candidates'],
}

const COMMON = `Our kernel: ${KERNEL} (a cutlass-free CUDA FA3 for Hopper sm90). FA3 reference: ${FA3}. Cutlass: ${CUTLASS}.
CRITICAL CONTEXT — read ${MEM} FIRST: it documents ~8 sessions of what's been TRIED and FAILED (do NOT re-propose those). The winning pattern just found: FA3's PagedKVManager broadcasts the computed int64 K/V row OFFSET across the 8 cooperating gather lanes instead of having each recompute ent*stride (8x redundant int64 muls) — this closed paged decode + qwen prefill. Look for MORE inefficiencies of that KIND: redundant per-lane/per-thread computation, a faithful FA3 schedule/layout we don't match, wasted instructions FA3 avoids.
Do NOT propose: benchmark-gaming (TMA on the arange page_table), de-scatter, parking/nanosleep waits, asymmetric K/V rings, STAGES=2 for paged, 240/24 registers for hd256, per-tile hd256 (all TRIED+FAILED per memory).
Report ONLY concrete, faithful, NOT-already-tried candidates with exact file:line for both our code and FA3's, the mechanism, and the specific edit. Be honest about expected_impact_pp (small is fine). Read-only; do not modify files.`

phase('Hunt')
const [pf, hd256, denselong, gather] = await parallel([
  () => agent(`${COMMON}\n\nTARGET: attn_real_llama8b_pf_s2k (0.908x) and _s16k (0.925x) — paged bf16 hd128 causal PREFILL. It uses prefill_kernel (per-tile, NOT persist). Current ncu: tensor 67%, barrier 0.79 (FA3 74%/0.44). The COOP gather is already offset-broadcast-optimized. Deep-read FA3's hd128 paged prefill: mainloop_fwd_sm90_tma_gmma_ws.hpp (consumer mma loop + ping-pong warp_scheduler_barrier), paged_kv.h, flash_fwd_kernel_sm90.h. Compare to our prefill_kernel consumer mainloop (~lines 1310-1432) + COOP producer (~1216-1287). Find concrete levers to raise tensor 67->74 / cut barrier 0.79->0.44: redundant consumer work, softmax/wgmma interleave differences, ping-pong placement, producer issue-slot contention, any redundant int64/address math like the offset-broadcast. Rank by EV.`, { label: 'hunt:llama_pf', phase: 'Hunt', schema: CAND_SCHEMA }),

  () => agent(`${COMMON}\n\nTARGET: attn_dense_hd256_causal (0.878x) — DENSE bf16 hd256 causal PREFILL. It uses prefill_persist_kernel (persistent, TMA producer, NOT the COOP gather), CN=48/STAGES=3/CM=128. ncu: tensor 46%, acc_o[128] (128 fp32 regs/thread), 1 CTA/SM (occupancy-capped), barrier 2.01 (mostly harmless producer bar.sync). FA3 gets ~0.92 at CN=80/STAGES=2. Deep-read FA3's hd256 path: tile_size.h (kBlockN=80), mainloop_fwd_sm90 (RescaleOBeforeGemm, the 2-CTA cluster/TMA-multicast for non-causal, acc layout), flash_fwd_kernel_sm90.h. Compare to our prefill_persist_kernel (~lines 1450-1810) + hd256 dispatch (~line 2672) + launch (~2470-2504). Find concrete levers for the acc_o[128]/occupancy/46%-tensor problem that are NOT in the tried list (rescale-before/240regs/per-tile/CN80 all failed). Consider: FA3's exact acc tiling, any redundant compute in the big-head consumer, TMA producer efficiency, whether a different smem layout frees a stage. Rank by EV.`, { label: 'hunt:hd256', phase: 'Hunt', schema: CAND_SCHEMA }),

  () => agent(`${COMMON}\n\nTARGETS: dense bf16 hd128 LONG-CONTEXT still <0.96: attn_dense_hd128_nc4k (0.931), _large4k (0.923), _seq8k (0.941), _seq16k (0.953), _hd192_dv128 (0.937), _hd192_causal (0.968). These use prefill_persist_kernel (dense TMA persist, CM=128). Deep-read FA3's dense long-context mainloop (mainloop_fwd_sm90_tma_gmma_ws.hpp consumer + tile scheduler tile_scheduler.hpp) vs our prefill_persist_kernel (~lines 1450-1810). Find concrete faithful levers: TMA multicast / cluster, tile-scheduler order (LPT / work distribution), softmax/wgmma overlap, any redundant compute. These share the persist kernel with hd256 so a lever may help both. Rank by EV and note cross-config impact.`, { label: 'hunt:dense_long', phase: 'Hunt', schema: CAND_SCHEMA }),

  () => agent(`${COMMON}\n\nTARGET: the COOP cp.async gather micro-architecture (prefill_kernel producer ~lines 1216-1287, shared by paged prefill + decode). It was just improved via offset-broadcast. Hunt for FURTHER faithful gather efficiencies vs FA3 paged_kv.h: (1) K and V share layout on the COOP path (HD==HDV) so koff==voff when strides match — can we broadcast ONE offset (2 shfl) instead of two (4 shfl)? Check k_row/v_row equality safety. (2) Is koffR/voffR (smem swizzle offset) recomputed redundantly? (3) cp.async granularity/coalescing vs FA3's kGmemElemsPerLoad. (4) Does FA3 issue K then V in separate commit groups or interleaved, and does our commit/publish (PF_PUBLISH) granularity thin the issue? (5) page-table LDG efficiency. For each, give the exact edit + safety guard + expected pp. Rank by EV.`, { label: 'hunt:gather', phase: 'Hunt', schema: CAND_SCHEMA }),
])

phase('Synthesize')
const all = [pf, hd256, denselong, gather].filter(Boolean).flatMap(r => r.candidates || [])
log(`collected ${all.length} raw candidates`)
const synth = await agent(`You are ranking candidate optimizations for a cutlass-free FA3 CUDA kernel to reach >=0.95x parity on the configs still below it. Here are ${all.length} raw candidates found by 4 analysis agents (JSON):\n\n${JSON.stringify(all, null, 2)}\n\nDedupe overlapping candidates. DROP any with already_tried=true or faithful=false. Cross-rank the survivors by expected value = expected_impact_pp × tractability × (breadth of configs helped) ÷ risk. Prefer the "redundant-computation / faithful-schedule-match" kind that just worked (offset-broadcast). Return the TOP candidates (up to 10) ranked best-first, each with target, lever, mechanism (with file:line), concrete_change, expected_impact_pp, risk. Be honest — if few are promising, return few.`, { label: 'synthesize', phase: 'Synthesize', schema: CAND_SCHEMA })
return synth