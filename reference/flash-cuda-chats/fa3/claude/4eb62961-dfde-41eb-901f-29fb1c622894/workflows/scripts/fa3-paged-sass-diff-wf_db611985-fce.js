export const meta = {
  name: 'fa3-paged-sass-diff',
  description: 'Dump FA3 paged-prefill SASS vs my prefill_kernel SASS + read cutlass cp.async pipeline; find the concrete instruction-level difference behind the 61->75% tensor-util gap and propose faithful fixes',
  phases: [
    { title: 'Extract', detail: 'FA3 SASS, my SASS, cutlass PipelineAsync' },
    { title: 'Synthesize', detail: 'concrete diff + faithful fix candidates' },
  ],
}

const FA3SO = '/home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so'
const MYSO = '/home/yak/fa3-cuda/.fa3_trials/nofence/nofence.so'
const CUTLASS = '/home/yak/fa3-cuda/reference/cutlass'
const REF = '/home/yak/fa3-cuda/reference/vllm-flash-attention/hopper'

phase('Extract')

const A = agent(
  `Use cuobjdump/nvdisasm on FA3's compiled kernel at ${FA3SO} to extract the SASS for the bf16 head_dim=128 causal PAGED (PagedKVNonTMA, page_size=1) forward kernel. The mangled name contains 'device_kernel' and 'FlashAttnFwdSm90' with template args including PagedKVNonTMA=1 (bf16, hd128). Steps:
  1. \`cuobjdump -sass ${FA3SO} | grep -c ...\` to list device_kernel symbols; find the hd128 bf16 paged one (there may be several; the paged-non-TMA one is what runs for page_size=1). Use \`cuobjdump -symbols\` or \`nvdisasm\` to find it. It may be in a separate cubin section.
  2. Dump that kernel's SASS to a temp file. Characterize the STEADY-STATE mainloop (the loop body between backward branches that contains HGMMA):
     - How many HGMMA (wgmma) per loop iteration, and their shapes (m64nNNk16)?
     - How are the cp.async loads issued (LDGSTS instructions) -- count per iteration, cache hint (.E / .BYPASS / .128 etc.), and whether they are interleaved with HGMMA or in a separate block.
     - The barrier/sync instructions in the loop: BAR.SYNC (named barriers, which IDs), BAR.ARV, and the mbarrier waits (how the consumer waits for K/V -- via what instruction). Count them per iteration.
     - The producer<->consumer and warpgroup-scheduler (ping-pong) barrier pattern.
     - The exp2 (MUFU.EX2) count per iteration and how they interleave with HGMMA.
  3. Report the per-iteration instruction MIX: #HGMMA, #LDGSTS, #MUFU.EX2, #BAR.*, #shuffle, #other, and the overall structure (issue order). Quote representative SASS snippets.
  Goal: identify what FA3's paged mainloop does per iteration that keeps its tensor pipe at ~75%. Be concrete with instruction counts + snippets.`,
  { label: 'fa3-sass', phase: 'Extract', agentType: 'Explore' }
)

const B = agent(
  `Use cuobjdump on MY kernel at ${MYSO}. Find the symbol matching \`prefill_kernel<__nv_bfloat16, __nv_bfloat16, 128, 128, 0, 0, 3, 128, 128, 1>\` (regex: prefill_kernelI13__nv_bfloat16S1_Li128ELi128ELb0ELb0ELi3ELi128ELi128ELi1E). Dump its SASS. This is the paged-prefill kernel (USE_TMA=0 -> the COOP cp.async gather path; but note the producer TMA and COOP paths are both compiled -- focus on the mainloop CONSUMER which is shared, and the COOP producer). Characterize the STEADY-STATE mainloop the SAME way as the FA3 analysis:
  - #HGMMA per iteration + shapes; #LDGSTS (cp.async) + cache hint; #MUFU.EX2; #BAR.SYNC/BAR.ARV (named barrier IDs) + mbarrier waits; #SHFL; issue order/interleave.
  - Report the per-iteration instruction MIX and structure. Quote representative snippets.
  Goal: get the same instruction-mix profile as FA3 so they can be diffed. Be concrete with counts + snippets.`,
  { label: 'my-sass', phase: 'Extract', agentType: 'Explore' }
)

const C = agent(
  `Read cutlass's PipelineAsync implementation in ${CUTLASS} (search include/cutlass/pipeline/ for PipelineAsync / sm90 pipeline). Also skim ${REF}/mainloop_fwd_sm90_tma_gmma_ws.hpp for how it drives PipelineAsync for the cp.async paged path. Answer concretely (file:line + quotes):
  1. When a producer commits a cp.async stage via PipelineAsync producer_commit with cpasync_barrier_arrive, what EXACT PTX/mechanism is emitted (cp.async.mbarrier.arrive? a commit_group? a fence)? 
  2. When the consumer does consumer_wait on a PipelineAsync (cp.async, NOT TMA), what mechanism (mbarrier.try_wait? test_wait? a spin? a suspend)? Is there any fence.proxy.async or is the ordering purely via the mbarrier?
  3. Does cutlass's cp.async consumer_wait differ from a naive \`while(!mbarrier.try_wait(bar,phase)){}\` spin -- e.g. does it use mbarrier.try_wait with a suspend/sleep, or test_wait, or a specific backoff? Quote it.
  4. Any detail in how cutlass issues the cp.async (the copy atom for SM80_CP_ASYNC_CACHEGLOBAL) that differs from a plain \`cp.async.ca.shared.global [dst], [src], 16\` -- e.g. a prefetch, a specific cache operator, a zfill.
  Goal: find any cheap ordering/wait mechanism cutlass uses that a hand-written kernel typically gets wrong (making its barrier/wait stalls higher). Concrete quotes.`,
  { label: 'cutlass-pipeline', phase: 'Extract', agentType: 'Explore' }
)

const [fa3, mine, cut] = await parallel([() => A, () => B, () => C])

phase('Synthesize')

const plan = await agent(
  `Synthesize a concrete, faithful fix for my hand-written FA3 paged-prefill kernel. ESTABLISHED context (do not re-derive): my dense-causal hd128 mainloop already BEATS FA3 (1.04x) with the same schedule, so the mainloop is fine. Paged prefill is 0.85-0.91x vs FA3. ncu says: my tensor 61% vs FA3 75%; my barrier-stall 0.73 vs FA3 0.44; my long_scoreboard 0.37 vs FA3 1.14; dispatch identical. Paged DECODE beats FA3 (1.22x) with the same cp.async gather, so the gather is fine -- the gap is the compute-bound prefill's barrier/scheduling at 1 CTA/SM. Persist (5 variants), K-ahead separate-K/V (2x), 2-CTA occupancy, prefetched scheduler ALL regressed; only removing a redundant fence helped (+1.5%, committed).

Here are three SASS/source analyses:

FA3 PAGED MAINLOOP SASS (instruction mix + structure):
${fa3}

MY PAGED MAINLOOP SASS (instruction mix + structure):
${mine}

CUTLASS cp.async PIPELINE MECHANISM:
${cut}

Produce:
1. The CONCRETE per-iteration instruction-mix DIFFERENCES between FA3 and mine (e.g. FA3 has fewer BAR.SYNC, or a different mbarrier-wait mechanism, or a different cp.async cache hint, or fewer non-HGMMA instructions). Rank by likely impact on the 61->75% tensor-util / 0.73->0.44 barrier-stall gap.
2. For the TOP 1-2 differences, a SPECIFIC faithful code change to my kernel (kernels/fa3_hopper.cu) -- e.g. change the mbarrier wait to X, remove/add barrier Y, change the cp.async to hint Z. Must be faithful (no cutlass dep, handles real scattered KV) and must NOT be one of the already-failed levers (persist/K-ahead/occupancy/prefetch-scheduler).
3. If the SASS shows the gap is NOT closable by an instruction-level change (e.g. it's fundamentally the persist continuity), say so plainly with the evidence.
Be decisive and concrete -- I will implement the top recommendation.`,
  { label: 'synth', phase: 'Synthesize' }
)

return { plan }
