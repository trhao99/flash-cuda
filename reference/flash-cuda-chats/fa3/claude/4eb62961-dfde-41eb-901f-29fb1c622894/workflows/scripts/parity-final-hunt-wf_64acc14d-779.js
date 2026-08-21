export const meta = {
  name: 'parity-final-hunt',
  description: 'Exhaustive hunt for llama8b_pf_s2k (cold-start) + dense_hd256 levers; return all candidates',
  phases: [
    { title: 'Hunt', detail: 's2k cold-start + hd256 + consumer, multiple angles' },
    { title: 'Verify', detail: 'adversarially verify each candidate is real + not-tried' },
  ],
}
const KERNEL = '/home/yak/fa3-cuda/kernels/fa3_hopper.cu'
const FA3 = '/home/yak/fa3-cuda/reference/vllm-flash-attention/hopper/'
const CUTLASS = '/home/yak/fa3-cuda/reference/cutlass'
const MEM = '/home/yak/.claude/projects/-home-yak-fa3-cuda/memory/fa3-perf-gap-closing.md'
const SCH = {
  type: 'object',
  properties: { candidates: { type: 'array', items: { type: 'object', properties: {
    target: { type: 'string' }, lever: { type: 'string' },
    mechanism: { type: 'string', description: 'the inefficiency in our code vs FA3, file:line both' },
    expected_impact_pp: { type: 'number' }, faithful: { type: 'boolean' }, already_tried: { type: 'boolean' },
    concrete_change: { type: 'string' }, risk: { type: 'string' },
  }, required: ['target','lever','mechanism','expected_impact_pp','faithful','already_tried','concrete_change'] } } },
  required: ['candidates'],
}
const COMMON = `Our cutlass-free FA3 CUDA kernel (Hopper sm90): ${KERNEL}. FA3 ref: ${FA3}. Cutlass: ${CUTLASS}.
READ ${MEM} FIRST — it lists ~8 sessions of TRIED+FAILED levers; do NOT re-propose them. RECENT WINS (this session, committed): (a) COOP-gather offset-broadcast (compute ent*stride once/row, shfl-broadcast the 64-bit offset) and (b) pow2 shift/mask for the page-divmod (block_size runtime int -> nvcc emitted software idiv/imod per key). BOTH removed redundant PRODUCER int-ALU that was thinning the cp.async issue stream. That KIND of fix (redundant per-lane int math, a faithful FA3 schedule/layout we don't match) is what to hunt for. Be brutally honest: mark already_tried=true if memory shows it, faithful=false if benchmark-gaming. Prefer concrete edits with file:line. Read-only.`

phase('Hunt')
const groups = await parallel([
  () => agent(`${COMMON}\n\nTARGET: attn_real_llama8b_pf_s2k (0.919x, the last named prefill config <0.95; s16k is 0.945). ncu: tensor 47.8%, long_scoreboard 0.97 (HIGH -- memory-latency-bound), barrier 0.74, waves 7.76. It uses prefill_kernel (per-tile). DIAGNOSIS: short causal work-items (n_cnt small) -> the fast consumer drains the STAGES=3 K/V ring faster than the (now-lean) COOP producer refills it from L2 (KV ~16MB fits L2) -> consumer stalls on full[] (long_sb 0.97). This is per-CTA cold-start / short-work-item ring-drain. Hunt CONCRETE faithful levers to hide that L2 gather latency for short work-items WITHOUT the persist (persist Q-bubble on short items already failed per memory): e.g. deeper producer run-ahead, issuing more cp.async before the first consumer wait, a warmer prologue, reducing the consumer's per-tile drain rate mismatch, prefetch hints, or ANY remaining redundant producer/consumer work (like the offset-broadcast/idiv wins). Also RE-EVALUATE whether the paged persist is now viable given the lean producer (offset-broadcast+idiv make load_kv_coop cheaper) -- but load_kv_coop (persist producer ~line 1600) does NOT yet have those two fixes; would applying them + re-enabling the paged persist (gate '&& false' ~line 2530) beat per-tile 0.919? Rank by EV.`, { label: 'hunt:s2k', phase: 'Hunt', schema: SCH }),

  () => agent(`${COMMON}\n\nTARGET: attn_dense_hd256_causal (0.879x). DENSE bf16 hd256, prefill_persist_kernel (TMA producer, NOT the COOP gather -- so the gather wins don't apply), CN=48/STAGES=3/CM=128, acc_o[128] (128 fp32 regs/thread), 1 CTA/SM, tensor 46%. FA3 ~0.92 at CN=80/STAGES=2. The circular constraint (memory): our fast consumer needs STAGES=3 to hide the 256-wide TMA latency, so CN can't widen to 80 (smem), so tensor stays low. TRIED+FAILED: rescale-before, 240/24 regs, per-tile, CN=80/STAGES=2. Think DIFFERENTLY: can we REDUCE the hd256 consumer's NON-TENSOR work (softmax EX2, the 128-FMUL acc_o rescale, mask, pack) so it becomes tensor-bound enough that STAGES=2 suffices -> then CN=80 fits -> parity? Or a smem-layout trick (e.g. Q in a cheaper form, split-K accumulation to smem) that frees a stage at CN>=64? Or does FA3's hd256 causal use a specific acc tiling / 2-stage-that-works we don't? Deep-read tile_size.h + mainloop_fwd_sm90 + flash_fwd_kernel_sm90.h for hd256. Only NEW faithful candidates. Rank by EV; be honest if none exist.`, { label: 'hunt:hd256', phase: 'Hunt', schema: SCH }),

  () => agent(`${COMMON}\n\nTARGET: the consumer mainloop shared by paged prefill (prefill_kernel ~lines 1310-1432) -- raise tensor / cut barrier for llama8b_pf (s16k tensor 70% barrier 0.75 vs FA3 74%/0.44). The producer is now lean (offset-broadcast+idiv). Hunt in the CONSUMER for the same KIND of redundant per-thread work or faithful-schedule gaps vs FA3's mainloop_fwd_sm90 consumer: redundant int64/address math in wgmma descriptor construction (make_smem_desc per tile?), redundant recomputation across the ping-pong WGs, the softmax reduction (quad_max/quad_sum shfl count vs FA3), pack_2x, the mask fast-path. Also: FA3's exact ping-pong (warp_scheduler_barrier) placement/count vs ours (nb_sync/nb_arrive) -- any extra barrier or a cheaper form. Only NEW faithful candidates (ping-pong on/off, K-ahead, parking, STAGES already tried per memory). Rank by EV.`, { label: 'hunt:consumer', phase: 'Hunt', schema: SCH }),
])
const raw = groups.filter(Boolean).flatMap(g => g.candidates || [])
log(`hunt found ${raw.length} candidates`)

phase('Verify')
const verified = await parallel(raw.map(c => () =>
  agent(`Adversarially verify this candidate optimization for our cutlass-free FA3 kernel (${KERNEL}, FA3 ref ${FA3}, tried-list ${MEM}). Confirm by reading the actual code: (1) is the claimed inefficiency REAL (cite our file:line + FA3's)? (2) is it NOT already in the tried-and-failed memory? (3) is it faithful (not benchmark-gaming)? (4) is the concrete_change correct + safe? Return the candidate with corrected fields and set already_tried=true / faithful=false / expected_impact_pp=0 if it fails verification. CANDIDATE:\n${JSON.stringify(c, null, 2)}`,
    { label: `verify:${(c.lever||'').slice(0,24)}`, phase: 'Verify', schema: { type:'object', properties:{ candidate:{ type:'object' }, verdict:{ type:'string', enum:['CONFIRMED','REJECTED'] }, note:{ type:'string' } }, required:['candidate','verdict'] } })
      .then(v => v ? { ...v.candidate, verdict: v.verdict, note: v.note } : null)))
const survivors = verified.filter(Boolean).filter(c => c.verdict === 'CONFIRMED' && !c.already_tried && c.faithful !== false)
log(`${survivors.length} candidates CONFIRMED + untried + faithful`)
return { candidates: survivors.sort((a,b)=>(b.expected_impact_pp||0)-(a.expected_impact_pp||0)) }