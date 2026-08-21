export const meta = {
  name: 'fa3-hard-recipes',
  description: 'Extract exact FA3 implementation recipes for hd256 prefill and fp8 dense from reference source',
  phases: [{ title: 'Research', detail: 'parallel deep-reads of FA3/CUTLASS reference' },
           { title: 'Synthesize', detail: 'merge into concrete implementation recipes' }],
}

const REF = '/home/yak/fa3-cuda/reference/vllm-flash-attention/hopper'
const KERN = '/home/yak/fa3-cuda/kernels/fa3_fwd.cu'

const RECIPE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['summary', 'concrete_steps', 'layout_math', 'risks', 'confidence'],
  properties: {
    summary: { type: 'string', description: 'One-paragraph distilled answer' },
    concrete_steps: { type: 'array', items: { type: 'string' }, description: 'Ordered, specific implementation steps referencing exact PTX/layout' },
    layout_math: { type: 'string', description: 'Exact register/smem layout math (thread->element mapping) needed to implement, with file:line citations' },
    risks: { type: 'array', items: { type: 'string' } },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

phase('Research')

const tasks = [
  {
    label: 'fp8-registerP',
    prompt: `Read FA3's fp8 (e4m3) forward mainloop in ${REF}/mainloop_fwd_sm90_tma_gmma_ws.hpp and any helper in ${REF}/utils.h, plus CUTLASS layout helpers under /home/yak/fa3-cuda/reference/cutlass for the GMMA register-source A operand.

GOAL: produce the EXACT recipe to make the softmax-probability matrix P register-resident as the A operand of the PV wgmma for fp8 e4m3 (FA3 mandates MmaPV_is_RS=true for fp8). Our kernel (${KERN}, function qk_fp8_ss / pv_fp8_ss / scatter_P_fp8 around lines 672-718) currently round-trips P through smem (scatter_P_fp8 + bar.sync + SS PV), which is the documented gap.

I need:
1. The exact register layout of the QK f32 accumulator (m64n128k32 e4m3, C fragment) per thread: thread (warp w, lane l) -> which (row, n-col=key) elements. Our layout: s[c*4+r*2+p] = (row=16w+l/4+8r, col=8c+2(l&3)+p), c in 0..15. Confirm against CUTLASS GMMA C layout.
2. The exact register layout of the PV A operand (P, m64n128k32 e4m3, A-from-registers): thread (w,l) -> which (row, k=key) elements, packed how many e4m3 per 32-bit register and in what key order.
3. The precise transform (the cutlass convert_layout_acc_Aregs equivalent): which warp shuffles / byte permutes (prmt) / no-ops are needed to go from layout (1) to layout (2) for e4m3, INCLUDING the f32->e4m3 conversion packing. Be concrete enough that I can hand-write the PTX without CUTLASS.
4. Whether FA3 ALSO uses TMA (not cp.async) to load fp8 K and Q in the dense non-paged case, and the exact tensor-map params (dtype, swizzle mode, box dims) it uses for 1-byte e4m3 K/Q/V loads. Our cp.async path is long_scoreboard-bound (2.4 cyc/issue), so the load path may matter MORE than the P round-trip.

Return the schema. In concrete_steps, rank: is register-P or TMA-loads the bigger expected win for an fp8 dense prefill (hd128, q1024 k1024, batch9, causal) and why.`,
  },
  {
    label: 'hd256-pipeline',
    prompt: `Read FA3's forward mainloop ${REF}/mainloop_fwd_sm90_tma_gmma_ws.hpp, the kernel body ${REF}/flash_fwd_kernel_sm90.h, the pipeline ${REF}/sm90_pipeline_no_cluster.hpp, named barriers ${REF}/named_barrier.hpp, and tile sizes ${REF}/tile_size.h.

GOAL: explain EXACTLY how FA3 achieves ~62% tensor-core utilization for head_dim=256 causal prefill using kBlockM=128, kBlockN=80, only 2 K/V pipeline stages (smem ~224KB). Our hand-written kernel (${KERN}, prefill_persist_kernel ~line 1206 and the consumer mainloop ~lines 1400-1488, plus per-tile prefill_kernel ~line 882) STARVES badly at 2 stages (measured 0.58-0.63x of FA3) and only works at CN=48 / 3 stages (0.84x). The KV working set (~144MB) exceeds L2 (50MB) so it is DRAM-latency sensitive; our stalls are dominated by wgmma 'wait' + 'long_scoreboard' between narrow CN=48 tiles.

I need to know what FA3 does structurally that lets 2 stages suffice for hd256 where ours can't:
1. How many warpgroups (producer/consumer split), and exactly how the K and V loads are staged (separate K and V pipelines? different stage counts for K vs V? is V loaded once and reused?).
2. The IntraWGOverlap mechanism: how QK(i+1) overlaps PV(i) within a consumer warpgroup, and how softmax is scheduled so the tensor pipe never idles. The exact named-barrier ping-pong between the two consumer warpgroups (arrive/sync counts, who goes first).
3. Whether FA3 uses ONE persistent producer warp (NumProducerThreads) issuing TMA vs a full warpgroup, and whether that matters for keeping 2 stages fed.
4. Any smem-saving trick that lets CN=80 + CM=128 fit at higher effective pipeline depth (e.g. MmaPV_is_RS frees the P smem tile; Q in smem once; V reused).
5. Concretely: what is the SMALLEST change to our existing CN=48/STAGES=3/CM=128 persist mainloop that would most raise tensor utilization toward FA3's, given we CANNOT widen CN (smem-capped) and CANNOT use 2 stages (starves)? E.g. enabling the inter-WG ping-pong for HD>128, separating K/V stage depth, single-warp TMA producer, reducing acc_o register pressure (HDV=256 -> acc_o[128] f32 is huge).

Return the schema; concrete_steps should be the ranked list of portable changes to our kernel.`,
  },
  {
    label: 'fp8-vtranspose-and-occupancy',
    prompt: `Read FA3 ${REF}/mainloop_fwd_sm90_tma_gmma_ws.hpp (focus on Transpose_V = Is_FP8 && !V_colmajor, the LDSM.T/STSM V transpose around the static_asserts on kHeadDimV, lines ~183-215) and ${REF}/utils.h.

GOAL: two things for our fp8 dense prefill (${KERN}: transpose_v_smem8 ~line 757 does a byte-by-byte smem->smem transpose; gather_row_k8_cp / gather_row_plain8_cp ~lines 735-753 do cp.async; prefill_fp8_persist_kernel ~line 1700):
1. FA3's exact V-transpose method for fp8 (it uses hardware ldmatrix.trans / stmatrix). Give the exact PTX (ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 or stmatrix) and the smem layout it transposes between, so I could replace our slow byte-by-byte transpose_v_smem8 with ldmatrix.trans. Note: fp8 is 1-byte; how does FA3 handle the 1-byte transpose (ldmatrix.trans works on 16-bit elements — does FA3 transpose pairs of fp8 as bf16-like 16-bit, or use a different trick)?
2. Whether FA3 runs fp8 hd128 prefill at 1 or 2 CTAs/SM and the exact smem budget (it reports 167KB in our profiling). Our fp8 persist uses ~180KB (1 CTA/SM) and is long_scoreboard-bound. Does FA3's smaller smem come from NOT double-storing V (no separate plain-staging + transposed buffer)? How does it avoid the staging buffer if it uses ldmatrix.trans?

Return the schema. concrete_steps = how to cut our fp8 smem and speed the V transpose.`,
  },
]

const research = await parallel(tasks.map(t => () =>
  agent(t.prompt, { label: t.label, phase: 'Research', schema: RECIPE_SCHEMA, effort: 'high' })
    .then(r => ({ label: t.label, recipe: r }))))

phase('Synthesize')
const valid = research.filter(Boolean)
const synth = await agent(
  `You are synthesizing implementation guidance for optimizing a hand-written FA3 CUDA kernel (${KERN}). Two scenarios remain below the 0.90x-of-FA3 target: hd256 causal prefill (0.838x) and fp8 dense prefill (0.852x). Below are research findings from reading the FA3 reference source. Produce a SINGLE prioritized action plan: for EACH of the two scenarios, the single highest-expected-value, lowest-risk concrete change to try first, with the exact layout/PTX details merged in, and a fallback. Be concrete and implementation-ready.\n\nFINDINGS:\n${valid.map(v => '### ' + v.label + '\n' + JSON.stringify(v.recipe, null, 2)).join('\n\n')}`,
  { label: 'synthesize', phase: 'Synthesize', effort: 'high' })

return { research: valid, plan: synth }
