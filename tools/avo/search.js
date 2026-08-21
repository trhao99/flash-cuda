// AVO-style evolutionary kernel search — a Claude Code WORKFLOW script (not standalone Node).
// Run it from Claude Code:  Workflow({ scriptPath: "tools/avo/search.js" }).  See tools/avo/README.md.
// It uses the runtime hooks agent()/parallel()/phase()/log() + `schema`, provided by the Workflow tool.
//
// TO RE-RUN ON A FRESH MACHINE, edit the 3 marked lines near the bottom:
//   `head`    -> a copy of your seed kernel (e.g. cp kernels/fa4_blackwell.cu /tmp/avo/seed.cu)
//   `bestGeo` -> that seed's geomean from tools/avo/fitness.sh
//   `SEED[]`  -> the mutation directives for generation 1 (this file's are the stage-3 set;
//                the big early wins came from the gen-1 set documented in README.md).
// A "stage" = one run of this script; chain stages by seeding `head` with the previous winner.
export const meta = {
  name: 'avo-search-fa4-s3',
  description: 'AVO search STAGE 3: climb from the 0.765 cuda-only FA4 kernel toward parity (surgical, higher-EV directives)',
  phases: [
    { title: 'Gen1', detail: '6 seed mutations vs the 0.765 base' },
    { title: 'Gen2', detail: 'strategist re-direction + 6 mutations' },
    { title: 'Gen3', detail: 'strategist re-direction + 6 mutations' },
  ],
}

const HARNESS = `bash tools/avo/fitness.sh <yourfile.cu> <unique_ns> "FA4_REBUILD=1"`
const K = `
GOAL: push a hand-written CUDA (+ inline PTX; NO CUTLASS/CuTe includes) Blackwell SM100 FlashAttention kernel toward
FA4 parity (geomean 1.0 on 3 hd128 short configs causal/noncausal/gqa). AVO paper proves a pure CUDA+PTX kernel CAN
beat FA4. You are ONE variation operator in an evolutionary search. Current best cuda-only = 0.765x (already beat the
old ws2 inline kernel 0.66); STILL CLIMBING but into the codegen-realization wall — push it higher.

THE KERNEL: fwd_kernel_fa4<T,HD,HDV> (grep it), gated by getenv("FA4_REBUILD") in launch_ws. FA4's warp-specialized
structure: w0-3 Softmax0 (rows0-127), w4-7 Softmax1 (rows128-255), w8-11 Correction, w12 MMA, w13 Load, w14/15
Q-producer/Epilogue; ONE 256-row tile = 2 stacked halves SHARING K/V; tmem S0@0 S1@128 (P aliases S) O0@256 O1@384
(512 cols full). Correct 4/4 at ~0.765x. SAME FILE has fwd_kernel_ws2 (0.66 inline) + fwd_kernel_ws2_cont
(persistent) — MINE them for idioms (K/V ring, persistent grid, continuous overlap, inline rescale, tcgen05/mbar/TMA
primitives, softmax_full_ptx / _ord / _ordcs).

ALREADY IN THE 0.765 BASE (do NOT re-add): PERSISTENT grid + LPT + CONTINUOUS cross-tile overlap; tile-boundary
overlap (early sQ release + q_sched mbar + MMA pre-waits KVF[0]); Q prescale FOLDED into softmax exp2 (Q lands raw,
fma.rn.f32x2 dp,vp,qsc2,nm2); KVS=2 K/V ring; lagging PV; order_s @col15; early-corr-publish; SPLIT kf_full/vf_full
Load->MMA barrier (QK waits only K, not trailing V TMA); DEPTH-2 softmax<->correction C-channel with c_empty
backpressure (c_full[4]/c_empty[4], sScale/sFinSum/sFinMax double-buffered, softmax gets 2-block credit).
STILL-LOCKSTEP / UNTRIED (high-value targets): the O pipeline (MMA->correction OF/OE is still DEPTH-1/lockstep —
extending the depth-2 idea to it is the top candidate); inline-fold correction into softmax (drop the correction-WG
hop entirely — was 0.734 on an EARLIER base, never tested vs the depth-2 champion); cross-block S-read prefetch.
NEUTRAL/DEAD (don't retry as-is): address-hoist (won on ws2, neutral here), reg-split sweep (192/64/64 best),
order_s relax, ex2 emulation, 2-CTA/SM (tmem-blocked), correction-WG graft onto ws2.

THE BOTTLENECK: latency-bound; too few eligible warps hide the tmem-S-read + cross-warp handshakes. Raise tensor-active.

HARD RULES:
1. Start: cp <the head file you're given> /tmp/avo/<yourfile>.cu ; edit ONLY your copy. NEVER touch
   /home/yak/fa3-cuda/kernels/fa4_blackwell.cu.
2. CUDA-only (prose comments mentioning cutlass/cute are fine; NO includes). GPU shared: never kill; FA4_EVAL_GPU=0.
3. Evaluate: ${HARNESS} (3 configs, correctness+perf). Stay correct 3/3. Deadlock-timeout = mbar phase/count bug.
   EVAL CAVEAT: the shared GPU occasionally gives a false ~0.03x (constant ~2450us across ALL configs = contention
   tell) — ALWAYS re-run a big regression before trusting it. Build ~92s; ONE coherent change; report honestly.`

const IMPL_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['built','correct','geomean','path','mutation','diagnosis'],
  properties: {
    built: { type: 'boolean' }, correct: { type: 'boolean' },
    geomean: { type: 'number' }, causal: { type: 'number' }, noncausal: { type: 'number' }, gqa: { type: 'number' },
    path: { type: 'string' }, mutation: { type: 'string' }, diagnosis: { type: 'string' },
  },
}
const STRAT_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['directives','reasoning'],
  properties: { reasoning: { type: 'string' },
    directives: { type: 'array', minItems: 6, maxItems: 6, items: {
      type: 'object', additionalProperties: false, required: ['id','directive'],
      properties: { id: { type: 'string' }, directive: { type: 'string' } } } } },
}

const SEED = [
  { id: 'o-pipeline-depth2', directive: `Extend the winning depth-2 handshake idea to the O (MMA->correction) pipeline, which is still DEPTH-1/lockstep (the MMA produces O for a block then effectively waits correction to drain it before reusing the O slot). Give it a 2-deep per-half-slot ping-pong WITH backpressure (mirror exactly the c_full[4]/c_empty[4] depth-2 scheme already in the softmax<->correction channel: o_full/o_empty per [half*2+slot], MMA gets a 2-block credit so it keeps issuing PV/committing O while correction lags <=2). tmem is full so you canNOT add O columns — this is a BARRIER/credit change over the existing O0@256/O1@384, not new tmem. This is the highest-value candidate (direct extension of the +1.4% stage-2 win).` },
  { id: 'inline-fold', directive: `Test inline-fold: do the O-rescale INLINE in the softmax warp (reuse ws2's tmem_o_rescale), ELIMINATING the correction-WG cross-warp hop entirely, and reassign w8-11. This was 0.734 on an earlier base but was NEVER tested against the current depth-2 champion (0.765). It's an ALTERNATIVE to the depth-2 correction channel, not additive — measure whether removing the hop beats optimizing it. Keep the persistent/overlap/Q-fold base intact.` },
  { id: 'ncu-guided', directive: `NCU-profile the 0.765 kernel on fa3_dense_hd128_causal (build it; use bench/ncu_probe.py or a raw ncu of the torch op). Report the NOW-dominant stall (tensor-active %, top stall reason + %, eligible-warps/issue-rate), then implement the SINGLE change that most directly attacks it. The profile is valuable output regardless.` },
  { id: 'mma-runahead', directive: `Increase MMA run-ahead / warp eligibility. Try KVS=3 (deeper K/V ring) again now that other smem was freed — if it still overflows, find more smem to cut (tighter Q double-buffer, reuse sO). Deepen the Load warp's prefetch so it stays >=2 blocks ahead, and tighten the QK/PV issue so the MMA has queued work during the softmax S-read window. Goal: more simultaneously-eligible warps.` },
  { id: 's-read-prefetch', directive: `Cross-block S-read prefetch on the persistent base: issue the NEXT block's tcgen05.ld for S (or its first 32-col chunk) while the current block's exp2 runs, so the long S-read latency overlaps compute. Register-blocked on ws2 (needed 2x the 128-fp32 fragment) but the correction warps are separate here — try a partial prefetch (next block's first chunk only) to stay within budget. Watch ptxas -v spills + deadlock.` },
  { id: 'longctx-broaden', directive: `Validate + extend to LONG context: add fa3_dense_hd128_seq8k,fa3_dense_hd128_seq16k to your fitness configs. The persistence+continuous-overlap should help long-context (per the ws2_cont finding); if it regresses or fails there, fix it, then deepen any long-context-specific overlap. Also spot-check fa3_dense_hd128_local + fa3_dense_hd128_softcap for correctness (de-risks banking as production). Report short AND long geomeans.` },
]

function implPrompt(d, head, bestGeo, gen) {
  const file = `/tmp/avo/s3_g${gen}_${d.id}.cu`
  return `${K}\n\nYOUR DIRECTION (${d.id}): ${d.directive}\n\nStart: cp ${head} ${file}\nYour file: ${file}\nYour eval ns: _avo_s3_g${gen}_${d.id}\nCurrent best to beat: geomean ${bestGeo}. Iterate build->eval until correct+improved or you exhaust the idea, then return the schema honestly.`
}

let head = '/tmp/avo/fa4rebuild.CKPT_0765.cu'
let bestGeo = 0.765
const progression = [{ gen: 0, best: 0.765, head }]
const allDiag = []
let batch = SEED

for (let gen = 1; gen <= 3; gen++) {
  phase(`Gen${gen}`)
  const results = await parallel(batch.map(d => () => agent(implPrompt(d, head, bestGeo, gen), { label: `g${gen}:${d.id}`, phase: `Gen${gen}`, schema: IMPL_SCHEMA, effort: 'high' })))
  const ok = results.filter(Boolean).filter(r => r.correct && r.geomean > 0)
  ok.sort((a, b) => b.geomean - a.geomean)
  results.filter(Boolean).forEach((r, i) => allDiag.push({ gen, id: batch[i] ? batch[i].id : `d${i}`, geomean: r.geomean, correct: r.correct, mutation: r.mutation, diagnosis: r.diagnosis }))
  const champ = ok[0]
  if (champ && champ.geomean > bestGeo) { bestGeo = champ.geomean; head = champ.path; log(`Gen${gen}: NEW BEST ${bestGeo.toFixed(3)} head=${head}`) }
  else { log(`Gen${gen}: no improvement (best correct ${champ ? champ.geomean.toFixed(3) : 'none'} <= ${bestGeo.toFixed(3)}).`) }
  progression.push({ gen, best: bestGeo, genBestCorrect: champ ? Number(champ.geomean.toFixed(3)) : null, head })

  if (gen < 3) {
    const diagText = allDiag.map(d => `[g${d.gen} ${d.id}] geo=${d.correct ? d.geomean.toFixed(3) : 'FAIL'} :: ${d.mutation || ''} :: ${d.diagnosis || ''}`).join('\n')
    const strat = await agent(`${K}\n\nYou are the SEARCH STRATEGIST (self-supervision). Current best = ${bestGeo.toFixed(3)}x at ${head}. Review the FULL trajectory below and propose the NEXT 6 mutation directives against the current best. AVOID directions shown dead; DOUBLE DOWN on promising ones (combine/deepen); introduce NEW structural ideas if stalling. Concrete + self-contained. Target the measured bottleneck.\n\nTRAJECTORY:\n${diagText}\n\nReturn 6 directives + reasoning.`, { label: `strategist:g${gen}`, phase: `Gen${gen + 1}`, schema: STRAT_SCHEMA, effort: 'high' })
    if (strat && strat.directives && strat.directives.length) { batch = strat.directives; log(`Strategist steered gen${gen + 1}: ${strat.directives.map(d => d.id).join(', ')}`) }
  }
}

return {
  stage3_baseline: 0.765, best_geomean: Number(bestGeo.toFixed(3)), reached_parity: bestGeo >= 1.0,
  best_kernel_path: head, progression,
  top_attempts: allDiag.filter(d => d.correct).sort((a, b) => b.geomean - a.geomean).slice(0, 8).map(d => ({ gen: d.gen, id: d.id, geomean: Number(d.geomean.toFixed(3)), mutation: d.mutation })),
  key_diagnoses: allDiag.sort((a, b) => (b.correct ? b.geomean : 0) - (a.correct ? a.geomean : 0)).slice(0, 10).map(d => ({ gen: d.gen, id: d.id, result: d.correct ? d.geomean.toFixed(3) : 'FAIL', diagnosis: d.diagnosis })),
}
