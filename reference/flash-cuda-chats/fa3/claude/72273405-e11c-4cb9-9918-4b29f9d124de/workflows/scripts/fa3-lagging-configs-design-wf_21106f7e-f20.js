export const meta = {
  name: 'fa3-lagging-configs-design',
  description: 'Parallel design/profiling of every lagging fa3_hopper.cu config group -> concrete implementation blueprints',
  phases: [{ title: 'Design', detail: 'one agent per structurally-distinct lagging group' }],
}

phase('Design')

const KERNEL = '/home/yak/fa3-cuda/kernels/fa3_hopper.cu'
const REF = '/home/yak/fa3-cuda/reference/vllm-flash-attention/hopper'

const tasks = [
  {
    label: 'design:fp8-hd256',
    prompt: `Design (concrete implementation blueprint, do NOT edit files) a NATIVE fp8 (e4m3) head_dim=256 path for ${KERNEL}.
Context: attn_dense_hd256_fp8 runs 0.196x and attn_kv_hd256_fp8_decode 0.342x vs FA3 -- they fall to the bf16-CONVERT path (fp8->bf16 in smem then bf16 wgmma, 1x tensor throughput). Native fp8 wgmma exists only for hd/hdv in {128 (prefill+decode), 64, 96 (decode)}: launch_prefill_fp8 (~line 2470) hard-checks hd==hdv==128 and only instantiates <128,128>; launch_decode_split_fp8 instantiates <128,128>,<64,64>,<96,96>.
Study prefill_fp8_kernel (~line 1620) and prefill_fp8_persist_kernel: the V-transpose (transpose_v_smem8), P->fp8 round-trip through smem for SS PV, the HDQK=128 padding, acc_o register budget, and the smem sizing. Determine EXACTLY what it takes to add a native <256,256> instantiation: acc_o[128] register budget (hd256 is already 1 CTA/SM at bf16), smem for K/V/Vt/P tiles at hd256 (CTA_N tile), whether it fits at STAGES>=2, the wgmma m64nNk32 tiling over 256 cols (KBOX/VBOX=256/64=4), and the routing changes in mha_fwd (~line 3120) + launch_prefill_fp8 + launch_decode_split_fp8. Report the exact functions/lines to change, the smem/register budget math, and correctness risks. If a native path can't fit hd256 in registers/smem, say so and propose the best alternative (e.g. a faster convert path, or a split-D approach).`,
  },
  {
    label: 'design:persist-paged',
    prompt: `Produce a HARDENED, deadlock-free implementation diff (do NOT edit files) to enable the PERSISTENT scheduler for PAGED attention in ${KERNEL}, reusing the coalesced cp.async gather that now exists in prefill_kernel's producer (search for "COALESCED COOPERATIVE gather" ~line 1090: TPR=HD/8 lanes/row + smem page-row prefetch sPR[STAGES][CN] + producer-WG bar.sync id 1).
Current state: prefill_persist_kernel (~line 1250) is a 132-CTA continuous-ring atomic-work_ctr scheduler, TMA-only load_kv, gated to !paged && num_splits<=1 (launch_prefill_t ~line 2125). Paged prefill (0.64-0.70x) currently runs the non-persistent per-tile grid (~8 waves); paged decode (0.62-0.84x, DRAM-latency-bound at ~40% BW) runs launch_decode_split at DCM=64.
TWO deliverables:
(1) PAGED PREFILL (num_splits=1): add the coalesced-gather branch (port the exact prefill_kernel coop gather incl. sPR smem prefetch + a producer-WG bar.sync using a FREE barrier id -- note the persist producer already uses bar.sync 1 for the scheduler broadcast @~line 1330, so pick id 2-7) into the persist ring's load_kv lambda (~line 1345, currently TMA-only), and relax the launch gate to allow paged. Wave-collapse 8->1.
(2) PAGED DECODE split-KV on the persistent ring: the harder one. Give the exact work-item decomposition ((hk,bi,split)), the pf_work split subdivision, the split epilogue (partial O/LSE + emit_empty_lse for empty splits), reuse of dec::combine_kernel, and a persistent num_splits heuristic (combine_kernel sw[64] cap). Address the smem/STAGES budget and every deadlock/lockstep risk (full[]/empty[] global-g gating, producer-WG barrier not blocking consumers, Q-gather pt<CM guard @~1015).
Report ordered steps, exact functions/lines, and a correctness-risk checklist. Ground everything in the actual code.`,
  },
  {
    label: 'design:dense-0.9x-band',
    prompt: `Profile and design fixes (do NOT edit files) for the DENSE/non-paged attntest configs stuck at 0.85-0.95x vs FA3 in ${KERNEL}. From the latest run these are: attn_dense_hd256_causal 0.848, attn_dense_hd192_dv128 0.923, attn_kv_hd128_prefill3577 0.921, attn_dense_hd128_large4k 0.931, attn_dense_hd192_causal 0.940, attn_dense_hd128_softcap 0.941, attn_dense_hd128_seq8k 0.941, attn_dense_hd128_fp8_gqa 0.946, attn_dense_hd128_softcap_gqa 0.948.
You have a prebuilt .so at /home/yak/fa3-cuda/.fa3_trials/_distill_fa3_C/_distill_fa3_C.so. Use bench/prof_real.py to run configs (FA3_NS=_distill_fa3_C FA3_CFG=<name> FA3_ITERS=3 /home/yak/miniconda3/envs/dev/bin/python -m bench.prof_real) under ncu ( -k "regex:prefill|persist" -c 1 --section SpeedOfLight --section WarpStateStats --section Occupancy ). Profile 3-4 representative ones (hd256_causal, large4k or seq8k, softcap, hd192_dv128) and identify the bottleneck per group (tensor-pipe? register spill? barrier? occupancy? softcap SFU?). The kernel comments already discuss some (hd256 register-cap @~2340; large4k barrier/compute-overlap; softcap tanh). Cross-reference how ${REF} handles these (tile sizes in tile_size.h, the softcap fast-tanh, hd256 register allocation). Report per-config: measured bottleneck + a concrete, tractable code change (tile width, STAGES, ping-pong gating, softcap path, register rebalance) with exact lines. Rank by expected gain/effort. Flag any that are CUTLASS-codegen-ceiling (not worth it in pure CUDA).`,
  },
  {
    label: 'design:decode-mlp',
    prompt: `Adversarial fresh analysis (do NOT edit files) of the DRAM-latency wall on paged GQA decode in ${KERNEL}. FACTS: after a coalesced-gather fix, attn_real_llama8b_dec_s4k (h32/kv8/d128, q=1, page_size=1, ctx4k, batch16) is DRAM-latency-bound at ~40% of peak BW; FA3 hits ~55% (1.5x faster). REFUTED levers (measured): more num_splits (0..32 all regress, even post-coalescing), 2 CTA/SM (neutral -- halving STAGES cancels extra warps), per-row TMA (overhead). The decode runs the prefill wgmma mainloop at DCM=64 (1 consumer WG + 1 producer WG, STAGES=5, 1 CTA/SM, 12.5% occ). It packs q*rep rows into M (rep=4 -> M=4 of a 64-row tile = mostly wasted wgmma).
QUESTION: what maximizes memory-level parallelism (outstanding loads) to lift DRAM utilization toward FA3's? Consider and rank: (a) the persistent scheduler with splits (does it actually raise concurrent load streams vs the current 128-CTA grid, or not?); (b) a DEDICATED tiny-M decode kernel -- since q=1 is a GEMV not a GEMM, a warp-parallel flash-decoding kernel (cf. the DEAD dec::decode_kernel ~line 2500, one warp per (b,hk,split)) with many warps each streaming a KV chunk may have far higher MLP than the DCM=64 wgmma mainloop that wastes 15/16 of its tensor tile; (c) more producer warps / a deeper software-prefetch (async cp.async issued further ahead); (d) increasing loads-in-flight per SM some other way. Study how ${REF} flash_fwd + the split/combine + PackGQA keep tiny-M decode's DRAM busy, and flash_fwd_combine_kernel. Report the SINGLE highest-MLP approach with a concrete implementation sketch (exact kernel structure, warp roles, smem, expected DRAM% gain), plus why it beats the refuted levers.`,
  },
]

const results = await parallel(tasks.map(t => () => agent(t.prompt, { label: t.label, phase: 'Design' })))
return tasks.map((t, i) => ({ group: t.label, blueprint: results[i] }))
