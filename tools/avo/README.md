# AVO-style evolutionary kernel search (reproducer)

How the `fwd_kernel_fa4` Blackwell kernel was tuned from a correct-but-slow 0.29× port
to ~0.78× of the FA4 oracle (matching CUTLASS on long context), in pure CUDA + inline PTX.
Method inspired by NVIDIA's AVO paper (arXiv:2603.24517): an LLM coding agent as the
"variation operator", **monotone single lineage** (commit only correct improvements),
**breadth** (several diverse mutations per generation), and **self-supervision** (a
strategist reviews the whole trajectory and steers the next generation when it stalls).

## Files
- `fitness.sh` — the score function: build a candidate `.cu`, correctness-gate it, print
  `geomean` vs the FA4 oracle. Pure bash + the repo's `bench.eval --target blackwell`. **Portable.**
- `search.js` — the workflow: the monotone-lineage / breadth / strategist loop + the seed
  directives. **Runs via Claude Code's Workflow tool**, not standalone Node (see caveat).

## Requirements
- NVIDIA **Blackwell B200 (SM100 / `compute_100a`)** GPU + CUDA 12.x, PyTorch.
- The **FA4 oracle** for the A/B (Dao-AILab `flash_attn.cute` / `flash-attn-4`) — see the oracle adapter in `bench/eval.py`.
- To run the *orchestration*: **Claude Code with the Workflow tool** (the `.js` calls the
  `agent()`/`parallel()`/`phase()`/`log()` runtime hooks — it is not a plain Node script).

## Run it (via Claude Code on the target machine)
```
# 1. sanity-check the fitness function on one candidate (no Claude needed):
cp kernels/fa4_blackwell.cu /tmp/avo/seed.cu
tools/avo/fitness.sh /tmp/avo/seed.cu _avo_probe ""      # -> RESULT ns=_avo_probe geomean=..x

# 2. seed the search head + run the workflow from Claude Code:
#    In Claude Code, invoke the Workflow tool with this script:
#        Workflow({ scriptPath: "tools/avo/search.js" })
#    Edit the top of search.js first: `head` = seed kernel copy, `bestGeo` = its geomean,
#    and the SEED[] directives. Each "stage" = re-run with the previous stage's winner as `head`.
```

## What each generation does (in `search.js`)
1. Fan out N agents; each `cp`s the current lineage `head`, applies its assigned structural
   mutation on an **isolated copy**, runs `fitness.sh`, returns `{geomean, correct, path, diagnosis}`.
2. **Monotone commit:** the best *correct* variant that beats `bestGeo` becomes the new `head`.
3. **Self-supervision:** a strategist agent reads every attempt + diagnosis and emits the next
   generation's directives (avoid dead ends, deepen what worked, introduce new ideas on a stall).

## The seed directives that produced the wins (hd128 dense, seeded from the 0.29× port)
Gen-1 (biggest lever first): **persistence + LPT + continuous cross-tile overlap** (0.29→0.68,
the decisive win), smem-cut for a deeper K/V ring, leaner softmax→correction→MMA handshakes,
raise warp-eligibility, inline-vs-split correction test, SASS-schedule matching to FA4.
Later stages: tile-boundary overlap + a `q_sched` decouple, **Q-prescale folded into the softmax
exp** (0.73→0.74), split the K/V "full" barrier so QK doesn't wait on the trailing V-TMA,
**depth-2 softmax↔correction handshake with backpressure** (0.75→0.77). Register rebalance,
depth-3 handshake, and address-hoisting were tried and found neutral/negative — kept out.

## Results (dense hd128 prefill, geomean vs FA4 oracle)
`0.287 (seed) → 0.675 (gen1) → 0.743 → 0.765 → ~0.777`. Full-suite: 48/59 correct, 0 wrong.
Per-config and CUTLASS comparison are in the top-level `README.md`.

## Standalone (no Claude Code)
The loop is small. `fitness.sh` is the only GPU-side piece and runs anywhere. To reimplement the
orchestration framework-independently, drive `fitness.sh` from any harness that uses a coding LLM
(e.g. the Anthropic API) as the per-generation variation operator — the algorithm is exactly the
three steps above. Ask and a minimal Python version can be provided.
