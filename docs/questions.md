# Possible things to investigate

### Scope (only the losing configs)
- Which losing scenarios share a kernel body (`fwd_kernel_fa4`, `fwd_kernel_fa4dk`, `fwd_kernel_fa4_paged`, `fwd_kernel_ws2`, `ws_paged` / `ws2_paged`, combine), vs only sharing a host launcher?
- Is every loser in one family — long-context dense, large-q paged prefill, medium-q split decode (`decode_qlen120` / `ctx20k`), hd96-padded, or a grab bag?
- For each loser: is FA4 beating us on *kernel cycles* at locked clocks, or only on wall-clock (CuteDSL/JIT/host)?
- Does the gap grow with `seqlen_k` (feed/pipeline), with `seqlen_q` (tile/scheduling), or stay flat (fixed launch tax)?
- Are GQA/MQA losers the same relative gap as MHA at equal tokens, or is head packing / KV broadcast the differentiator?

### Confirm it’s the CUDA kernel, not the harness
- With NCU at locked B200 clocks and L2 flush, is `sm__cycles_active` (or equivalent) still worse than FA4’s cubin for that op?
- Are we comparing against FA4’s *steady-state* JIT cubin (warm cache), not cold CuteDSL compile?
- Same buffers, page table, dtype, causal/window/softcap/sink flags, and split count as the FA4 call?
- Is any of our time outside the attention cubin (scheduler metadata, combine, D2H, `atomicExch` setup, TMA desc build on host)?

### Host / launch (B200 FA4 port — already a known trap)
- Any remaining sync D2H (`cu_seqlens` / max_seqlen) on the losing route (`ws2`, paged, serial) that the dense `fa4` path already removed?
- Per-call host TMA descriptor / gather map build that FA4 amortizes inside the JIT path?
- Extra clears of scheduler counters, O-partials, or LSE that FA4 skips or fuses?
- Are short-grid losers (sub-wave varlen/decode) mostly launch overhead vs FA4’s Python dispatch — and does kernel-only time already match?

### Routing (wrong kernel = false “kernel slow”)
- Is this shape on `fwd_kernel_fa4` / `_paged` / `_fa4dk`, or still on `ws2` / older paged because a gate is stale?
- Would forcing the FA4-faithful kernel (where correct) close the gap without changing PTX?
- Is page_size / hd / window / sink / softcap forcing a slow instantiation that FA4 specializes differently?
- For single-tile / underfilled-q shapes: are we stuck on a 2-tile choreography that can’t map, while FA4 uses a different schedule?

### NCU bottleneck class on B200 (tcgen05 / tmem / TMA)
- Bound by tensor MMA pipe, `long_scoreboard` (KV/Q feed), barrier / mbarrier / `tcgen05.wait`, tmem capacity/serialization, LSU, or low occupancy?
- MMA-pipe % and scoreboard stalls vs FA4 on the *same* scenario — is this KV-feed latency (DRAM low, scoreboard high) like the paged-prefill case?
- CTA/SM and achieved occupancy vs FA4 — same, or are we 1 CTA/SM while they overlap more work per SM?
- Issue-slot breakdown: are gather4 / TMA / cp.async producers freeing slots the consumer actually converts (as on `fa4_paged`), or is the consumer still bubble-bound?

### Warp specialization / FA4 choreography (CUDA-only replica)
- Do we match FA4’s warpgroup roles (load / MMA / softmax0 / softmax1 / correction / epilogue) on this path, or a collapsed `ws2`-style schedule?
- Is the 9-stage (or path-specific) mbarrier pipeline actually full-depth on losers, or draining early on partial K-tiles / masks?
- Dual stacked 128-row halves: is correction/softmax ping-pong overlapping like FA4, or does one half serialize the other?
- Persistent + LPT continuous-overlap: on long-context losers, are we failing to overlap tile N→N+1 the way FA4 does?
- Any divergent `tcgen05.wait` / barrier under a seqlen predicate (correctness risk *and* silent serialization)?

### Paged KV path (largest residual on real shapes)
- For paged prefill losers: still KV-feed-latency-bound after gather4 + Q-preload + one-shot page-pointer resolve — or a new binder?
- Would contiguous (non-paged) TMA on the same math match FA4? If yes → gather/page feed; if no → MMA/softmax/schedule.
- Is `tile::gather4` still right here, or does this shape want cp.async coalesced (and did `FA4_CPASYNC` already lose for this choreography)?
- Are decode-short losers paying gather4 + TMA-desc setup that doesn’t amortize over few K-blocks (should stay on non-gather4 paged)?
- Page size 1 vs 128: does the gap track page fragmentation / gather efficiency vs FA4’s paged path?

### Tile geometry / tmem / registers (no CUTLASS allocator)
- Same effective tile as FA4 for this hd (Q rows, K-block N, pipeline stages, tmem S/O layout)?
- Is tmem S/O layout (e.g. S0/S1/O0/O1 512-col style) identical in capacity and banking, or are we spilling / serializing MMA vs softmax?
- Does widening stages or N-tile hit smem/tmem/register caps that CuteDSL’s allocator avoids — C7515-class pressure, forced stage drop, or wait bubbles?
- hd96 / non-multiple-of-64: are we on padded-128 math (known tax) while FA4 runs native width?
- hd192/dv128 path: is the decoupled K-depth-2 / V-depth-1 ring balanced on losers, or is one side starving?

### Algorithmic taxes vs FA4 (same math, worse CUDA)
- Softmax / rescale / mask between MMAs: do we burn issue slots FA4’s specialized softmax warps hide?
- Per-element mask on full tiles instead of range fast-out / early K-tile prune?
- P/scores round-trip through smem/tmem more times than FA4’s collective?
- Split-KV: worse split heuristic, slower combine, or extra partial buffers FA4 doesn’t need for this shape?
- Extra type converts or absences (fp8 paths not built yet — exclude; no silent bf16 upcast on the hot path)?

### Isolation experiments (CUDA-only, one variable)
- **Launcher-only A/B** (identical cubin): gap gone? → host/routing, stop kernel-diving.
- **Force FA4-route kernel** on the loser (if numerically safe): gap gone? → dispatch.
- **NCU stage split**: QK MMA vs PV MMA vs softmax/correction vs TMA/gather vs epilogue store — which stage’s cycles don’t match FA4?
- **Paged → dense TMA** on same problem size: gap gone? → KV feed.
- **Disable window/sink/softcap** at same shape: gap gone? → feature graft tax on `fa4`.
- **Stage depth / Q-preload on/off / gather4 vs cp.async**: which single toggle moves the ratio without wrecking decode?
- **Persistent vs non-persistent** on that grid: is long-context loss scheduling, not MMA?
