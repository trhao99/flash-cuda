---
name: fa3-page128-nan-fix
description: "Why fa3_kv_hd128_page128 produced NaN only in full-suite runs, and the combine_kernel fix"
metadata: 
  node_type: memory
  type: project
  originSessionId: c4987da0-9b92-4860-a389-7d95f1c5d56f
---

fa3_kv_hd128_page128 FAILed with NaN/inf ONLY when run in the full 40-scenario suite (passed in isolation). Root cause: split-KV `combine_kernel` (kernels/fa3_fwd.cu ~L2371) computed `acc += sw[s] * Opart[s]`. For an EMPTY split (num_splits not dividing the key-tile count → trailing split starts past the last tile), `emit_empty_lse` stamps LSE=-inf (→ weight sw[s]=0) but NEVER writes that split's `Opart` slice — it's uninitialized `at::empty`. Fresh alloc = clean zeros (isolation passes); after 30+ scenarios the caching allocator recycles dirty pages full of NaN → `0*NaN = NaN` corrupts output.

Fix: guard each combine term so a zero weight yields exactly 0 regardless of Opart (`acc += w!=0 ? w*Opart[s] : 0`). Always correct: non-empty splits always store finite Opart. The warp-path decode_kernel was never affected (it writes Opart=0 for empty splits).

See [[fa3-perf-baseline]].
