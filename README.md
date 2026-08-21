# fa3-cuda

A **dependency-free CUDA + inline-PTX re-implementation of the FlashAttention-3 forward kernel**
for NVIDIA **Hopper (sm90a)**. No CUTLASS / CuTe / cuBLAS / cuDNN — just `nvcc`, hand-written
`wgmma` / TMA / mbarrier PTX, and `torch` only at the host op boundary.

`kernels/fa3_hopper.cu` registers the exact vLLM `_vllm_fa3_C` op schema (`fwd` +
`get_scheduler_metadata`) so it can A/B against — and drop in for — real FA3. It covers a broad
surface: bf16 / fp16 / fp8-e4m3, paged + non-paged KV, GQA/MQA, causal, sliding-window, softcap,
varlen, decode + split-KV, `head_dim_v != head_dim` (prefill + decode, hd=192/hdv=128), attention
sink (`s_aux`), `cache_leftpad`, and fully-masked (causal q≫k) rows. **All 58 scenarios are
numerically correct.** One scenario falls short of the 0.92× perf bar (`hd256_causal` 0.84×) — a
hand-written-kernel architectural limit, analysed under
[Perf-limited scenario](#perf-limited-scenario).

> **Blackwell (SM100):** a separate from-scratch port, `kernels/fa4_blackwell.cu`, targets the FA4 /
> CUTLASS `77_blackwell_fmha` design (`tcgen05` 5th-gen tensor cores, `tmem`, TMA, warp
> specialization; still no CUTLASS/CuTe) and is A/B'd against the upstream Dao-AILab **FA4**
> CuteDSL oracle (`flash-attn-4`). See [Blackwell (SM100) FA4 port](#blackwell-sm100-fa4-port).

Requires CUDA ≥ 12.3, a Hopper GPU, and a `torch` + vLLM (ships FA3 for sm90) install.

## Build / run / test

Everything needed lives in this repo (`bench/`). The harness compiles `fa3_hopper.cu` into a torch
`.so` (via `torch.utils.cpp_extension` — no manual `nvcc`), checks correctness vs FA3 + an fp32
reference, and benchmarks each scenario against `_vllm_fa3_C`.

```bash
./setup.sh                 # torch + FA4 CuteDSL baseline + cuDNN frontend (needs CUDA 13 + GPU)
./setup.sh --with-vllm     # also install vLLM for the Hopper FA3 baseline

python -m bench.eval --target hopper                 # build + correctness + per-scenario speedup
python -m bench.eval --target hopper --no-perf       # build + correctness only
python -m bench.eval --target hopper --configs attn_dense_hd128_causal,attn_dense_hd128_fp8   # a subset
python -m bench.eval --target hopper --gpus 0,1      # restrict to these GPUs
python -m bench.eval --target hopper --gpu 1         # a single GPU (serial, in-process)
```

Scenarios are scheduled across **every visible GPU by default**, one worker process per GPU pulling
from a shared queue — so a GPU picks up the next scenario as soon as it frees up. At most one
scenario runs per GPU at a time, and a scenario's baseline and candidate are timed back-to-back on
that same GPU, so an A/B pair never spans devices or shares one. `--gpu N` restores the old
single-GPU serial path; both paths evaluate a scenario through the same code, so parallelism changes
only where and when work runs, never what is measured.

`bench/scenarios.py` defines the 58 scenarios and builds the inputs + fp32 reference, and
`bench/eval.py` is the build + A/B driver for both targets (`--target hopper|blackwell`, required).
(If vLLM isn't installed, correctness falls back to an absolute fp32-ref tolerance and perf is
skipped.)

Optional standalone debug micro-tools (compile directly, NCU-able, no torch):

```bash
A="-gencode arch=compute_90a,code=sm_90a"
nvcc -O3 -std=c++17 $A kernels/probes/mainloop_micro.cu -o /tmp/mlmicro && /tmp/mlmicro   # consumer mainloop
nvcc -O3 -std=c++17 $A kernels/probes/fp8_probe.cu      -o /tmp/fp8probe && /tmp/fp8probe  # fp8 QK/PV path
```

To NCU-profile one scenario on the built op (e.g. `ncu --set detailed python -m bench.microbenchmarks.prof_one`),
`bench/microbenchmarks/prof_one.py` loads a prebuilt `.so` and runs a single config; select it via the `FA3_CFG`,
`FA3_NS`, `FA3_ITERS`, `FA3_GPU` env vars.

## Per-scenario speedup vs FA3 (and vs cuDNN)

Every run measures **three** implementations per scenario — `fa3_hopper.cu`, real FA3
(`_vllm_fa3_C`), and cuDNN 9.19's fused SDPA — from byte-identical inputs, timed back-to-back on one
GPU. Speedup = FA3 time / that implementation's time (>1 = faster than FA3), so FA3 is 1.0 by
construction. Idle H200, warmup 300 + 100×5 L2-flushed CUDA-event timing. Scenarios mirror
FlashAttention-3's own forward pytest grids (`hopper/test_flash_attn.py` dense + varlen,
`hopper/test_attn_kvcache.py` kvcache/split-KV), plus an extended set (+18) that stresses dtype and
feature breadth (section E of `bench/scenarios.py`).

**All 105 scenarios correct · ours geomean 1.153× · median 1.164× · 95/105 ≥0.95× · 100/105 ≥0.92×.**
The seven previously-broken/slow extended scenarios (attention sink dense+varlen, `cache_leftpad`,
dense+varlen `head_dim_v != head_dim`, fully-masked `q≫k`, and hd64 fp8 decode) are now all correct
and ≥0.92×. Five sit under 0.92× — `edge_causal_q2048k2048` 0.88×, `real_llama8b_pf_s2k` 0.91×,
`dense_hd256_fp8` 0.91×, `dense_hd192_dv128` 0.92×, `dense_hd128_seq8k` 0.92× — the hd256/hd192
family being a hand-written-kernel register-pressure limit (see
[Perf-limited scenario](#perf-limited-scenario)).

cuDNN is a **measured reference, not a correctness gate**: it is timed and sanity-checked against the
fp32 reference, but the OK/FAIL verdict stays ours-vs-FA3. It gets the *same* packed and paged
buffers our kernel reads — varlen via `set_ragged_offset`, a paged cache as a cuDNN container plus
page table — so no implementation is handed a layout advantage. Over the full 105-config suite cuDNN
runs on **81**, geomean **0.829× vs FA3**, and **we are faster than cuDNN on 54 of those 81**: cuDNN
wins the dense-prefill envelope (up to 1.55×) and loses heavily on paged decode (down to 0.15×). The
24 it cannot express are 14 `max_s_q > max_s_kv` (rejected by cuDNN), 7 fp8 (`sdpa_fp8` is wired up
but **Blackwell-only** — it validates on sm90 then refuses to build a plan), 2 softcap (`score_mod`
is mutually exclusive with the mask subgraphs), and 1 `cache_leftpad` (no equivalent argument); those
scenarios simply get no cuDNN bar. `--no-cudnn` drops the column entirely. FlashInfer is wired
the same way as a fourth measured baseline (`--no-flashinfer` / `FA_NO_FLASHINFER=1`); native
ragged/paged paths only (no leftpad, asymmetric `window_right`, or per-head FP8 descales).

![fa3_hopper.cu vs FA3 vs cuDNN vs FlashInfer](docs/perf_fa3_hopper.png)

> Regenerate the chart with `python -m bench.plot` (snapshot data in `bench/plot.py`).

### A) dense `flash_attn_func` (non-paged)

| Scenario | Shape | Speedup | Note |
|---|---|--:|---|
| attn_dense_hd64_causal     | 16h · q1024×k1024            | 1.364× | non-paged TMA + persist; beats FA3 |
| attn_dense_hd64_fp16       | 16h · q1024×k1024 · fp16     | 1.334× | fp16 on the hd64 path |
| attn_dense_hd128_oddseq    | 16/2 · q113×k203             | 1.209× | small odd-masking shape; overhead win |
| attn_dense_hd128_splits3   | 16/2 · q1024×k1024 · split3  | 1.185× | non-paged + forced 3 splits; +ping-pong |
| attn_dense_hd128_kgtq      | 16h · q256×k512              | 1.116× | k>q (prefix/partial fill) |
| attn_dense_hd128_seq256    | 16h · q256×k256              | 1.083× | small square; beats FA3 |
| attn_dense_hd128_noncausal | 16h · q1024×k1024            | 1.052× | TMA-Q persist; full attention |
| attn_dense_hd128_deg_q239k1 | 16h · q239×k1               | 1.051× | causal q≫k; fully-masked m-tiles zeroed (was wrong) |
| attn_dense_hd128_local     | 16/2 · q1024×k1024 · win[256,0] | 1.035× | GQA TMA-Q + CN=64 + ping-pong; +mask fast-out |
| attn_dense_hd128_fp16      | 16h · q1024×k1024 · fp16     | 1.031× | shares the bf16 path |
| attn_dense_hd128_causal    | 16h · q1024×k1024            | 1.028× | TMA-Q persist + dyn-sched + inter-WG ping-pong |
| attn_dense_hd128_fp8       | 16h · q1024×k1024 · fp8      | 1.009× | fp8 persist; K + V-stage via TMA |
| attn_dense_hd128_mqa       | 16/1 MQA · q1024×k1024       | 1.001× | GQA TMA-Q (3D box) persist + ping-pong |
| attn_dense_hd128_gqa       | 16/2 GQA · q1024×k1024       | 1.000× | GQA TMA-Q (3D box) persist + ping-pong |
| attn_dense_hd128_local_noncausal | 16h · q1024×k1024 · win[256,256] | 0.999× | symmetric sliding window, non-causal |
| attn_dense_hd128_sink      | 16h · q1024×k1024 · s_aux    | 0.992× | attention sink folded into the denom (was wrong 495×) |
| attn_dense_hd96_causal     | 16h · q1024×k1024            | 0.981× | hd96 not %64 → cp.async (no TMA) |
| attn_dense_hd128_seq16k    | 16h · q16384×k16384         | 0.958× | long-context prefill |
| attn_dense_hd128_softcap_gqa | 16/2 GQA · q1024×k1024 · softcap=15 | 0.954× | softcap + GQA |
| attn_dense_hd128_seq8k     | 16h · q8192×k8192           | 0.946× | long-context prefill |
| attn_dense_hd192_fp16      | 16h · q1024×k1024 · fp16     | 0.946× | fp16 on the hd192 path |
| attn_dense_hd128_fp8_gqa   | 16/2 GQA · q1024×k1024 · fp8 | 0.945× | fp8 + GQA |
| attn_dense_hd192_causal    | 16h · q1024×k1024            | 0.939× | TMA-Q persist; CN=64@STAGES=3; +mask fast-out |
| attn_dense_hd128_softcap   | 16h · q1024×k1024 · softcap=15 | 0.934× | TMA persist + EXTRA; mask→[lo,hi] fast-out |
| attn_dense_hd128_large4k   | 16h · q4096×k4096            | 0.928× | persist (1 wave); +mask fast-out on full tiles |
| attn_dense_hd192_dv128     | 16h · q1024×k1024 · hdv=128  | 0.921× | hd≠hdv prefill (FA3 hdimdiff); CN=64/STAGES=4 + ping-pong (was unsupported) |
| attn_dense_hd256_causal    | 16h · q1024×k1024            | 0.843× | register-bound: acc_o[128] caps occupancy at 1 CTA/SM; see Perf-limited |

### B) varlen `flash_attn_varlen_func` (ragged)

| Scenario | Shape | Speedup | Note |
|---|---|--:|---|
| attn_varlen_hd64_causal    | 6h · ragged q≤256    | 1.484× | ragged varlen hd64; biggest win |
| attn_varlen_hd128_fp8      | 6/2 · ragged q≤256 · fp8 | 1.447× | ragged varlen fp8 |
| attn_varlen_hd128_causal   | 6/2 · ragged q≤256   | 1.335× | ragged varlen; overhead win |
| attn_varlen_hd128_mqa      | 6/1 MQA · ragged q≤256 | 1.321× | ragged varlen MQA |
| attn_varlen_hd128_noncausal| 6/2 · ragged q≤256   | 1.309× | ragged varlen, full attention |
| attn_varlen_hd192_dv128    | 6/2 · ragged q≤256 · hdv=128 | 1.307× | varlen hd≠hdv prefill (was unsupported) |
| attn_varlen_hd128_local    | 6/2 · ragged q≤256 · win[128,0] | 1.278× | sliding-window varlen |
| attn_varlen_hd128_sink     | 6/2 · ragged q≤256 · s_aux | 1.036× | attention sink, varlen (was wrong 395×) |

### C) kvcache decode / split-KV (paged)

| Scenario | Shape | Speedup | Note |
|---|---|--:|---|
| attn_kv_hd128_dec1024        | 6/2 · q3×k1024 · paged        | 1.288× | spec decode q=3 |
| attn_kv_hd128_decode_mqa32   | 32/1 MQA · q1×k1024 · paged   | 1.281× | rep=32 (q·rep=32) → split-KV decode |
| attn_kv_hd128_leftpad        | 6/2 · q1×k2048 · leftpad=128  | 1.249× | cache_leftpad as absolute key lower-bound (was wrong 103×) |
| attn_kv_hd128_local_decode   | 8/2 · q1×k4096 · win[1024,0]  | 1.223× | sliding-window decode |
| attn_kv_hd128_decode_gqa5    | 10/2 · q1×k3131 · paged       | 1.198× | decode GQA ratio 5 |
| attn_kv_hd64_decode_gqa3     | 9/3 · q1×k3131 · paged        | 1.196× | decode hd64 GQA ratio 3 |
| attn_kv_hd128_dec339         | 6/2 · q1×k339 · paged         | 1.190× | short-context decode |
| attn_kv_hd128_page128        | 6/2 · q1×k2048 · page_size=128 | 1.164× | block_size 128 |
| attn_kv_hd128_decode_noncausal | 8/2 · q1×k3131 · paged      | 1.157× | decode, non-causal |
| attn_kv_hd128_page1          | 6/2 · q1×k2048 · page_size=1  | 1.134× | block_size 1 |
| attn_kv_hd128_dec800         | 6/2 · q64×k800 · paged        | 1.133× | q=64 → prefill fast path |
| attn_kv_hd128_page4          | 6/2 · q1×k2048 · page_size=4  | 1.131× | block_size 4 |
| attn_kv_hd256_split_auto     | 9/3 · q1×k4000 · split-auto   | 1.114× | hd256 auto split-KV |
| attn_kv_hd128_fp16_decode    | 16/8 · q1×k4000 · paged · fp16 | 1.113× | fp16 paged decode |
| attn_kv_hd128_decode_qlen8   | 16/8 · q8×k4224 · paged       | 1.106× | medium-q q=8 → split-KV decode |
| attn_kv_hd128_decode_mha     | 8/8 · q1×k4224 · paged        | 1.091× | decode MHA |
| attn_kv_hd256_decode_mqa16   | 16/1 · q1×k4224 · paged       | 1.060× | decode hd256 rep=16 |
| attn_kv_hd128_ctx20k         | 6/2 · q16×k20000 · paged      | 1.054× | q=16/k=20000 → split-KV decode |
| attn_kv_hd128_split2         | 16/8 · q1×k5555 · split2      | 1.023× | forced 2 KV-splits |
| attn_kv_hd64_fp8_decode      | 16/8 · q1×k4000 · paged · fp8 | 0.978× | native e4m3 wgmma hd64 (HDQK=128-padded koff8 tile); was 0.297× |
| attn_kv_hd128_fp8_decode     | 16/8 · q1×k4000 · paged · fp8 | 0.973× | native e4m3 wgmma decode (no convert pass) |
| attn_kv_hd128_decode_qlen120 | 16/8 · q120×k4224 · paged     | 0.948× | underfill→split-KV decode (CM=64, STAGES=5) |
| attn_kv_hd128_prefill3577    | 6/2 · q2048×k3577 · paged     | 0.937× | paged chunked-prefill; +mask fast-out |

### Resolved gaps

The seven extended scenarios that the earlier revision could not match (or ran slowly) are now
**correct and ≥0.92×**:

| Scenario | Was | Now | Fix |
|---|---|--:|---|
| attn_dense_hd192_dv128  | unsupported | 0.921× | prefill `hd!=hdv` dispatch (the kernel already templated HD/HDV separately for decode); CN=64/STAGES=4 + inter-WG ping-pong |
| attn_varlen_hd192_dv128 | unsupported | 1.307× | same dispatch (varlen per-tile path) |
| attn_dense_hd128_sink   | wrong (495×) | 0.992× | `s_aux` folded into the softmax denominator at the epilogue (FA3 `finalize_aux`: `l += exp2(log2e·s_aux − max·s2)`) |
| attn_varlen_hd128_sink  | wrong (395×) | 1.036× | same |
| attn_kv_hd128_leftpad   | wrong (103×) | 1.249× | `cache_leftpad` as an absolute key lower-bound in the mask + leading-tile pruning |
| attn_dense_hd128_deg_q239k1 | wrong (bias 8e-2) | 1.051× | zero O + set LSE=−inf for fully-masked m-tiles that early-out before the mainloop (`emit_zero_o`) |
| attn_kv_hd64_fp8_decode | 0.297× | 0.978× | native e4m3 `wgmma` for hd64: pad the K/Q smem tiles to HDQK=128 so `koff8`'s `row*128` swizzle stride fits, load the real 64 cols via cp.async, contract `qk_fp8_ss<64>` over the real HD (the unused upper half is never read). Was blocked because the 128B-swizzle `koff8` overran the 64-wide tile (OOB); the fp8→bf16 cast workaround was LSU-bound at ~14µs |

### Perf-limited scenario

One correct scenario remains below 0.92×, an **architectural limit of the no-CUTLASS hand-written
kernel**, confirmed by NCU profiling and direct experiment:

**`hd256_causal` — 0.84× (register-capped).** `head_dim_v=256` forces a 128-register output
accumulator (`acc_o[HDVsm/2]`) per consumer thread → only **1 CTA/SM** (2 CTAs would need ≤85
regs/thread; `acc_o` alone is 128), so achieved occupancy is ~18% (3 warps/scheduler). NCU shows the
kernel is **barrier/long-scoreboard-bound** (the lone CTA's consumer starves waiting for K/V), not
tensor-bound (44%). Every hand-written lever was tried and hit a wall:
- The contiguous-TMA 128B swizzle only composes at `CN ∈ {48,64,128}` (CN=32/72/80 corrupt → NaN), so
  the tile can't widen toward FA3's CN=80.
- `CN=64` at CM=128 either overflows the consumer (ptxas C7512 wgmma serialization) or forces a
  `STAGES=2` ring that **deadlocks** the persistent continuous ring; a 240-reg rebalance hits the
  exact-65536 register edge and also deadlocks.
- The **CM=64 path** (half-height Q frees smem for a deeper STAGES=4 ring) is now correct (a `tiles_m`
  bug that processed only half the rows was fixed) but measures **0.72×**: dropping to one consumer
  warpgroup costs 8-vs-12 warps/SM, and the deeper ring can't recover it — **occupancy, not ring
  depth, is the binding constraint**, and `acc_o[128]` hard-caps occupancy at 1 CTA/SM.
- A **head-dim-split** (two consumer WGs on the same rows, each owning half of HDV at `acc_o[64]`)
  would restore 12 warps + STAGES=4, but pays either a 1.5× redundant-QK cost (which by the 44%-tensor
  arithmetic does not net beat 0.85×) or a complex asymmetric P-via-smem rewrite.

FA3 reaches 0.92×+ at CN=80 only because CUTLASS's register allocator schedules that tile without
serializing — exactly the co-design the project's hard `no CUTLASS/CuTe` constraint forbids.

## Blackwell (SM100) FA4 port

`kernels/fa4_blackwell.cu` is a from-scratch, dependency-free hand-CUDA + inline-PTX FlashAttention
forward for **Blackwell (SM100 / `compute_100a`)**, built on `tcgen05` 5th-gen tensor-core MMAs,
tensor memory (`tmem`), TMA, and mbarrier-driven warp specialization — still no CUTLASS / CuTe / cuBLAS.
It carries two prefill kernels with **automatic config-aware routing**: an inline 2-tile kernel
(`fwd_kernel_ws2`) for masked / softcap / sink / small-seqlen / varlen / decode configs, and a
**FA4-faithful** warp-specialized kernel (`fwd_kernel_fa4`) — one 256-row Q tile as two stacked
128-row halves sharing K/V, with dedicated softmax0/1 / correction / MMA / load / epilogue warpgroups,
a 9-stage mbarrier pipeline, and persistent + LPT continuous-overlap scheduling that mirrors FA4's own
collective structure — auto-selected for hd128 dense large-seqlen causal / noncausal / GQA / MQA. It
registers the same op schema as the Hopper kernel and is A/B'd against the upstream Dao-AILab
**FA4** oracle (`flash_attn.cute` / `flash-attn-4`, the CuteDSL FA4 kernel), so the target here is
much stiffer than FA3.

**48 / 59 scenarios run and are all numerically correct** (vs FA4 + the fp32 reference); the other
10 — every fp8 path and every `head_dim ≥ 192` — are not built in this candidate yet. Speed =
FA4 time / our time, so `1.0×` is parity with the FA4 oracle. **The hd128 dense-prefill envelope now
matches/beats FA4** — short-seqlen causal/GQA/MQA/fp16/noncausal at **1.02–1.13×**, dense-envelope
geomean **≈1.0×** — in dependency-free CUDA + inline PTX.

```bash
python -m bench.eval --target blackwell            # build fa4_blackwell.cu + correctness + perf vs FA4
python -m bench.eval --target blackwell --configs attn_dense_hd128_causal
python -m bench.eval --target blackwell --set edge --no-perf   # dense seqlen/masking grid (FA4-derived)
python -m bench.eval --target blackwell --set real             # production shapes (SOL-ExecBench)
python -m bench.plot --arch blackwell    # regenerate the chart below (snapshot in bench/plot.py)
```

Beyond the headline `attntest` set, `--set` selects extra coverage: **`edge`** — a dense (seqlen_q,
seqlen_k) masking/boundary grid mirrored from FA4's own cute pytest; **`real`** — production GQA
shapes from [nvidia/SOL-ExecBench](https://huggingface.co/datasets/nvidia/SOL-ExecBench) (Llama-3.1-8B
h32/kv8/d128, Qwen3-30B-A3B h32/kv4/d128; decode + causal/ragged prefill); **`all`** — the union;
**`known_bad`** — shapes surfaced as `fwd_kernel_ws2` bugs. The `edge` grid caught several: `q512k256`
(wrong output, err 576× vs the FA4 oracle) and causal q≫k with a single K-block (`q799k3`, `q640k128`)
all **hung or mis-computed on ws2** — now **fixed** by routing uniform-dense sub-wave grids to
`fwd_kernel_fa4` (which is correct on them and, being tuned, runs 1.5–1.8×). The only remaining
`known_bad` entry is an h32 varlen-ragged shape that deadlocks on ws2 — deferred, since sub-wave varlen
must stay on ws2 (persistent fa4 has no amortization headroom there); it's excluded from the green sets
so default runs don't wedge. `FA4_NO_ORACLE=1` skips the (slow-to-JIT) reference for fast fp32-ref-only
correctness sweeps.

The chart also includes 6 **real production shapes** captured in [nvidia/SOL-ExecBench](https://huggingface.co/datasets/nvidia/SOL-ExecBench)
(`--set real`; geomean **0.86×**): we **beat FA4 on short-context decode** — the highest-volume serving op — (Llama-3.1-8B
**1.34×**, Qwen3-30B **1.29×** at k=512), stay competitive on long-context decode (0.73–0.92× at k=4096), and trail on
**paged prefill** (0.56–0.64×, lifted from 0.44–0.50× by Tier-2 below).

![fa4_blackwell.cu vs FA4 vs cuDNN vs FlashInfer](docs/perf_fa4_blackwell.png)

> The Blackwell chart has no cuDNN bars yet: cuDNN is measured on every run, but these numbers were
> captured before that landed and regenerating them needs SM100 hardware. Re-run
> `python -m bench.eval --target blackwell --set all` on a B200 to fill them in.

| Group | Runnable | Geomean vs FA4 | Notes |
|---|--:|--:|---|
| C) kvcache decode / split-KV (`attn_kv_*`) | 18 | 1.14× | nearly all at/above parity (`dec1024` 1.41×, `page*` 1.43×, `mqa32` 1.42×, `hd64_decode_gqa3` 1.50×); residual laggards are the paged prefills — `prefill3577` 0.67× (large-q paged prefill on `fwd_kernel_fa4_paged`, lifted from 0.55× by Tier-2), `ctx20k` 0.79×, `decode_qlen120` 0.69× |
| B) varlen `flash_attn_varlen_func` (`attn_varlen_*`) | 7 | 1.30× | all **beat FA4** (1.09–1.47×) — the ragged shapes are sub-single-wave and were pinned by a redundant host D2H, not a kernel gap (see the host/routing sweep below) |
| A) dense `flash_attn_func` (`attn_dense_*`) | 23 | 1.22× | hd128/hd192/hd64 dense all **beat FA4** (causal 1.09×, GQA 1.12×, MQA 1.13×, fp16 1.10×, hd192_dv128 1.06×, **hd64 1.29×**, small-q seq256/oddseq/kgtq 1.33–1.46×); sliding-window `local` **1.04×** + `sink` **1.02×** now beat FA4 (grafted onto `fwd_kernel_fa4`, below); `softcap`/`splits3` >1× are FA4-oracle-slow paths; residual sub-1.0 = `hd96` (0.84×, padded-128 kernel — native-96 would beat it), `local_noncausal` (0.99×, ~parity), and dense long-context nc4k/large4k/seq8k/seq16k (0.83–0.94×, the shared CuteDSL-scheduling edge) |
| D) real production (`attn_real_*`, [SOL-ExecBench](https://huggingface.co/datasets/nvidia/SOL-ExecBench)) | 6 | 0.86× | Llama-3.1-8B (h32/kv8) + Qwen3-30B-A3B (h32/kv4), d128, page_size 1: short-ctx decode **1.29–1.34×** (beat FA4), long-ctx decode 0.73–0.92×, paged prefill 0.56–0.64× (Tier-2) |
| **Overall** | **54** | **1.161×** | median **1.20×** · **40 configs ≥1.0×** · 48 attntest-runnable (10 fp8/hd≥192 not built) + 6 real; 0 wrong (`attn_kv_hd128_leftpad` is correct but has no FA4 oracle, so no ratio) |

> **2026-08 pass — see [docs/blackwell_gap_analysis.md](docs/blackwell_gap_analysis.md).** All
> **154/154 runnable scenarios are numerically correct** (was 2 FAIL) and the suite geomean is
> **1.412×** vs the upstream FA4 oracle (was 1.234×), min **0.551×** (was 0.277×), **114/153 at or
> above parity** (was 98/151). Landed: a page-aligned TMA paged-KV producer (8 TMA boxes per tile
> instead of ~4096 `cp.async`), a 256-thread spill-free paged CTA, K/V ring depth 3, degenerate
> sliding-window normalization, a ragged-batch empty-tile skip, mixed-batch splitting by query
> length, paged prefill moved onto the persistent `fwd_kernel_fa4`, folding the Q scale into the
> softmax exp2 (fixes both FAILs), and de-spilling the runtime-indexed ring state (local-memory
> traffic 38.6M → 0.72M sectors). Still open, with root causes measured: page_size=1 paged
> (0.55–0.82×, dependent-load latency in a 2-deep ring), dense long context (0.76–0.84×, softmax
> `ex2` on the XU pipe co-limiting with the MMAs at 56% each), and hd96 (0.79×, padded-128 path).
> That document also lists the five remaining differences from FA4's own SM100 pipeline.

A recon-guided **host/routing sweep** took the overall geomean **0.879×→1.15×** (18→**36** configs ≥1.0×, **0 regressions**) via ~15 lines of launcher code — no kernel bodies changed (every kernel register-identical). NCU-at-locked-clocks showed most sub-0.8× configs were *not* kernel-bound: the same ~20–38 µs/call synchronous-host-D2H that the dense-parity fix removed had never been propagated to the `ws2` fall-through or the paged/serial launchers, and a `max_mq>=4` routing gate had gone stale (it predated the counter-self-reset fix, so `fwd_kernel_fa4` now wins down to sub-wave grids). The fixes: route hd64 + small-regular-dense to the already-correct `fwd_kernel_fa4` (grid-fill gate, keeping sub-wave varlen on `ws2`); guard the redundant D2H; thread `max_seqlen` into the paged/serial launchers (this also nudged `prefill3577` 0.51→0.55). (Honest caveat: on short kernels this is *launcher* efficiency — the wall-clock win partly reflects FA4's own Python/CuteDSL dispatch overhead; kernel-only ratios there are unchanged, so the true kernel gaps remain on the residual sub-1.0 configs below.) A follow-on **kernel port** then grafted **sliding-window, attention-sink, and hd96** onto `fwd_kernel_fa4` behind `if constexpr` template bools (`WINDOW`/`SINK`/`PADQ`) — the default instantiation stays **SASS-byte-identical** (envelope provably untouched), while `local` 0.71→**1.04×**, `sink` 0.74→**1.02×**, `local_noncausal` 0.66→0.99×, and `hd96` 0.73→0.84× (the padded-128 path does 128-wide QK/PV so a native-96 kernel would be needed to beat it). Overall geomean **→1.20×, 38/48 ≥1.0×**.




The dense-prefill envelope now **matches/beats FA4** (short-seqlen 1.02–1.13×, envelope geomean ≈1.0), reached in two
steps. First, a from-scratch replica of FA4's warp-specialized collective (`fwd_kernel_fa4`: persistent + LPT
scheduling, dual stacked-half softmax with an `OrderedSequenceBarrier` ping-pong, correction + epilogue warpgroups,
continuous cross-tile overlap), tuned by an evolutionary search from a 0.29× naive port to ~0.78×. Then the decisive
step: `ncu` at locked clocks showed the kernel was *already faster than the CUTLASS kernel* (15% fewer cycles, 14%
fewer instructions on causal) — the remaining wall-clock gap was **~38 µs/call of synchronous host launch overhead**
(two `cu_seqlens.to(cpu)` copies that drain the stream, plus a per-call scheduler-counter `at::full`), not a kernel
deficit. Removing it (compute `max_seqlen` from op args; device-side `atomicExch` counter self-reset) took the dense
envelope from ~0.78× to **≈1.0×**. The remaining sub-1.0 dense configs are long-context (nc4k/large4k/seq8k/seq16k
~0.83–0.94×), where a residual CuteDSL-scheduling edge (shared by CUTLASS) persists, plus `hd96` (0.84×, the padded-128
fa4 path — a native-96 kernel would be needed to beat it). Separately, `hd192_dv128` (asymmetric head
dim, hd=192/hdv=128) was lifted **0.21×→1.06× — beating FA4** by porting the fast persistent `fwd_kernel_fa4` to
hd192 (`fwd_kernel_fa4dk`, with a decoupled K-depth-2 / V-depth-1 ring + a small smem reclaim to fit the cap); its
tmem S/O layout matches hd128, so only the wider 192-dim Q/K needed the ring split (varlen hd192 stays on `ws2` at
0.69×). Among the paged configs, chunked-prefill (`prefill3577`) was lifted **0.31×→0.51×→0.67×** in two steps.
First, giving the paged path the treatment that won dense parity: a new `fwd_kernel_fa4_paged` grafts the
`fwd_kernel_fa4` intra-CTA choreography (deep software pipeline, S0/S1/O0/O1 512-col tmem layout, softmax warp
specialization) onto the paged gather, fed by a **hardware-async TMA `tile::gather4`** loader that frees ~35% of
issue slots — `ncu` confirms the fa4 consumer *converts* those freed slots where the old `ws2_paged` consumer could
not. Then **Tier-2**: `ncu` showed the paged prefills are **KV-feed-latency-bound** (long_scoreboard ~7 vs FA4's ~2,
MMA-pipe ~18% vs ~45%; DRAM <5% — *not* bandwidth), so the fix was latency-hiding, not a bandwidth rewrite —
async `cp.async` Q-preload + resolving the page-row pointers **once** off the per-block critical path lifted the three
large-q paged prefills ~+28% (prefill3577 0.55→0.67, Llama-3.1-8B 0.50→0.64, Qwen3-30B 0.44→0.56). A full
`cp.async`-coalesced producer (replacing `tile::gather4`) was tried and **lost** — at this choreography the HW gather
engine hides KV-feed latency better — so it's kept behind `FA4_CPASYNC=1` and gather4 remains the default. The
`tile::gather4` loader is scoped to
`fwd_kernel_fa4_paged` only — it's a net loss on short paged-decode (the 128 gather4 issues + per-launch TMA-descriptor
build don't amortize over few K-blocks), so it's compiled out of `ws_paged`/`ws2_paged` (verified register-identical to
the pre-bank decode kernels). The two single-tile paged shapes — short-q
`decode_qlen120` (0.67×) and long-ctx `ctx20k` (0.78×) — don't map onto the 2-tile fa4 structure and are the
residual laggards; notably this is **not** an occupancy limit (the 256-thread / 2-CTA-per-SM `SD==1` path regressed
0.46→0.17, and FA4 itself runs ~1 CTA/SM here — the gap is choreography, not occupancy). Every runnable scenario is **deadlock-free**: a boundary-warp hang in the
2-tile prefill kernel (`fwd_kernel_ws2`) — a warp-convergent `tcgen05.wait::ld.sync.aligned` issued
inside a `qr < seqlen_q` divergent branch, which stalls forever whenever `seqlen_q % 32 != 0` — was
fixed by running the tmem read/wait on the full warp and predicating only the global stores. That
removed all 8 previously-hanging shapes (every ragged-varlen config plus the `oddseq`/`deg_q239k1`
degenerate dense shapes).

## Layout

```
kernels/
  fa3_hopper.cu       # FA3 forward torch op — Hopper / sm90a (vs FA3)
  fa4_blackwell.cu    # FA4 forward torch op — Blackwell / sm100a, tcgen05 + tmem (vs the FA4 oracle)
  probes/             # standalone, torch-free debug tools (compile with nvcc, NCU-able)
    mainloop_micro.cu #   consumer-mainloop microbenchmark
    fp8_probe.cu      #   fp8 e4m3 wgmma QK/PV validation
    fa4_*.cu          #   Blackwell tcgen05 / TMA / tmem / warp-spec micro-probes
    probe_common.cuh  #   shared host helpers for the probes
bench/
  scenarios.py        # the 58 attntest scenarios + build/tolerance constants,
                      #   deterministic inputs + pure-fp32 reference attention
  eval.py             # build (cpp_extension) + correctness + perf A/B, both targets:
                      #   --target hopper    -> fa3_hopper.cu    vs FA3 (_vllm_fa3_C)
                      #   --target blackwell -> fa4_blackwell.cu vs the FA4 CuteDSL oracle
                      #   (the oracle adapter lives here too)
  plot.py             # render the per-scenario charts (--arch hopper|blackwell|both)
  microbenchmarks/    # profiling / sweep tools the two evaluators do NOT import
    prof_one.py       #   NCU/nsys: prebuilt .so, one attntest config (FA3_CFG/FA3_NS/FA3_ITERS)
    prof_real.py      #   NCU/nsys: prebuilt .so, one config from any set
    prof_cand.py      #   NCU/nsys: the Blackwell candidate in isolation
    prof_fa3.py       #   NCU/nsys: the FA3 baseline (_vllm_fa3_C), for side-by-side
    prof_fa4.py       #   NCU/nsys: the FA4 oracle
    prof_oracle.py    #   NCU/nsys: the FA4 oracle (nsys-friendly variant)
    ncu_probe.py      #   build fa4_blackwell.cu + warm one config, then a single profiled fwd
    sweep_splits.py   #   sweep num_splits on one decode config vs the FA3 baseline
    dump_ref_ptx.py   #   dump the CuteDSL SM100 forward PTX/SASS/CUBIN (port ground truth)
docs/                 # per-scenario speedup charts (Hopper vs FA3, Blackwell vs FA4)
reference/            # FA3 + CUTLASS source (submodules), CUDA/PTX/NCU docs, agent chats (see reference/README.md)
setup.sh              # one-shot env install (torch + FA4 CuteDSL + cutlass-dsl[cu13])
```

Fetch the reference source submodules (FA3, CUTLASS) on a fresh clone with
`git submodule update --init reference/cutlass reference/flash-attention`, then
`./setup.sh` (installs torch, a known-good `nvidia-cutlass-dsl[cu13]`, and FA4 CuteDSL
from `reference/flash-attention/flash_attn/cute` without the broken `.dev0` cutlass pin).
Hopper FA3 baseline: `./setup.sh --with-vllm`.

The repo is self-contained: `bench/` builds, verifies, and benchmarks the kernel with no external
tooling — for Hopper the optional outside dependency is the FA3 baseline (`_vllm_fa3_C` from
vLLM). See `.claude/skills/distill-kernel/references/hopper-fa3.md` for design notes and the
FA3→CUDA distillation methodology (and `portability.md` there for what carries over to the
FA4/Blackwell port).
