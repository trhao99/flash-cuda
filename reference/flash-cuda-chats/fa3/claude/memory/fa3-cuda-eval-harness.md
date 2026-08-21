---
name: fa3-cuda-eval-harness
description: How to build/eval/profile the fa3-cuda kernel (fa3_hopper.cu) against the vLLM FA3 baseline
metadata: 
  node_type: memory
  type: reference
  originSessionId: 4eb62961-dfde-41eb-901f-29fb1c622894
  modified: 2026-07-22T02:10:52.792Z
---

Repo `/home/yak/fa3-cuda`. Kernel: `kernels/fa3_hopper.cu`. H200 x2 (sm90), CUDA 13.0, conda env `dev`.

Build + correctness + perf vs vLLM FA3 (`_vllm_fa3_C`):
- `python -m bench.eval --set real` (SOL-ExecBench prod shapes) | `--set attntest` (FA3 pytest grid) | `--set all`
- `--configs a,b,c` filters within the set; `--namespace <ns>` (writes `.fa3_trials/<ns>/<ns>.so`); `--gpu 0|1`; `--no-perf`.
- Build is ~200s (rebuilds whole TU). Correctness gate: err_ratio<=2x FA3 + allclose + lse + bias.
- Ratio >1.0 = faster than FA3. geomean/min/median/max printed at end.

Profile a prebuilt .so WITHOUT rebuild:
`FA3_NS=<ns> FA3_CFG=<cfg> FA3_ITERS=3 ncu -k "regex:prefill_kernel" -c 1 --section SpeedOfLight ... python -m bench.prof_real`
(prof_real.py loads `.fa3_trials/<ns>/<ns>.so`). For FA3 baseline: import vllm.vllm_flash_attn._vllm_fa3_C and call C.BASELINE_NS; skip past make_inputs' randn/arange setup kernels with `ncu -s <N>`.

Target real-shape configs: attn_real_{llama8b,qwen3moe}_{dec_*,pf_*} (paged page_size=1, bf16, hd128, GQA 4:1 / 8:1) + attn_dense_hd256_fp8 + attn_kv_hd256_fp8_decode. See [[fa3-perf-gap-closing]].
