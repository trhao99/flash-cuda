# reference/

Read-only reference material for developing `fa3-cuda` / `fa4_blackwell`. Nothing here is built or
linked into the kernels — it exists so the FA3/FA4 implementations, CUTLASS, and the CUDA/PTX/NCU
docs can be browsed from within this repo.

## Submodules (source code, pinned commits)

| path | upstream | commit |
|---|---|---|
| `flash-attention/` | github.com/Dao-AILab/flash-attention | tip of `main` — FA3 (`hopper/`) source reference + FA4 CuteDSL (`flash_attn/cute/`, pip package `flash-attn-4`) used as the Blackwell A/B oracle |
| `cutlass/` | github.com/NVIDIA/cutlass | `1fc71b3` (v4.5.0-15) — CUTLASS/CuTe source the PTX recipes were traced from |
| `flashinfer/` | github.com/flashinfer-ai/flashinfer | tip of `main` (`968fa048`) — FlashInfer inference kernel library (attention / GEMM / MoE) source reference |

These are git submodules, so a fresh clone gets only pointers. Populate them with:

```bash
git submodule update --init reference/cutlass reference/flash-attention reference/flashinfer
```

`./setup.sh` installs the FA4 CuteDSL oracle from
`reference/flash-attention/flash_attn/cute` (CUDA 13 / B200). The Hopper FA3 *runtime*
baseline (`_vllm_fa3_C`) is optional (`./setup.sh --with-vllm`);
`reference/flash-attention/hopper/` is only the source to read.

## Agent transcripts (copied, not a submodule)

| path | what |
|---|---|
| `flash-cuda-chats/` | Cursor + Claude Code sessions that wrote/optimized `fa3_hopper.cu` and `fa4_blackwell.cu`. Catalog: `flash-cuda-chats/INDEX.md`. |

These are checked-in files (not a git submodule). Kernel sources stay in `kernels/`; this tree is transcripts, subagents, file-history, and distilled notes only.

## Docs (copied markdown)

| file | what |
|---|---|
| `FA3.md` | FlashAttention-3 design notes / repo helper |
| `FA4.md` | FlashAttention-4 (CuteDSL) distillation map |
| `cutlass.md` | CUTLASS documentation |
| `parallel-thread-execution.md` | PTX ISA reference |
| `cuda-programming-guide.md` | CUDA C++ programming guide |
| `ncu.md` | Nsight Compute (NCU) guide |
