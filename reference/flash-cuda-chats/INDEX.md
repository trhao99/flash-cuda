# Session index

Machine-readable: [`inventory.json`](inventory.json). Provenance: [`provenance.json`](provenance.json).

**16 parent sessions** — 12 FA3 (Hopper), 4 FA4 (Blackwell).

## FA3 (Hopper / sm90)

### Claude Code
- `4eb62961-dfde-41eb-901f-29fb1c622894` — **fa3_hopper.cu: faithful CUDA-only port matching FA3 output + perf**
  - claude, 19.01 MB, 78 user turns, 45 subagents
  - `fa3/claude/4eb62961-dfde-41eb-901f-29fb1c622894.jsonl`
  - tools: Agent, AskUserQuestion, Bash, Edit, Monitor, Read, TaskCreate, TaskList, TaskOutput, TaskStop, TaskUpdate, Workflow, Write
  - first prompt: A session-scoped Stop hook is now active with condition: "I'm writing a faithful cuda-only implementation of flash attention 3 in @kernels/fa3_hopper.cu , without any cutlass depen…
- `c4987da0-9b92-4860-a389-7d95f1c5d56f` — **fa3_fwd.cu: ≥0.90× vs vLLM FA3 on all bench configs**
  - claude, 6.36 MB, 8 user turns, 5 subagents
  - `fa3/claude/c4987da0-9b92-4860-a389-7d95f1c5d56f.jsonl`
  - tools: AskUserQuestion, Bash, Edit, Read, TaskCreate, TaskGet, TaskOutput, TaskStop, TaskUpdate, Workflow, Write
  - first prompt: A session-scoped Stop hook is now active with condition: "the fa3_fwd.cu kernel is at least >=0.90x, ideally 0.95x+ as fast as FA3 (flash attention 3, vllm's fork) in all scenarios…
- `ecfde6a3-dcfa-46bd-8e47-d9800e334d03` — **Close README gaps — full scenario support, correctness, ≥0.92×**
  - claude, 4.50 MB, 4 user turns, 0 subagents
  - `fa3/claude/ecfde6a3-dcfa-46bd-8e47-d9800e334d03.jsonl`
  - tools: AskUserQuestion, Bash, Edit, Monitor, Read, TaskCreate, TaskGet, TaskOutput, TaskStop, TaskUpdate, Write
  - first prompt: A session-scoped Stop hook is now active with condition: "Fix the gaps from @README.md to ensure that we support all scenarios, we achieve correctness on all scenarios, and that we…
- `72273405-e11c-4cb9-9918-4b29f9d124de` — **Are bench scenarios comprehensive vs SOL-ExecBench + FA3 tests?**
  - claude, 7.27 MB, 19 user turns, 18 subagents
  - `fa3/claude/72273405-e11c-4cb9-9918-4b29f9d124de.jsonl`
  - tools: Agent, AskUserQuestion, Bash, Edit, EnterPlanMode, ExitPlanMode, Read, TaskCreate, TaskList, TaskUpdate, Workflow, Write
  - first prompt: are the scenarios used to benchmark our @kernels/fa3_hopper.cu kernel comprehensive? do we cover all attention shapes/sizes/dtypes from the @../SOL-ExecBench/ trace/dataset? Do we …
- `17a633ac-acc7-4eef-b89e-5be3a7eff171` — **Package FA3 CUDA work + skill into a portable repo**
  - claude, 1.66 MB, 12 user turns, 4 subagents
  - `fa3/claude/17a633ac-acc7-4eef-b89e-5be3a7eff171.jsonl`
  - tools: Agent, AskUserQuestion, Bash, Edit, Read, TaskCreate, TaskUpdate, Write
  - first prompt: I just finished using cursor and claude code to write a cuda-only flash-attention 3 implementation in @fa3-cuda , matching the output and performance of the original flash attentio…
- `7fd6ced6-d91d-437d-8581-3314431c8ea9` — **Same packaging prompt (interrupted / retry)**
  - claude, 0.20 MB, 1 user turns, 3 subagents
  - `fa3/claude/7fd6ced6-d91d-437d-8581-3314431c8ea9.jsonl`
  - tools: Agent, AskUserQuestion, Bash
  - first prompt: I just finished using cursor and claude code to write a cuda-only flash-attention 3 implementation in @fa3-cuda , matching the output and performance of the original flash attentio…
- `f221d216-f8ea-4984-b19f-f0e8234767f3` — **Same packaging prompt (short session)**
  - claude, 0.04 MB, 2 user turns, 0 subagents
  - `fa3/claude/f221d216-f8ea-4984-b19f-f0e8234767f3.jsonl`
  - tools: Bash
  - first prompt: I just finished using cursor and claude code to write a cuda-only flash-attention 3 implementation in @fa3-cuda , matching the output and performance of the original flash attentio…

### Cursor
- `69fff5d7-d194-4a1c-9ec8-de9efe54b617` — **FA3 agent draft + CUDA-only kernel that matches FA3**
  - cursor, 7.87 MB, 97 user turns, 22 subagents
  - `fa3/cursor/69fff5d7-d194-4a1c-9ec8-de9efe54b617/69fff5d7-d194-4a1c-9ec8-de9efe54b617.jsonl`
  - tools: AskQuestion, AwaitShell, Delete, Glob, Grep, Read, ReadLints, Shell, StrReplace, Task, TodoWrite, Write
  - first prompt: I have built a draft of an agent (@data-fast/agent/fa3agent) to write cuda kernels (with no external library dependencies, e.g. no cutlass/cublas/cudnn) that matches flash attentio…
- `5bf704c4-c209-40b0-8b03-e9702d42e1ca` — **FA3/FA4 reference test shapes; run eval; update README**
  - cursor, 0.14 MB, 4 user turns, 0 subagents
  - `fa3/cursor/5bf704c4-c209-40b0-8b03-e9702d42e1ca/5bf704c4-c209-40b0-8b03-e9702d42e1ca.jsonl`
  - tools: Glob, Grep, Read, ReadLints, Shell, StrReplace, TodoWrite
  - first prompt: what shapes/sizes/data types/etc does the @fa3-cuda/reference/vllm-flash-attention flash attention repository use for testing the correctness/performance of their implementation fo…
- `c70e67c2-fa65-464c-84da-8966455e7b3a` — **Download NVIDIA SOL-ExecBench dataset**
  - cursor, 0.03 MB, 2 user turns, 0 subagents
  - `fa3/cursor/c70e67c2-fa65-464c-84da-8966455e7b3a/c70e67c2-fa65-464c-84da-8966455e7b3a.jsonl`
  - tools: AskQuestion, AwaitShell, Glob, Grep, Read, Shell
  - first prompt: please download the https://github.com/NVIDIA/SOL-ExecBench.git dataset
- `1814441e-5032-455d-bb8a-520719dac001` — **FA3 CUDA-only impl finished — next steps**
  - cursor, 0.03 MB, 1 user turns, 0 subagents
  - `fa3/cursor/1814441e-5032-455d-bb8a-520719dac001/1814441e-5032-455d-bb8a-520719dac001.jsonl`
  - tools: AskQuestion, Glob, Grep, Read, Shell, SwitchMode, TodoWrite
  - first prompt: I just finished using cursor and claude code to write a cuda-only flash-attention 3 implementation in @fa3-cuda , matching the output and performance of the original flash attentio…
- `6355d96f-1df1-462b-b267-a573e27421b4` — **FA3 CUDA-only impl finished — next steps (follow-up)**
  - cursor, 0.08 MB, 1 user turns, 1 subagents
  - `fa3/cursor/6355d96f-1df1-462b-b267-a573e27421b4/6355d96f-1df1-462b-b267-a573e27421b4.jsonl`
  - tools: AskQuestion, Glob, Read, ReadLints, Shell, StrReplace, TodoWrite, Write
  - first prompt: I just finished using cursor and claude code to write a cuda-only flash-attention 3 implementation in @fa3-cuda , matching the output and performance of the original flash attentio…

### Distilled notes
- `fa3/claude/memory/fa3-perf-gap-closing.md` — wins, failures, NCU diagnoses, Hopper architectural limits
- `fa3/claude/memory/fa3-perf-baseline.md` — earlier fa3_fwd.cu baseline + tile-size notes
- `fa3/claude/memory/fa3-page128-nan-fix.md` — combine-kernel NaN (empty splits)
- `fa3/claude/memory/fa3-cuda-eval-harness.md` — build / eval / NCU commands
- `fa3/context/plans/fancy-puzzling-cook.md` — should we reproduce the CUTLASS FA3 pipeline in CUDA?
- `fa3/context/cutlass-oracle-20260618/` — early CUTLASS-free replication plan + oracle kernel sources
- `fa3/file-history/<session-id>/` — Claude file snapshots

## FA4 (Blackwell / sm100)

### Claude Code
- `6d1d7b3b-7537-4354-ab2c-f53891e12861` — **fa4_blackwell.cu: optimize to FA4 baseline via skill + subagents**
  - claude, 2.78 MB, 2 user turns, 6 subagents
  - `fa4/claude/6d1d7b3b-7537-4354-ab2c-f53891e12861.jsonl`
  - tools: Agent, Bash, Edit, Read, Skill, TaskCreate, TaskUpdate, Write
  - first prompt: A session-scoped Stop hook is now active with condition: "optimize our cuda-only flash attention 4 kernel @kernels/fa4_blackwell.cu to match (and hopefully, exceed) the FA4 baselin…
- `593417ba-d98c-4c97-8a11-26fa37054973` — **fa4_blackwell.cu: match/exceed FA4 on the real subset (setup)**
  - claude, 0.01 MB, 1 user turns, 0 subagents
  - `fa4/claude/593417ba-d98c-4c97-8a11-26fa37054973.jsonl`
  - tools: Bash
  - first prompt: A session-scoped Stop hook is now active with condition: "optimize our cuda-only flash attention 4 kernel @kernels/fa4_blackwell.cu to match (and hopefully, exceed) the FA4 baselin…

### Cursor
- `d5135234-ac2e-4bd1-aeda-59e8faae3416` — **Optimize fa4_blackwell.cu to FA4 parity (long run)**
  - cursor, 3.85 MB, 34 user turns, 2 subagents
  - `fa4/cursor/d5135234-ac2e-4bd1-aeda-59e8faae3416/d5135234-ac2e-4bd1-aeda-59e8faae3416.jsonl`
  - tools: AskQuestion, AwaitShell, Delete, Glob, Grep, Read, ReadLints, Shell, StrReplace, Task, TodoWrite, Write
  - first prompt: Optimize the implementation of @fa3-cuda/kernels/fa4_blackwell.cu until we reach or exceed performance parity with the original flash attention 4 implementation (see repo @fa3-cuda…
- `62db84ae-fda9-44ad-8c61-a6c12b92ecdf` — **FA4 prefill perf — hide latency, port missing FA4 opts**
  - cursor, 0.58 MB, 6 user turns, 6 subagents
  - `fa4/cursor/62db84ae-fda9-44ad-8c61-a6c12b92ecdf/62db84ae-fda9-44ad-8c61-a6c12b92ecdf.jsonl`
  - tools: AskQuestion, AwaitShell, Glob, Grep, Read, ReadLints, Shell, StrReplace, Task, TodoWrite, Write
  - first prompt: I have been working on implementing flash attention 4 in pure cuda in @fa3-cuda/kernels/fa4_blackwell.cu . The performance is still not very good for prefills compared to the origi…

### Distilled notes
- `fa4/context/skills/optimize-fa4-blackwell/` — skill that drove later FA4 loops in `flash-cuda`

## Removed in this pass
Not FA3/FA4 kernel work:
- All `-home-yak-fastkernels` Claude sessions (serve API, registry, repo layout, which tasks call FlashAttention as a library)
- Cursor fastkernels chats (capture.py, registry, list-ops, bench_vllm, Qwen safetensor dtypes)
- Cursor rsync-backup-of-`.cursor` script
- Cursor “why does Claude Code stop / compact” (ops, not kernel)
- Empty FA4 `/effort ultracode` stub (`68b76351-…`)
- Claude `eventual-floating-nygaard.md` plan (fastkernels package move)

Previously excluded: personal/legal chat, auth-token updater, empty “hi” stubs, plugin caches, compiled oracle binaries, live Aug 2026 fastkernels H200 chats, `re3-claude`.

A JWT dumped into Claude session `17a633ac-…` was replaced with `REDACTED_JWT`.
