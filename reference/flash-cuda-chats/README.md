# flash-cuda-chats

Archive of the **Cursor** and **Claude Code** sessions used to manually write and
optimize the CUDA-only FlashAttention kernels in this repo. Checked in under
`reference/flash-cuda-chats/` (plain files, not a submodule).

Organized by target so a later agent can mine **what worked / what failed**
separately for Hopper vs Blackwell.

| Dir | Kernel | GPU |
|---|---|---|
| `fa3/` | `kernels/fa3_hopper.cu` (and earlier `fa3_fwd.cu`) | Hopper sm90 |
| `fa4/` | `kernels/fa4_blackwell.cu` | Blackwell sm100 |

Kernel sources stay in `kernels/`. This directory is transcripts, subagents,
file-history, and distilled notes only.

## Layout

```
fa3/
  claude/          parent JSONL + sidecars + memory/
  cursor/<uuid>/   parent JSONL + subagents/
  file-history/    Claude Code file snapshots
  context/         oracle sources + Hopper plan
fa4/
  claude/
  cursor/<uuid>/
  context/skills/optimize-fa4-blackwell/
INDEX.md
inventory.json
provenance.json
```

Catalog: [`INDEX.md`](INDEX.md). Highest-signal FA3 extract:
`fa3/claude/memory/fa3-perf-gap-closing.md`. Highest-signal FA4 extract:
`fa4/context/skills/optimize-fa4-blackwell/SKILL.md`.

## Formats

- **Claude Code:** JSONL, one event per line (`type: user|assistant|…`).
  Tool uses are `type: tool_use` inside assistant `message.content`.
- **Cursor:** `fa{3,4}/cursor/<uuid>/<uuid>.jsonl`. User turns are
  `<user_query>…</user_query>`. Subagents live in `subagents/`.

## What was dropped

Fastkernels infra, personal/ops chats, empty stubs, plugin caches, tokens,
compiled binaries. See `INDEX.md`.
