#!/usr/bin/env python3
"""Offload a fraction of the softmax's `ex2.approx` from the transcendental (XU) pipe to the FMA pipe.

NCU on `dense_c_d128_s8k` (fwd_kernel_fa4, hd128 causal 8k): XU pipe 56.3% of peak, tensor pipe
55.2%, FMA pipe 9.8%. The two are CO-LIMITING -- the 1-pass softmax issues 128 `ex2.approx` per row
per K block, and at 4 XU lanes per SMSP that costs about as many cycles as the four 128x128x128 MMAs
it feeds. FA4 hits the same wall and solves it the same way (`ex2_emu_freq` in
flash_fwd_sm100.py + `e2e_asm2` in utils.py): evaluate a fraction of the exponentials with a packed
f32x2 degree-3 polynomial on the FMA pipe instead.

For dp <= 0 (always true on the full path, where m_new is the row max):
    clamp to >= -127, add 2^23+2^22 with round-toward-minus-infinity so the integer floor lands in
    the mantissa, recover the fraction, evaluate 2^frac by a minimax cubic, then splice the integer
    back into the exponent field with a shift+add. Bit-exact reasoning: for v in [2^23, 2^24) the
    raw pattern is (150<<23)|m with m = 2^22 + floor(dp), so (raw << 23) mod 2^32 == floor(dp) << 23,
    and adding the float bits of 2^frac (exponent 127) yields 2^floor(dp) * 2^frac.

The generated softmax blocks already DECLARE the registers this needs (mgc, pc0..pc3, e7..e10,
ei0/ei1, ef0/ef1) -- they were reserved and never used. This script fills them in.

    tools/ex2_emu.py 4        # emulate every 4th pair (25% of the exponentials)
    tools/ex2_emu.py 0        # revert to all-hardware ex2
"""
from __future__ import annotations

import re
import sys

SRC = "kernels/fa4_blackwell.cu"
# Only the variants the fa4-family kernels call (fwd_kernel_fa4 / _paged / dk use ORDCS; ORD is the
# no-correction-channel sibling). The ws2 / ws_paged softmaxes are left alone.
FUNCS = ("softmax_full_ptx_ord_bf16", "softmax_full_ptx_ord_f16",
         "softmax_full_ptx_ordcs_bf16", "softmax_full_ptx_ordcs_f16")

CONSTS = (
    '        "// ex2 emulation constants (magic round + minimax cubic for 2^f on [0,1))\\n"\n'
    '        "mov.f32 d0, 0f4B400000; mov.b64 mgc, {d0, d0};\\n"\n'
    '        "mov.f32 d0, 0f3D9DF09D; mov.b64 pc3, {d0, d0};\\n"\n'
    '        "mov.f32 d0, 0f3E6906A4; mov.b64 pc2, {d0, d0};\\n"\n'
    '        "mov.f32 d0, 0f3F31F519; mov.b64 pc1, {d0, d0};\\n"\n'
    '        "mov.f32 d0, 0f3F800000; mov.b64 pc0, {d0, d0};\\n"\n'
)

# The two hardware ex2 lines for one pair (fA, fB) read d0/d1, which the preceding
# "mov.b64 {d0, d1}, dp;" produced.
PAIR_RE = re.compile(
    r'( *)"mov\.b64 \{d0, d1\}, dp;\\n"\n'
    r' *"ex2\.approx\.ftz\.f32 (f\d+), d0;\\n"\n'
    r' *"ex2\.approx\.ftz\.f32 (f\d+), d1;\\n"\n'
)


def emu(indent: str, fa: str, fb: str) -> str:
    L = [
        'mov.b64 {d0, d1}, dp;',
        'max.f32 d0, d0, 0fC2FE0000;',
        'max.f32 d1, d1, 0fC2FE0000;',
        'mov.b64 e7, {d0, d1};',
        'add.rm.ftz.f32x2 e8, e7, mgc;',
        'sub.rn.ftz.f32x2 e9, e8, mgc;',
        'sub.rn.ftz.f32x2 e10, e7, e9;',
        'fma.rn.ftz.f32x2 e9, e10, pc3, pc2;',
        'fma.rn.ftz.f32x2 e9, e9, e10, pc1;',
        'fma.rn.ftz.f32x2 e9, e9, e10, pc0;',
        'mov.b64 {ei0, ei1}, e8;',
        'mov.b64 {ef0, ef1}, e9;',
        'shl.b32 ei0, ei0, 23;',
        'shl.b32 ei1, ei1, 23;',
        'add.s32 ei0, ei0, ef0;',
        'add.s32 ei1, ei1, ef1;',
        f'mov.b32 {fa}, ei0;',
        f'mov.b32 {fb}, ei1;',
    ]
    return "".join(f'{indent}"{ln}\\n"\n' for ln in L)


def main() -> int:
    every = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    src = open(SRC).read()
    # strip any previous run
    src = src.replace(CONSTS, "")
    src = re.sub(
        r' *"// ex2 emulation constants[^\n]*\n(?: *"mov\.f32 d0, 0f[0-9A-F]{8}; mov\.b64 (?:mgc|pc[0-3]), \{d0, d0\};\\n"\n)+',
        "", src)
    total = 0
    for fn in FUNCS:
        i = src.index(f"void {fn}(")
        j = src.index("\n}\n", i)
        body = src[i:j]
        if every > 0:
            n = [0]

            def rep(m):
                idx = n[0]
                n[0] += 1
                if idx % every != every - 1:
                    return m.group(0)
                return emu(m.group(1), m.group(2), m.group(3))

            nb = PAIR_RE.sub(rep, body)
            done = n[0] // every
            if done:
                # inject the constant setup after the last .reg declaration
                k = nb.rindex('.reg ')
                k = nb.index("\n", k) + 1
                nb = nb[:k] + CONSTS + nb[k:]
            total += done
            print(f"  {fn}: {n[0]} pairs, {done} emulated")
            src = src[:i] + nb + src[j:]
    open(SRC, "w").write(src)
    print(f"total emulated pairs: {total} (every={every})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
