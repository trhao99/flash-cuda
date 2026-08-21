// Fused FlashAttention-3 forward (prefill, head_dim=128, bf16, causal) in pure CUDA + inline PTX.
// NO cutlass / cute. Wires together the oracle-validated components:
//   * QK GEMM      wgmma.mma_async.m64n128k16.f32.bf16.bf16  (SS, Q & K K-major in smem)
//   * online softmax (log2): RAW S, exp2.approx via exp2f, per-block running max/sum rescale, mask
//   * P (=softmax(S)) -> bf16 register fragment == PV A-operand (RS) directly (no smem round-trip)
//   * PV GEMM      wgmma.mma_async.m64n128k16.f32.bf16.bf16  (RS, A=P regs, B=V in smem)
//   * epilogue     O /= rowsum ; LSE = scale*rowmax + ln(rowsum) ; bf16 O store, fp32 LSE store
// One CTA = (m_block, batch*head). Layout [B,H,S,D] dense. Compile-time mainloop variants:
//   default : 2 MMA warpgroups (256 thr, each owns 64 rows); single-buffered synchronous loads.
//   -DPIPE  : 2 WG; cp.async double-buffered K (prefetch n+1) + V load overlapped with QK.
//   -DOVL   : 2 WG; cp.async K AND V (V natural/dv-contiguous -> PV wgmma imm-trans-b=1) + intra-WG
//             software pipeline (softmax(nb) overlaps in-flight PV(nb-1)).             ~0.39x FA3.
//   -DWS    : 3 warpgroups (384 thr) = 1 producer (cp.async loads + mbarrier) + 2 consumers with a
//             named-barrier ping-pong scheduler (softmax overlaps the PEER WG's GEMMs). 2-stage ring.
//   -DWSO   : -DWS + intra-WG PV-lag overlap (FA3's full structure). 3-stage ring.  ~0.50-0.52x FA3.
//   -DWST   : -DWSO load path replaced by TMA (cp.async.bulk.tensor) + 128B-swizzle smem (FA3's real
//             load path): single-warp producer issues 1 TMA/box w/ mbarrier complete-tx; consumers read
//             swizzled smem with B128 wgmma descriptors. K/V validated bit-exact in tma_swz_test*.cu.
//             ~0.71-0.76x FA3 (TMA kills the long-scoreboard memory stall: NCU 1.62->0.68).
//   -DWSTP  : -DWST + FA3's StaticPersistentTileScheduler (grid=#SMs, work-strided tiles, K/V ring phase
//             carried across tiles). Neutral for causal (static loses the HW block scheduler's dynamic
//             load-balancing), ~+3% for noncausal.
//   -DNOPP  : (WST/WSTP) drop the inter-WG ping-pong token. Post-TMA the kernel is BARRIER-bound (NCU
//             barrier stall 2.2, dominant), so the ping-pong hurts: removing it is +6-7% universally.
//             Recommended: -DWSTP -DNOPP (~0.76-0.80x FA3) or -DWST -DNOPP.
//   -DVTRANS: (default kernel only) swap V to the cp.async trans-b path for isolated testing.
//
// Usage: ./fused_attn <S> [D=128] [causal=1] [B=1] [H=1] [iters] [trials]

#include "attn_ref.cuh"
#include <cuda.h>

constexpr int CTA_M = 128, CTA_N = 128, HD = 128;   // block sizes (kBlockM,kBlockN,head_dim)
constexpr int NWG = 2, WG = 128, NThreads = NWG * WG;
// warp-specialized variant: 1 producer (loads) + 2 consumer (MMA) warpgroups = 384 threads.
constexpr int NThreadsWS = 3 * WG, NumProd = WG, NumCons = 2 * WG;

// ---- GMMA descriptor + K-major smem layout (from ptx_gemm.cu) ----
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t a, uint32_t lbo, uint32_t sbo, uint32_t lt) {
    uint64_t d = 0;
    d |= (uint64_t)((a >> 4) & 0x3FFFu);
    d |= (uint64_t)((lbo >> 4) & 0x3FFFu) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFFu) << 32;
    d |= (uint64_t)(lt & 0x3u) << 62;
    return d;
}
// Broadcast a warp-uniform value from lane 0 so ptxas represents it in a UNIFORM register. wgmma reads
// its smem descriptor from a uniform register; if the descriptor base is in a regular register ptxas
// R2UR's it into ONE reused uniform pair per k-step (WAR hazard) -> DEPBAR after every HGMMA (full
// serialization). Sourcing the base from a uniform reg lets ptxas keep 8 distinct uniform descriptors.
__device__ __forceinline__ uint32_t uniform_u32(uint32_t x) { return __shfl_sync(0xffffffffu, x, 0); }
__device__ __forceinline__ int gmma_k_off(int mn, int k, int MN) {
    int a = mn >> 3, b = mn & 7, kb = k >> 4, kk = k & 15, c = kk >> 3, dd = kk & 7;
    return kb * (MN * 16) + a * 128 + c * 64 + b * 8 + dd;
}
// MN-major (mn-contiguous) no-swizzle layout (CUTLASS Layout_MN_INTER_Atom: 8x8 atom, MN contiguous).
// 8(mn)x8(k) atom = 64 contiguous elems; mn-atoms stride 64 (regular over the N extent); k-groups of 8
// stride 1024 (= 128 mn * 8). Used for V loaded naturally (dv contiguous) -> PV wgmma imm-trans-b=1.
__device__ __forceinline__ int gmma_mn_off(int mn, int k, int K) {
    return (k >> 3) * 1024 + (mn >> 3) * 64 + (k & 7) * 8 + (mn & 7);
}
__device__ __forceinline__ void load_kmajor(const __nv_bfloat16* __restrict__ G, __nv_bfloat16* S,
                                            int row0, int MN, int K, int tid, int nt) {
    const int chunks = MN * (K / 8);
    for (int idx = tid; idx < chunks; idx += nt) {
        int mn = idx / (K / 8), kc = (idx % (K / 8)) * 8;
        *reinterpret_cast<float4*>(&S[gmma_k_off(mn, kc, MN)]) =
            *reinterpret_cast<const float4*>(&G[(size_t)(row0 + mn) * K + kc]);
    }
}
// V[key,dv] (row-major, key offset by ptr) -> sVt[gmma_k_off(dv,key,CTA_N)] (V^T, K=key major)
__device__ __forceinline__ void load_vt(const __nv_bfloat16* __restrict__ V, __nv_bfloat16* S,
                                        int N, int Kk, int tid, int nt) {
    const int chunks = (CTA_N / 8) * Kk;
    for (int idx = tid; idx < chunks; idx += nt) {
        int n0 = (idx % (CTA_N / 8)) * 8, k = idx / (CTA_N / 8);
        float4 v = *reinterpret_cast<const float4*>(&V[(size_t)k * N + n0]);
        const __nv_bfloat16* vv = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int i = 0; i < 8; ++i) S[gmma_k_off(n0 + i, k, CTA_N)] = vv[i];
    }
}

__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;
}
__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat16 a = __float2bfloat16_rn(lo), b = __float2bfloat16_rn(hi);
    uint16_t ua, ub; memcpy(&ua, &a, 2); memcpy(&ub, &b, 2);
    return uint32_t(ua) | (uint32_t(ub) << 16);
}

// ---- wgmma building blocks (8 k16 steps, m64n128) ----
__device__ __forceinline__ void qk_gemm(float (&s)[64], uint32_t sQ_wg, uint32_t sK_base) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < HD / 16; ++kb) {
        uint64_t dA = make_smem_desc(sQ_wg   + (uint32_t)(kb * CTA_M * 16 * 2), 128, 256, 0);
        uint64_t dB = make_smem_desc(sK_base + (uint32_t)(kb * CTA_N * 16 * 2), 128, 256, 0);
        int scaleD = (kb == 0) ? 0 : 1;
        asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "%64,%65,p,%67,%68,%69,%70;\n}\n"
        : "+f"(s[0]),"+f"(s[1]),"+f"(s[2]),"+f"(s[3]),"+f"(s[4]),"+f"(s[5]),"+f"(s[6]),"+f"(s[7]),
          "+f"(s[8]),"+f"(s[9]),"+f"(s[10]),"+f"(s[11]),"+f"(s[12]),"+f"(s[13]),"+f"(s[14]),"+f"(s[15]),
          "+f"(s[16]),"+f"(s[17]),"+f"(s[18]),"+f"(s[19]),"+f"(s[20]),"+f"(s[21]),"+f"(s[22]),"+f"(s[23]),
          "+f"(s[24]),"+f"(s[25]),"+f"(s[26]),"+f"(s[27]),"+f"(s[28]),"+f"(s[29]),"+f"(s[30]),"+f"(s[31]),
          "+f"(s[32]),"+f"(s[33]),"+f"(s[34]),"+f"(s[35]),"+f"(s[36]),"+f"(s[37]),"+f"(s[38]),"+f"(s[39]),
          "+f"(s[40]),"+f"(s[41]),"+f"(s[42]),"+f"(s[43]),"+f"(s[44]),"+f"(s[45]),"+f"(s[46]),"+f"(s[47]),
          "+f"(s[48]),"+f"(s[49]),"+f"(s[50]),"+f"(s[51]),"+f"(s[52]),"+f"(s[53]),"+f"(s[54]),"+f"(s[55]),
          "+f"(s[56]),"+f"(s[57]),"+f"(s[58]),"+f"(s[59]),"+f"(s[60]),"+f"(s[61]),"+f"(s[62]),"+f"(s[63])
        : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(0));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

__device__ __forceinline__ void pv_gemm(float (&acc_o)[64], const uint32_t (&P)[32], uint32_t sVt_base) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < CTA_N / 16; ++kb) {
        uint32_t a0 = P[kb*4+0], a1 = P[kb*4+1], a2 = P[kb*4+2], a3 = P[kb*4+3];
        uint64_t dB = make_smem_desc(sVt_base + (uint32_t)(kb * CTA_N * 16 * 2), 128, 256, 0);
        int scaleD = 1;
        asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "{%64,%65,%66,%67},%68,p,%70,%71,%72;\n}\n"
        : "+f"(acc_o[0]),"+f"(acc_o[1]),"+f"(acc_o[2]),"+f"(acc_o[3]),"+f"(acc_o[4]),"+f"(acc_o[5]),"+f"(acc_o[6]),"+f"(acc_o[7]),
          "+f"(acc_o[8]),"+f"(acc_o[9]),"+f"(acc_o[10]),"+f"(acc_o[11]),"+f"(acc_o[12]),"+f"(acc_o[13]),"+f"(acc_o[14]),"+f"(acc_o[15]),
          "+f"(acc_o[16]),"+f"(acc_o[17]),"+f"(acc_o[18]),"+f"(acc_o[19]),"+f"(acc_o[20]),"+f"(acc_o[21]),"+f"(acc_o[22]),"+f"(acc_o[23]),
          "+f"(acc_o[24]),"+f"(acc_o[25]),"+f"(acc_o[26]),"+f"(acc_o[27]),"+f"(acc_o[28]),"+f"(acc_o[29]),"+f"(acc_o[30]),"+f"(acc_o[31]),
          "+f"(acc_o[32]),"+f"(acc_o[33]),"+f"(acc_o[34]),"+f"(acc_o[35]),"+f"(acc_o[36]),"+f"(acc_o[37]),"+f"(acc_o[38]),"+f"(acc_o[39]),
          "+f"(acc_o[40]),"+f"(acc_o[41]),"+f"(acc_o[42]),"+f"(acc_o[43]),"+f"(acc_o[44]),"+f"(acc_o[45]),"+f"(acc_o[46]),"+f"(acc_o[47]),
          "+f"(acc_o[48]),"+f"(acc_o[49]),"+f"(acc_o[50]),"+f"(acc_o[51]),"+f"(acc_o[52]),"+f"(acc_o[53]),"+f"(acc_o[54]),"+f"(acc_o[55]),
          "+f"(acc_o[56]),"+f"(acc_o[57]),"+f"(acc_o[58]),"+f"(acc_o[59]),"+f"(acc_o[60]),"+f"(acc_o[61]),"+f"(acc_o[62]),"+f"(acc_o[63])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// async (cp.async) variant of load_kmajor: K-major direct copy (8 contiguous bf16 = 16B per cp.async).
__device__ __forceinline__ void load_kmajor_async(const __nv_bfloat16* __restrict__ G,
                                                  __nv_bfloat16* S, int row0, int MN, int K,
                                                  int tid, int nt) {
    const int chunks = MN * (K / 8);
    for (int idx = tid; idx < chunks; idx += nt) {
        int mn = idx / (K / 8), kc = (idx % (K / 8)) * 8;
        uint32_t s = (uint32_t)__cvta_generic_to_shared(&S[gmma_k_off(mn, kc, MN)]);
        const void* g = &G[(size_t)(row0 + mn) * K + kc];
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g));
    }
}
// cp.async V in NATURAL layout (dv contiguous) into the MN-major smem layout (gmma_mn_off). Each
// thread copies 16B = 8 contiguous dv of one key row -> one contiguous 16B smem chunk. Read by PV
// wgmma with imm-trans-b=1. K-dim extent for the layout is CTA_N (V's key axis within the block).
__device__ __forceinline__ void load_vnat_async(const __nv_bfloat16* __restrict__ V,
                                                __nv_bfloat16* Sb, int tid, int nt) {
    const int chunks = CTA_N * (HD / 8);
    for (int idx = tid; idx < chunks; idx += nt) {
        int key = idx / (HD / 8), dv0 = (idx % (HD / 8)) * 8;
        uint32_t s = (uint32_t)__cvta_generic_to_shared(&Sb[gmma_mn_off(dv0, key, CTA_N)]);
        const void* g = &V[(size_t)key * HD + dv0];
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g));
    }
}
#ifndef VLBO
#define VLBO 2048
#endif
#ifndef VSBO
#define VSBO 128
#endif
#ifndef VKB
#define VKB 4096
#endif
// PV wgmma with MN-major V (imm-trans-b=1); issue-only. K-block stride for gmma_mn_off is 16 keys ->
// (16>>3)*1024 = 2048 elems = 4096 bytes.
__device__ __forceinline__ void pv_gemm_issue_t(float (&acc_o)[64], const uint32_t (&P)[32], uint32_t sV_base) {
    uint64_t dBv[CTA_N / 16];
#pragma unroll
    for (int kb = 0; kb < CTA_N / 16; ++kb) dBv[kb] = make_smem_desc(sV_base + (uint32_t)(kb * VKB), VLBO, VSBO, 0);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < CTA_N / 16; ++kb) {
        uint32_t a0 = P[kb*4+0], a1 = P[kb*4+1], a2 = P[kb*4+2], a3 = P[kb*4+3];
        uint64_t dB = dBv[kb];
        int scaleD = 1;
        asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "{%64,%65,%66,%67},%68,p,%70,%71,%72;\n}\n"
        : "+f"(acc_o[0]),"+f"(acc_o[1]),"+f"(acc_o[2]),"+f"(acc_o[3]),"+f"(acc_o[4]),"+f"(acc_o[5]),"+f"(acc_o[6]),"+f"(acc_o[7]),
          "+f"(acc_o[8]),"+f"(acc_o[9]),"+f"(acc_o[10]),"+f"(acc_o[11]),"+f"(acc_o[12]),"+f"(acc_o[13]),"+f"(acc_o[14]),"+f"(acc_o[15]),
          "+f"(acc_o[16]),"+f"(acc_o[17]),"+f"(acc_o[18]),"+f"(acc_o[19]),"+f"(acc_o[20]),"+f"(acc_o[21]),"+f"(acc_o[22]),"+f"(acc_o[23]),
          "+f"(acc_o[24]),"+f"(acc_o[25]),"+f"(acc_o[26]),"+f"(acc_o[27]),"+f"(acc_o[28]),"+f"(acc_o[29]),"+f"(acc_o[30]),"+f"(acc_o[31]),
          "+f"(acc_o[32]),"+f"(acc_o[33]),"+f"(acc_o[34]),"+f"(acc_o[35]),"+f"(acc_o[36]),"+f"(acc_o[37]),"+f"(acc_o[38]),"+f"(acc_o[39]),
          "+f"(acc_o[40]),"+f"(acc_o[41]),"+f"(acc_o[42]),"+f"(acc_o[43]),"+f"(acc_o[44]),"+f"(acc_o[45]),"+f"(acc_o[46]),"+f"(acc_o[47]),
          "+f"(acc_o[48]),"+f"(acc_o[49]),"+f"(acc_o[50]),"+f"(acc_o[51]),"+f"(acc_o[52]),"+f"(acc_o[53]),"+f"(acc_o[54]),"+f"(acc_o[55]),
          "+f"(acc_o[56]),"+f"(acc_o[57]),"+f"(acc_o[58]),"+f"(acc_o[59]),"+f"(acc_o[60]),"+f"(acc_o[61]),"+f"(acc_o[62]),"+f"(acc_o[63])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(1));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void pv_gemm_t(float (&acc_o)[64], const uint32_t (&P)[32], uint32_t sV_base) {
    pv_gemm_issue_t(acc_o, P, sV_base);
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// Combined QK(SS) + PV(RS, trans-b) issue under a SINGLE wgmma.fence with ALL inputs (descriptors,
// s, acc_o) written BEFORE the fence -> nothing defines a wgmma input between fence and the final
// commit, so ptxas can pipeline the two groups (no C7513). Caller pre-zeros s and pre-rescales acc_o.
// Two committed groups: group1=QK (wait1 -> s ready), group0=PV (wait0 -> acc_o ready).
__device__ __forceinline__ void qkpv_issue(float (&s)[64], float (&acc_o)[64], const uint32_t (&P)[32],
                                           uint32_t sQ_wg, uint32_t sK_base, uint32_t sV_base) {
    uint64_t dA[HD / 16], dB[HD / 16], dBv[CTA_N / 16];
#pragma unroll
    for (int kb = 0; kb < HD / 16; ++kb) {
        dA[kb] = make_smem_desc(sQ_wg   + (uint32_t)(kb * CTA_M * 16 * 2), 128, 256, 0);
        dB[kb] = make_smem_desc(sK_base + (uint32_t)(kb * CTA_N * 16 * 2), 128, 256, 0);
    }
#pragma unroll
    for (int kb = 0; kb < CTA_N / 16; ++kb) dBv[kb] = make_smem_desc(sV_base + (uint32_t)(kb * VKB), VLBO, VSBO, 0);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < HD / 16; ++kb) {
        // scale-d=1 for every step (caller pre-zeroed s, so +0.0 is bit-exact) -> constant predicate,
        // no per-step divergent path (avoids C7520 wgmma serialization).
        asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "%64,%65,p,%66,%67,%68,%69;\n}\n"
        : "+f"(s[0]),"+f"(s[1]),"+f"(s[2]),"+f"(s[3]),"+f"(s[4]),"+f"(s[5]),"+f"(s[6]),"+f"(s[7]),
          "+f"(s[8]),"+f"(s[9]),"+f"(s[10]),"+f"(s[11]),"+f"(s[12]),"+f"(s[13]),"+f"(s[14]),"+f"(s[15]),
          "+f"(s[16]),"+f"(s[17]),"+f"(s[18]),"+f"(s[19]),"+f"(s[20]),"+f"(s[21]),"+f"(s[22]),"+f"(s[23]),
          "+f"(s[24]),"+f"(s[25]),"+f"(s[26]),"+f"(s[27]),"+f"(s[28]),"+f"(s[29]),"+f"(s[30]),"+f"(s[31]),
          "+f"(s[32]),"+f"(s[33]),"+f"(s[34]),"+f"(s[35]),"+f"(s[36]),"+f"(s[37]),"+f"(s[38]),"+f"(s[39]),
          "+f"(s[40]),"+f"(s[41]),"+f"(s[42]),"+f"(s[43]),"+f"(s[44]),"+f"(s[45]),"+f"(s[46]),"+f"(s[47]),
          "+f"(s[48]),"+f"(s[49]),"+f"(s[50]),"+f"(s[51]),"+f"(s[52]),"+f"(s[53]),"+f"(s[54]),"+f"(s[55]),
          "+f"(s[56]),"+f"(s[57]),"+f"(s[58]),"+f"(s[59]),"+f"(s[60]),"+f"(s[61]),"+f"(s[62]),"+f"(s[63])
        : "l"(dA[kb]),"l"(dB[kb]),"n"(1),"n"(1),"n"(0),"n"(0));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");   // group1 = QK
#pragma unroll
    for (int kb = 0; kb < CTA_N / 16; ++kb) {
        uint32_t a0 = P[kb*4+0], a1 = P[kb*4+1], a2 = P[kb*4+2], a3 = P[kb*4+3];
        asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "{%64,%65,%66,%67},%68,p,%69,%70,%71;\n}\n"
        : "+f"(acc_o[0]),"+f"(acc_o[1]),"+f"(acc_o[2]),"+f"(acc_o[3]),"+f"(acc_o[4]),"+f"(acc_o[5]),"+f"(acc_o[6]),"+f"(acc_o[7]),
          "+f"(acc_o[8]),"+f"(acc_o[9]),"+f"(acc_o[10]),"+f"(acc_o[11]),"+f"(acc_o[12]),"+f"(acc_o[13]),"+f"(acc_o[14]),"+f"(acc_o[15]),
          "+f"(acc_o[16]),"+f"(acc_o[17]),"+f"(acc_o[18]),"+f"(acc_o[19]),"+f"(acc_o[20]),"+f"(acc_o[21]),"+f"(acc_o[22]),"+f"(acc_o[23]),
          "+f"(acc_o[24]),"+f"(acc_o[25]),"+f"(acc_o[26]),"+f"(acc_o[27]),"+f"(acc_o[28]),"+f"(acc_o[29]),"+f"(acc_o[30]),"+f"(acc_o[31]),
          "+f"(acc_o[32]),"+f"(acc_o[33]),"+f"(acc_o[34]),"+f"(acc_o[35]),"+f"(acc_o[36]),"+f"(acc_o[37]),"+f"(acc_o[38]),"+f"(acc_o[39]),
          "+f"(acc_o[40]),"+f"(acc_o[41]),"+f"(acc_o[42]),"+f"(acc_o[43]),"+f"(acc_o[44]),"+f"(acc_o[45]),"+f"(acc_o[46]),"+f"(acc_o[47]),
          "+f"(acc_o[48]),"+f"(acc_o[49]),"+f"(acc_o[50]),"+f"(acc_o[51]),"+f"(acc_o[52]),"+f"(acc_o[53]),"+f"(acc_o[54]),"+f"(acc_o[55]),
          "+f"(acc_o[56]),"+f"(acc_o[57]),"+f"(acc_o[58]),"+f"(acc_o[59]),"+f"(acc_o[60]),"+f"(acc_o[61]),"+f"(acc_o[62]),"+f"(acc_o[63])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"l"(dBv[kb]),"n"(1),"n"(1),"n"(1));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");   // group0 = PV
}
// QK wgmma, issue only (no wait) so the caller can overlap other work with the tensor compute.
// Descriptors are built BEFORE wgmma.fence (not between fence/commit) to avoid C7513 serialization.
__device__ __forceinline__ void qk_gemm_issue(float (&s)[64], uint32_t sQ_wg, uint32_t sK_base) {
    uint64_t dA[HD / 16], dB[HD / 16];
#pragma unroll
    for (int kb = 0; kb < HD / 16; ++kb) {
        dA[kb] = make_smem_desc(sQ_wg   + (uint32_t)(kb * CTA_M * 16 * 2), 128, 256, 0);
        dB[kb] = make_smem_desc(sK_base + (uint32_t)(kb * CTA_N * 16 * 2), 128, 256, 0);
    }
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < HD / 16; ++kb) {
        int scaleD = (kb == 0) ? 0 : 1;
        asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "%64,%65,p,%67,%68,%69,%70;\n}\n"
        : "+f"(s[0]),"+f"(s[1]),"+f"(s[2]),"+f"(s[3]),"+f"(s[4]),"+f"(s[5]),"+f"(s[6]),"+f"(s[7]),
          "+f"(s[8]),"+f"(s[9]),"+f"(s[10]),"+f"(s[11]),"+f"(s[12]),"+f"(s[13]),"+f"(s[14]),"+f"(s[15]),
          "+f"(s[16]),"+f"(s[17]),"+f"(s[18]),"+f"(s[19]),"+f"(s[20]),"+f"(s[21]),"+f"(s[22]),"+f"(s[23]),
          "+f"(s[24]),"+f"(s[25]),"+f"(s[26]),"+f"(s[27]),"+f"(s[28]),"+f"(s[29]),"+f"(s[30]),"+f"(s[31]),
          "+f"(s[32]),"+f"(s[33]),"+f"(s[34]),"+f"(s[35]),"+f"(s[36]),"+f"(s[37]),"+f"(s[38]),"+f"(s[39]),
          "+f"(s[40]),"+f"(s[41]),"+f"(s[42]),"+f"(s[43]),"+f"(s[44]),"+f"(s[45]),"+f"(s[46]),"+f"(s[47]),
          "+f"(s[48]),"+f"(s[49]),"+f"(s[50]),"+f"(s[51]),"+f"(s[52]),"+f"(s[53]),"+f"(s[54]),"+f"(s[55]),
          "+f"(s[56]),"+f"(s[57]),"+f"(s[58]),"+f"(s[59]),"+f"(s[60]),"+f"(s[61]),"+f"(s[62]),"+f"(s[63])
        : "l"(dA[kb]),"l"(dB[kb]),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(0));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait1() {
    asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
}
// PV wgmma, issue only (no wait) — for intra-WG overlap (softmax overlaps the in-flight PV).
__device__ __forceinline__ void pv_gemm_issue(float (&acc_o)[64], const uint32_t (&P)[32], uint32_t sVt_base) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < CTA_N / 16; ++kb) {
        uint32_t a0 = P[kb*4+0], a1 = P[kb*4+1], a2 = P[kb*4+2], a3 = P[kb*4+3];
        uint64_t dB = make_smem_desc(sVt_base + (uint32_t)(kb * CTA_N * 16 * 2), 128, 256, 0);
        int scaleD = 1;
        asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "{%64,%65,%66,%67},%68,p,%70,%71,%72;\n}\n"
        : "+f"(acc_o[0]),"+f"(acc_o[1]),"+f"(acc_o[2]),"+f"(acc_o[3]),"+f"(acc_o[4]),"+f"(acc_o[5]),"+f"(acc_o[6]),"+f"(acc_o[7]),
          "+f"(acc_o[8]),"+f"(acc_o[9]),"+f"(acc_o[10]),"+f"(acc_o[11]),"+f"(acc_o[12]),"+f"(acc_o[13]),"+f"(acc_o[14]),"+f"(acc_o[15]),
          "+f"(acc_o[16]),"+f"(acc_o[17]),"+f"(acc_o[18]),"+f"(acc_o[19]),"+f"(acc_o[20]),"+f"(acc_o[21]),"+f"(acc_o[22]),"+f"(acc_o[23]),
          "+f"(acc_o[24]),"+f"(acc_o[25]),"+f"(acc_o[26]),"+f"(acc_o[27]),"+f"(acc_o[28]),"+f"(acc_o[29]),"+f"(acc_o[30]),"+f"(acc_o[31]),
          "+f"(acc_o[32]),"+f"(acc_o[33]),"+f"(acc_o[34]),"+f"(acc_o[35]),"+f"(acc_o[36]),"+f"(acc_o[37]),"+f"(acc_o[38]),"+f"(acc_o[39]),
          "+f"(acc_o[40]),"+f"(acc_o[41]),"+f"(acc_o[42]),"+f"(acc_o[43]),"+f"(acc_o[44]),"+f"(acc_o[45]),"+f"(acc_o[46]),"+f"(acc_o[47]),
          "+f"(acc_o[48]),"+f"(acc_o[49]),"+f"(acc_o[50]),"+f"(acc_o[51]),"+f"(acc_o[52]),"+f"(acc_o[53]),"+f"(acc_o[54]),"+f"(acc_o[55]),
          "+f"(acc_o[56]),"+f"(acc_o[57]),"+f"(acc_o[58]),"+f"(acc_o[59]),"+f"(acc_o[60]),"+f"(acc_o[61]),"+f"(acc_o[62]),"+f"(acc_o[63])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
// softmax stats for one owned row: update running max/sum, write P into s, RETURN the O-rescale factor.
// Does NOT touch acc_o (so it can run while a PV wgmma accumulates into acc_o).
__device__ __forceinline__ float softmax_stats(float (&s)[64], float* m_r, float* l_r,
                                               int r, int nb, float s2) {
#ifdef ABLATE_SM
    return 1.0f;   // ablation: skip all softmax math (WRONG result; measures GEMM+overlap ceiling only)
#endif
    float bm = -INFINITY;
#pragma unroll
    for (int c = 0; c < CTA_N / 8; ++c) { bm = fmaxf(bm, s[c*4+r*2+0]); bm = fmaxf(bm, s[c*4+r*2+1]); }
    bm = quad_max(bm);
    float scale_f;
    if (nb == 0) { m_r[r] = bm; scale_f = 1.f; }
    else { float prev = m_r[r]; m_r[r] = fmaxf(m_r[r], bm);
           float cur = (m_r[r] == -INFINITY) ? 0.f : m_r[r];
           scale_f = exp2f((prev - cur) * s2); l_r[r] *= scale_f; }
    float max_scaled = (m_r[r] == -INFINITY) ? 0.f : m_r[r] * s2;
    float lsum = 0.f;
#pragma unroll
    for (int c = 0; c < CTA_N / 8; ++c) {
        float p0 = exp2f(s[c*4+r*2+0] * s2 - max_scaled);
        float p1 = exp2f(s[c*4+r*2+1] * s2 - max_scaled);
        s[c*4+r*2+0] = p0; s[c*4+r*2+1] = p1; lsum += p0 + p1;
    }
    l_r[r] = (nb == 0) ? lsum : l_r[r] + lsum;
    return scale_f;
}
__device__ __forceinline__ void rescale_o(float (&acc_o)[64], const float* sc) {
#pragma unroll
    for (int c = 0; c < CTA_N / 8; ++c) {
        acc_o[c*4+0] *= sc[0]; acc_o[c*4+1] *= sc[0];
        acc_o[c*4+2] *= sc[1]; acc_o[c*4+3] *= sc[1];
    }
}
__device__ __forceinline__ void mask_causal(float (&s)[64], int col0, int qlo, int qhi, int tg) {
#pragma unroll
    for (int c = 0; c < CTA_N / 8; ++c) {
        int key0 = col0 + 8 * c + 2 * tg;
        if (key0     > qlo) s[c*4+0] = -INFINITY;
        if (key0 + 1 > qlo) s[c*4+1] = -INFINITY;
        if (key0     > qhi) s[c*4+2] = -INFINITY;
        if (key0 + 1 > qhi) s[c*4+3] = -INFINITY;
    }
}

// online-softmax of one owned row r over the S fragment; updates m_r,l_r, rescales acc_o, writes P(=s).
__device__ __forceinline__ void softmax_row(float (&s)[64], float (&acc_o)[64], float* m_r, float* l_r,
                                            int r, int nb, float s2) {
    float bm = -INFINITY;
#pragma unroll
    for (int c = 0; c < CTA_N / 8; ++c) { bm = fmaxf(bm, s[c*4+r*2+0]); bm = fmaxf(bm, s[c*4+r*2+1]); }
    bm = quad_max(bm);
    float scale_f;
    if (nb == 0) { m_r[r] = bm; scale_f = 1.f; }
    else { float prev = m_r[r]; m_r[r] = fmaxf(m_r[r], bm);
           float cur = (m_r[r] == -INFINITY) ? 0.f : m_r[r];
           scale_f = exp2f((prev - cur) * s2); l_r[r] *= scale_f; }
    float max_scaled = (m_r[r] == -INFINITY) ? 0.f : m_r[r] * s2;
    float lsum = 0.f;
#pragma unroll
    for (int c = 0; c < CTA_N / 8; ++c) {
        float p0 = exp2f(s[c*4+r*2+0] * s2 - max_scaled);
        float p1 = exp2f(s[c*4+r*2+1] * s2 - max_scaled);
        s[c*4+r*2+0] = p0; s[c*4+r*2+1] = p1; lsum += p0 + p1;
        acc_o[c*4+r*2+0] *= scale_f; acc_o[c*4+r*2+1] *= scale_f;
    }
    l_r[r] = (nb == 0) ? lsum : l_r[r] + lsum;
}

// ---- warp-specialization primitives (mbarrier / named barrier / setmaxnreg) ----
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(a) : "memory");
}
// register the executing thread's prior cp.async ops with the mbarrier (deferred arrive on completion);
// .noinc -> mbarrier.init count must equal the number of producer threads doing this.
__device__ __forceinline__ void cpasync_mbar_arrive_noinc(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(a) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, int phase) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t ok;
    asm volatile("{\n.reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(a), "r"(phase));
    return ok != 0;
}
// named barrier across the 2 consumer warpgroups (256 thr): scheduler ping-pong token.
__device__ __forceinline__ void nbar_sync256(int id) {
    asm volatile("bar.sync %0, 256;\n" :: "r"(id) : "memory");
}
__device__ __forceinline__ void nbar_arrive256(int id) {
    asm volatile("bar.arrive %0, 256;\n" :: "r"(id) : "memory");
}
__device__ __forceinline__ void producer_regs() { asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory"); }
__device__ __forceinline__ void consumer_regs() { asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory"); }

// ================= TMA + 128B-swizzle path (-DWST) — validated in tma_swz_test{,_v}.cu =================
// One swizzled tile = 128 rows x 128 (HD or dv) bf16 = 16384 elems, laid out as TWO 64-wide (128B) boxes
// at smem offsets 0 and 8192 (both K-major and FA3's MN-major SmemLayoutVt place the 2 halves at clean
// stride 8192). wgmma reads with B128 descriptors:
//   K-major (Q,K): LBO=16B, SBO=1024B, lt=1; per-k16-step start = ((kb&3)*16 + (kb>>2)*8192) elems.
//   MN-major (V) : LBO=16384B, SBO=1024B, lt=1; per-k16-step (over keys) start = kb*1024 elems; trans-b=1.
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(a), "r"(bytes) : "memory");
}
// 2D bulk tensor copy gmem->smem with mbarrier complete-tx. coord {x=inner(HD/dv off), y=row}.
__device__ __forceinline__ void tma_box(void* smem_dst, const CUtensorMap* map, int x, int y, uint64_t* bar) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_dst), b = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];\n" :: "r"(s), "l"(map), "r"(x), "r"(y), "r"(b) : "memory");
}
// load one [128 rows, 128 cols] tile (2 boxes) into a swizzled smem tile; coords from global row base.
__device__ __forceinline__ void tma_tile(__nv_bfloat16* dst, const CUtensorMap* map, int grow, uint64_t* bar) {
    tma_box(dst,        map, 0,  grow, bar);   // cols 0..63   -> smem 0..8191
    tma_box(dst + 8192, map, 64, grow, bar);   // cols 64..127 -> smem 8192..16383
}
constexpr uint32_t SW_QK_LBO = 16, SW_QK_SBO = 1024;       // K-major B128
constexpr uint32_t SW_V_LBO = 16384, SW_V_SBO = 1024;      // MN-major B128
__device__ __forceinline__ uint32_t qk_kstep(int kb) { return (uint32_t)(((kb & 3) * 16 + (kb >> 2) * 8192) * 2); }
__device__ __forceinline__ uint32_t v_kstep(int kb)  { return (uint32_t)(kb * 1024 * 2); }

// 64-accumulator register list (operands %0..%63), shared by every wgmma inside a single-block gemm.
// Emitting all k-steps in ONE asm block forces ptxas to keep each step's descriptor in a DISTINCT
// register, eliminating the per-wgmma `R2UR UR,Rx` descriptor-reload (a WAR hazard that otherwise makes
// ptxas insert `WARPGROUP.DEPBAR.LE gsb0,0x0` after every HGMMA -> full serialization, C7513). With this
// the warpgroup issues all k-steps async (one trailing wait_group), so groups overlap softmax.
#define WG_ACC64 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
"%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31," \
"%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47," \
"%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}"
#define WG_OUT64(a) \
        "+f"(a[0]),"+f"(a[1]),"+f"(a[2]),"+f"(a[3]),"+f"(a[4]),"+f"(a[5]),"+f"(a[6]),"+f"(a[7]), \
        "+f"(a[8]),"+f"(a[9]),"+f"(a[10]),"+f"(a[11]),"+f"(a[12]),"+f"(a[13]),"+f"(a[14]),"+f"(a[15]), \
        "+f"(a[16]),"+f"(a[17]),"+f"(a[18]),"+f"(a[19]),"+f"(a[20]),"+f"(a[21]),"+f"(a[22]),"+f"(a[23]), \
        "+f"(a[24]),"+f"(a[25]),"+f"(a[26]),"+f"(a[27]),"+f"(a[28]),"+f"(a[29]),"+f"(a[30]),"+f"(a[31]), \
        "+f"(a[32]),"+f"(a[33]),"+f"(a[34]),"+f"(a[35]),"+f"(a[36]),"+f"(a[37]),"+f"(a[38]),"+f"(a[39]), \
        "+f"(a[40]),"+f"(a[41]),"+f"(a[42]),"+f"(a[43]),"+f"(a[44]),"+f"(a[45]),"+f"(a[46]),"+f"(a[47]), \
        "+f"(a[48]),"+f"(a[49]),"+f"(a[50]),"+f"(a[51]),"+f"(a[52]),"+f"(a[53]),"+f"(a[54]),"+f"(a[55]), \
        "+f"(a[56]),"+f"(a[57]),"+f"(a[58]),"+f"(a[59]),"+f"(a[60]),"+f"(a[61]),"+f"(a[62]),"+f"(a[63])
// QK (SS, K-major) wgmma == cute MMA_64x128x16_F32BF16BF16_SS::fma. Predicate hardcoded as an immediate
// (no runtime scaleD operand) so it is NOT a per-wgmma non-wgmma input (avoids C7513 serialization).
__device__ __forceinline__ void wgmma_ss_ow(float (&d)[64], uint64_t da, uint64_t db) {  // overwrite (k0)
    asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p,0,0;\n"   // p = false -> scaleD=0 (overwrite)
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",%64,%65,p,1,1,0,0;\n}\n"
        : WG_OUT64(d) : "l"(da),"l"(db));
}
__device__ __forceinline__ void wgmma_ss(float (&d)[64], uint64_t da, uint64_t db) {      // accumulate
    asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p,1,0;\n"   // p = true -> scaleD=1 (accumulate)
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",%64,%65,p,1,1,0,0;\n}\n"
        : WG_OUT64(d) : "l"(da),"l"(db));
}
// PV (RS, A=regs, trans-b) wgmma == cute MMA_64x128x16_F32BF16BF16_RS::fma. Always accumulate.
__device__ __forceinline__ void wgmma_rs(float (&d)[64], uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
                                         uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p,1,0;\n"   // p = true -> scaleD=1 (accumulate)
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",{%64,%65,%66,%67},%68,p,1,1,1;\n}\n"
        : WG_OUT64(d) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"l"(db));
}
// ---- single-asm-block batches: ALL descriptors precomputed into distinct regs FIRST, then 8 back-to-back
// wgmma with NOTHING between them (matches FA3 SASS: 1 fence, 8 HGMMA, 1 commit; 0 per-HGMMA DEPBAR).
// Computing descriptors between wgmma triggers C7513 (non-wgmma def of a wgmma input mid-pipeline) ->
// per-HGMMA DEPBAR serialization, so we hoist all the add.s64 ahead of the fence.
// da[k]=%(64+k) (A descriptors), db[k]=%(72+k) (B descriptors), p0=%80. Nothing between wgmma.
#define QK_DIN(d) "l"(d[0]),"l"(d[1]),"l"(d[2]),"l"(d[3]),"l"(d[4]),"l"(d[5]),"l"(d[6]),"l"(d[7])
__device__ __forceinline__ void qk_batch_blk(float (&s)[64], const uint64_t (&da)[8], const uint64_t (&db)[8], int p0) {
    asm volatile(
        "{\n.reg .pred p,q;\n setp.ne.b32 p,%80,0;\n setp.ne.b32 q,1,0;\n"
        "wgmma.fence.sync.aligned;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",%64,%72,p,1,1,0,0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",%65,%73,q,1,1,0,0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",%66,%74,q,1,1,0,0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",%67,%75,q,1,1,0,0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",%68,%76,q,1,1,0,0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",%69,%77,q,1,1,0,0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",%70,%78,q,1,1,0,0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",%71,%79,q,1,1,0,0;\n"
        "wgmma.commit_group.sync.aligned;\n}\n"
        : WG_OUT64(s) : QK_DIN(da), QK_DIN(db), "r"(p0));
}
// P[i]=%(64+i) (32 A regs), dv[k]=%(96+k) (8 B descriptors). Nothing between wgmma.
#define PVW3(i0,i1,i2,i3,k) \
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",{%" #i0 ",%" #i1 ",%" #i2 ",%" #i3 "},%" #k ",q,1,1,1;\n"
#define PV_AIN(P) \
    "r"(P[0]),"r"(P[1]),"r"(P[2]),"r"(P[3]),"r"(P[4]),"r"(P[5]),"r"(P[6]),"r"(P[7]), \
    "r"(P[8]),"r"(P[9]),"r"(P[10]),"r"(P[11]),"r"(P[12]),"r"(P[13]),"r"(P[14]),"r"(P[15]), \
    "r"(P[16]),"r"(P[17]),"r"(P[18]),"r"(P[19]),"r"(P[20]),"r"(P[21]),"r"(P[22]),"r"(P[23]), \
    "r"(P[24]),"r"(P[25]),"r"(P[26]),"r"(P[27]),"r"(P[28]),"r"(P[29]),"r"(P[30]),"r"(P[31])
#define PV_DIN(d) "l"(d[0]),"l"(d[1]),"l"(d[2]),"l"(d[3]),"l"(d[4]),"l"(d[5]),"l"(d[6]),"l"(d[7])
__device__ __forceinline__ void pv_batch_blk(float (&acc)[64], const uint32_t (&P)[32], const uint64_t (&dv)[8]) {
    asm volatile(
        "{\n.reg .pred q;\n setp.ne.b32 q,1,0;\n"
        "wgmma.fence.sync.aligned;\n"
        PVW3(64,65,66,67,96)  PVW3(68,69,70,71,97)  PVW3(72,73,74,75,98)  PVW3(76,77,78,79,99)
        PVW3(80,81,82,83,100) PVW3(84,85,86,87,101) PVW3(88,89,90,91,102) PVW3(92,93,94,95,103)
        "wgmma.commit_group.sync.aligned;\n}\n"
        : WG_OUT64(acc) : PV_AIN(P), PV_DIN(dv));
}
// cute warpgroup wrappers. warpgroup_fence_operand (empty asm "+f"/"+r" w/ memory clobber) on EVERY
// accumulator/A register brackets the wgmma batch so ptxas treats them as one async-managed unit and
// does NOT inject a per-wgmma warpgroup.arrive+DEPBAR (C7519/C7513) — without these the k-loop fully
// serializes (DEPBAR after every HGMMA), defeating the QK/PV/softmax overlap.
__device__ __forceinline__ void wg_fence_c(float (&c)[64]) {
#ifndef NOFENCE
#pragma unroll
    for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(c[i]) :: "memory");
#endif
}
__device__ __forceinline__ void wg_fence_a(const uint32_t (&a)[32]) {
#ifndef NOFENCE
#pragma unroll
    for (int i = 0; i < 32; ++i) asm volatile("" : "+r"(const_cast<uint32_t&>(a[i])) :: "memory");
#endif
}
__device__ __forceinline__ void wg_fence_desc(uint64_t &d) { asm volatile("" : "+l"(d) :: "memory"); }
// cute DescriptorIterator::operator+ : increment ONLY the low 32 bits (address field lives there); the
// high 32 bits (layout/swizzle) are shared & constant across the 8 k-steps. This lets ptxas keep ONE
// shared hi reg + 8 distinct lo regs instead of recycling a full 64-bit desc reg (which forces a
// recompute->WAR->DEPBAR after every wgmma). Mirrors cute exactly.
__device__ __forceinline__ uint64_t desc_add(uint64_t base, uint32_t off) {
    uint32_t lo, hi;
    asm("mov.b64 {%0,%1}, %2;\n" : "=r"(lo), "=r"(hi) : "l"(base));
    lo += off;
    uint64_t r; asm("mov.b64 %0, {%1,%2};\n" : "=l"(r) : "r"(lo), "r"(hi));
    return r;
}
__device__ __forceinline__ void wg_arrive() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
// descriptor address-field (>>4) increments per k16-step: QK K-major, V MN-major.
#ifdef DIAG_CONSEC
__device__ __forceinline__ uint64_t qk_doff(int kb) { return (uint64_t)(kb * 2); }  // DIAG: consecutive (WRONG)
#else
__device__ __forceinline__ uint64_t qk_doff(int kb) { return (uint64_t)((kb & 3) * 2 + (kb >> 2) * 1024); }
#endif
__device__ __forceinline__ uint64_t v_doff(int kb)  { return (uint64_t)(kb * 128); }

// QK (SS) batch: EXACT structure of the proven-async qk_gemm (basic kernel) -> 8 back-to-back wgmma with
// only 1 DEPBAR. The descriptors are computed INLINE inside the unrolled loop (NOT precomputed into da[8]):
// this is the ONLY structure ptxas keeps async in the large warp-specialized consumer. Precomputing da[8]
// or separate helper calls makes ptxas recycle ONE descriptor UR (recompute->WAR->DEPBAR per HGMMA).
// fence (once), 8 wgmma, commit. NO wait/post-fence (deferred by caller for QK/softmax/PV overlap).
__device__ __forceinline__ void qk_batch(float (&s)[64], uint32_t sQ_wg, uint32_t sK_base, bool first) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < HD / 16; ++kb) {
        uint32_t off = (uint32_t)qk_doff(kb) << 4;                  // box-swizzle k-step in BYTES (addr<<4)
        uint64_t dA = make_smem_desc(sQ_wg + off, SW_QK_LBO, SW_QK_SBO, 1);
        uint64_t dB = make_smem_desc(sK_base + off, SW_QK_LBO, SW_QK_SBO, 1);
        int scaleD = (first && kb == 0) ? 0 : 1;
        asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",%64,%65,p,%67,%68,%69,%70;\n}\n"
        : WG_OUT64(s) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(0));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
// PV (RS) batch: EXACT structure of the proven-async pv_gemm. Inline descriptor, 8 back-to-back wgmma, commit.
__device__ __forceinline__ void pv_batch(float (&acc_o)[64], const uint32_t (&P)[32], uint32_t sV_base) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < CTA_N / 16; ++kb) {
        uint32_t off = (uint32_t)v_doff(kb) << 4;                   // addr-field -> bytes
        uint64_t dB = make_smem_desc(sV_base + off, SW_V_LBO, SW_V_SBO, 1);
        asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",{%64,%65,%66,%67},%68,p,%70,%71,%72;\n}\n"
        : WG_OUT64(acc_o)
        : "r"(P[kb*4+0]),"r"(P[kb*4+1]),"r"(P[kb*4+2]),"r"(P[kb*4+3]),"l"(dB),"r"(1),"n"(1),"n"(1),"n"(1));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
// QK (SS) full: batch + wait_group 0 + post-fence (prologue).
__device__ __forceinline__ void qk_gemm_sw(float (&s)[64], uint32_t sQ_wg, uint32_t sK_base) {
    qk_batch(s, sQ_wg, sK_base, true);
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    wg_fence_c(s);
}
// PV (RS) full: batch + wait_group 0 + post-fence (epilogue / non-overlapped path).
__device__ __forceinline__ void pv_gemm_t_sw(float (&acc_o)[64], const uint32_t (&P)[32], uint32_t sV_base) {
    pv_batch(acc_o, P, sV_base);
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    wg_fence_c(acc_o);
    wg_fence_a(P);
}
// Issue-only (deferred wait): batch + post-fence. Caller does wgmma_wait{1,0}() then reads.
__device__ __forceinline__ void qk_issue_sw(float (&s)[64], uint32_t sQ_wg, uint32_t sK_base) {
    qk_batch(s, sQ_wg, sK_base, false);
    wg_fence_c(s);
}
__device__ __forceinline__ void pv_issue_sw(float (&acc_o)[64], const uint32_t (&P)[32], uint32_t sV_base) {
    pv_batch(acc_o, P, sV_base);
    wg_fence_c(acc_o);
    wg_fence_a(P);
}
// Combined QK(SS) group1 + PV(RS) group0, both async (commit, no wait). Caller: wgmma_wait1() (QK done)
// then wgmma_wait0() (PV done) so softmax overlaps the in-flight PV.
__device__ __forceinline__ void qkpv_issue_sw(float (&s)[64], float (&acc_o)[64], const uint32_t (&P)[32],
                                              uint32_t sQ_wg, uint32_t sK_base, uint32_t sV_base) {
    qk_issue_sw(s, sQ_wg, sK_base);     // group1 = QK
    pv_issue_sw(acc_o, P, sV_base);     // group0 = PV
}
// Deferred 2-group issue with ALL descriptors precomputed BEFORE either group, and each group emitted as
// a single all-8-wgmma asm block (qk/pv_batch_blk). This keeps BOTH groups async (no make_smem_desc
// between the QK and PV pipeline stages -> no C7513) while leaving them in flight for softmax overlap.
__device__ __forceinline__ void qkpv_issue_blk(float (&s)[64], float (&acc_o)[64], const uint32_t (&P)[32],
                                               uint32_t sQ_wg, uint32_t sK_base, uint32_t sV_base) {
    uint64_t da[8], db[8], dv[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        uint32_t qoff = (uint32_t)qk_doff(k) << 4;
        da[k] = make_smem_desc(sQ_wg   + qoff, SW_QK_LBO, SW_QK_SBO, 1);
        db[k] = make_smem_desc(sK_base + qoff, SW_QK_LBO, SW_QK_SBO, 1);
        dv[k] = make_smem_desc(sV_base + ((uint32_t)v_doff(k) << 4), SW_V_LBO, SW_V_SBO, 1);
    }
    qk_batch_blk(s, da, db, 0);          // QK group (zero_init: p0=0 -> first k-step overwrites)
    pv_batch_blk(acc_o, P, dv);          // PV group (NOTHING between groups: no fence redefines an input)
}

// ============ cute-faithful deferred 2-group issue (the "implement it like cute" path, -DCUTE) ============
// cute keeps two wgmma groups async because its descriptor is a DescriptorIterator: ONE base GmmaDescriptor
// (high-32 = layout/swizzle, low-32 = addr>>4) computed once, and operator+ adds a *compile-time-constant*
// to ONLY the low 32 bits (cute mma_traits_sm90_gmma.hpp:323-329). That is a single cheap UIADD3 per k-step
// (high reg shared) instead of the multi-shift/OR `make_smem_desc` (which ptxas R2UR-recycles -> per-HGMMA
// DEPBAR when 2 groups overlap). `desc_add` is exactly cute's operator+. Bases are hoisted before BOTH
// fences so nothing heavy is scheduled between the QK and PV pipeline stages.
__device__ __forceinline__ void qk_batch_cute(float (&s)[64], uint64_t baseA, uint64_t baseB, int p0) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < HD / 16; ++kb) {
        uint64_t dA = desc_add(baseA, (uint32_t)qk_doff(kb));   // cute operator+: low-32 += const
        uint64_t dB = desc_add(baseB, (uint32_t)qk_doff(kb));
        int scaleD = (kb == 0) ? p0 : 1;
        asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",%64,%65,p,%67,%68,%69,%70;\n}\n"
        : WG_OUT64(s) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(0));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void pv_batch_cute(float (&acc_o)[64], const uint32_t (&P)[32], uint64_t baseV) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < CTA_N / 16; ++kb) {
        uint64_t dB = desc_add(baseV, (uint32_t)v_doff(kb));
        asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " WG_ACC64 ",{%64,%65,%66,%67},%68,p,%70,%71,%72;\n}\n"
        : WG_OUT64(acc_o)
        : "r"(P[kb*4+0]),"r"(P[kb*4+1]),"r"(P[kb*4+2]),"r"(P[kb*4+3]),"l"(dB),"r"(1),"n"(1),"n"(1),"n"(1));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
// Force a 64-bit value into a UNIFORM register (broadcast lane0 of each warp). The wgmma descriptor is
// warpgroup-uniform (one tile base addr for all 128 thr); making it provably warp-uniform lets ptxas read
// it as gdesc[UR] and keep 8 DISTINCT uniform descriptors (UIADD3.64 UR,UR,const) instead of R2UR-recycling
// ONE regular reg per k-step (the WAR hazard -> per-HGMMA DEPBAR). Mirrors cute canonical_warp_group_idx's
// __shfl_sync(threadIdx.x/128) which seeds the descriptor base from a uniform datapath.
__device__ __forceinline__ uint64_t uniform_u64(uint64_t x) {
    uint32_t lo = (uint32_t)x, hi = (uint32_t)(x >> 32);
    lo = __shfl_sync(0xffffffffu, lo, 0);
    hi = __shfl_sync(0xffffffffu, hi, 0);
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}
__device__ __forceinline__ void qkpv_issue_cute(float (&s)[64], float (&acc_o)[64], const uint32_t (&P)[32],
                                                uint32_t sQ_wg, uint32_t sK_base, uint32_t sV_base) {
    uint64_t baseA = make_smem_desc(sQ_wg,   SW_QK_LBO, SW_QK_SBO, 1);   // make_gmma_desc: ONCE (uniform datapath)
    uint64_t baseB = make_smem_desc(sK_base, SW_QK_LBO, SW_QK_SBO, 1);
    uint64_t baseV = make_smem_desc(sV_base, SW_V_LBO,  SW_V_SBO,  1);
    qk_batch_cute(s, baseA, baseB, 0);   // group1 = QK (zero_init; s pre-zeroed so p0=0 ok)
    pv_batch_cute(acc_o, P, baseV);      // group0 = PV
}

extern __shared__ __align__(1024) __nv_bfloat16 smem[];

__global__ __launch_bounds__(NThreads) void fused_attn_fwd(
        const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
        const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
        float* __restrict__ LSE, int S, int causal, float scale, int tiles_m) {
    const int m_block = blockIdx.x;
    if (m_block >= tiles_m) return;
    const int bh = blockIdx.y;
    const int row0 = m_block * CTA_M;
    const __nv_bfloat16* Qb = Q + (size_t)bh * S * HD;
    const __nv_bfloat16* Kb = K + (size_t)bh * S * HD;
    const __nv_bfloat16* Vb = V + (size_t)bh * S * HD;
    __nv_bfloat16* Ob = O + (size_t)bh * S * HD;
    float* LSEb = LSE + (size_t)bh * S;

    const int tid = threadIdx.x;
    const int wg = tid / WG, wtid = tid % WG, warp = wtid >> 5, lane = wtid & 31;
    const int group = lane >> 2, tg = lane & 3;
    const int row_lo = wg * 64 + warp * 16 + group, row_hi = row_lo + 8;   // within 128 tile
    const float s2 = scale * float(M_LOG2E);

    __nv_bfloat16* sQ = smem;
    __nv_bfloat16* sK = smem + CTA_M * HD;
    __nv_bfloat16* sVt = smem + 2 * CTA_M * HD;
#ifdef DIAG_WG0
    const uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ);  // DIAG: drop wg term (WRONG result)
#else
    const uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ) + (uint32_t)(wg * 1024 * 2);
#endif
    const uint32_t sK_base = (uint32_t)__cvta_generic_to_shared(sK);
    const uint32_t sVt_base = (uint32_t)__cvta_generic_to_shared(sVt);

    load_kmajor(Qb, sQ, row0, CTA_M, HD, tid, NThreads);   // Q resident for the whole n-loop

    float acc_o[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc_o[i] = 0.f;
    float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {0.f, 0.f};

    const int n_block_max = causal ? (m_block + 1) : tiles_m;   // S%128==0, q_len==k_len

    for (int nb = 0; nb < n_block_max; ++nb) {
        const int col0 = nb * CTA_N;     // key offset
        load_kmajor(Kb, sK, col0, CTA_N, HD, tid, NThreads);
#ifdef VTRANS
        load_vnat_async(Vb + (size_t)col0 * HD, sVt, tid, NThreads);
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
#else
        load_vt(Vb + (size_t)col0 * HD, sVt, HD, CTA_N, tid, NThreads);
#endif
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        // ---- GEMM0: S = Q . K^T  (SS) ----
        float s[64];
#pragma unroll
        for (int i = 0; i < 64; ++i) s[i] = 0.f;
        qk_gemm(s, sQ_wg, sK_base);

        // ---- mask (causal diagonal block only) ----
        if (causal && nb == m_block) {
#pragma unroll
            for (int c = 0; c < CTA_N / 8; ++c) {
                int key0 = col0 + 8 * c + 2 * tg;
                if (key0     > row0 + row_lo) s[c*4+0] = -INFINITY;
                if (key0 + 1 > row0 + row_lo) s[c*4+1] = -INFINITY;
                if (key0     > row0 + row_hi) s[c*4+2] = -INFINITY;
                if (key0 + 1 > row0 + row_hi) s[c*4+3] = -INFINITY;
            }
        }

        // ---- online softmax (per owned row r=0 lo, r=1 hi) ----
        softmax_row(s, acc_o, m_r, l_r, 0, nb, s2);
        softmax_row(s, acc_o, m_r, l_r, 1, nb, s2);

        // ---- P(fp32 frag) -> bf16 A-registers (linear pack == PV RS A-operand) ----
        uint32_t P[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) P[i] = pack_bf16(s[2*i], s[2*i+1]);

        // ---- GEMM1: O += P . V  (RS) ----
#ifdef VTRANS
        pv_gemm_t(acc_o, P, sVt_base);
#else
        pv_gemm(acc_o, P, sVt_base);
#endif
        __syncthreads();   // before next n-block overwrites sK/sVt
    }

    // ---- epilogue: normalize, LSE, store ----
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l_r[r] = quad_sum(l_r[r]);
        float inv = (l_r[r] == 0.f || l_r[r] != l_r[r]) ? 0.f : 1.f / l_r[r];
        int qr = row0 + (r == 0 ? row_lo : row_hi);
#pragma unroll
        for (int c = 0; c < HD / 8; ++c) {
            int dv0 = 8 * c + 2 * tg;
            uint32_t o = pack_bf16(acc_o[c*4+r*2+0] * inv, acc_o[c*4+r*2+1] * inv);
            if (qr < S) *reinterpret_cast<uint32_t*>(&Ob[(size_t)qr * HD + dv0]) = o;
        }
        if (tg == 0 && qr < S)
            LSEb[qr] = (l_r[r] == 0.f || l_r[r] != l_r[r]) ? -INFINITY
                                                           : m_r[r] * scale + __logf(l_r[r]);
    }
}

// ===================== Pipelined variant (cp.async double-buffered K; V load overlapped with QK) =====================
// Attacks the dominant bottleneck of the simple kernel (NCU: 43% long-scoreboard load stalls, tensor
// pipe 12.8% idle): K(nb+1) is cp.async-prefetched a block ahead into a 2nd smem buffer, and V(nb)'s
// global load is issued AFTER the QK wgmma so its latency overlaps the tensor compute.
__global__ __launch_bounds__(NThreads) void fused_attn_fwd_pipe(
        const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
        const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
        float* __restrict__ LSE, int S, int causal, float scale, int tiles_m) {
    const int m_block = blockIdx.x;
    if (m_block >= tiles_m) return;
    const int bh = blockIdx.y;
    const int row0 = m_block * CTA_M;
    const __nv_bfloat16* Qb = Q + (size_t)bh * S * HD;
    const __nv_bfloat16* Kb = K + (size_t)bh * S * HD;
    const __nv_bfloat16* Vb = V + (size_t)bh * S * HD;
    __nv_bfloat16* Ob = O + (size_t)bh * S * HD;
    float* LSEb = LSE + (size_t)bh * S;

    const int tid = threadIdx.x;
    const int wg = tid / WG, wtid = tid % WG, warp = wtid >> 5, lane = wtid & 31;
    const int group = lane >> 2, tg = lane & 3;
    const int row_lo = wg * 64 + warp * 16 + group, row_hi = row_lo + 8;
    const float s2 = scale * float(M_LOG2E);

    __nv_bfloat16* sQ = smem;
    __nv_bfloat16* sK[2] = {smem + CTA_M * HD, smem + CTA_M * HD + CTA_N * HD};
    __nv_bfloat16* sVt = smem + CTA_M * HD + 2 * CTA_N * HD;
#ifdef DIAG_WG0
    const uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ);  // DIAG: drop wg term (WRONG result)
#else
    const uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ) + (uint32_t)(wg * 1024 * 2);
#endif
    const uint32_t sK_base[2] = {(uint32_t)__cvta_generic_to_shared(sK[0]),
                                 (uint32_t)__cvta_generic_to_shared(sK[1])};
    const uint32_t sVt_base = (uint32_t)__cvta_generic_to_shared(sVt);

    const int n_max = causal ? (m_block + 1) : tiles_m;

    // prologue: Q + K(0)
    load_kmajor_async(Qb, sQ, row0, CTA_M, HD, tid, NThreads);
    load_kmajor_async(Kb, sK[0], 0, CTA_N, HD, tid, NThreads);
    asm volatile("cp.async.commit_group;\n" ::: "memory");

    float acc_o[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc_o[i] = 0.f;
    float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {0.f, 0.f};

    for (int nb = 0; nb < n_max; ++nb) {
        const int stage = nb & 1;
        const bool has_next = (nb + 1 < n_max);
        if (has_next) {
            load_kmajor_async(Kb, sK[(nb + 1) & 1], (nb + 1) * CTA_N, CTA_N, HD, tid, NThreads);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");   // K(nb) ready
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        // GEMM0 issue (async) — tensor cores work while we load V
        float s[64];
#pragma unroll
        for (int i = 0; i < 64; ++i) s[i] = 0.f;
        qk_gemm_issue(s, sQ_wg, sK_base[stage]);

        load_vt(Vb + (size_t)(nb * CTA_N) * HD, sVt, HD, CTA_N, tid, NThreads);  // overlaps QK
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        wgmma_wait0();   // QK done -> s valid

        if (causal && nb == m_block) {
#pragma unroll
            for (int c = 0; c < CTA_N / 8; ++c) {
                int key0 = nb * CTA_N + 8 * c + 2 * tg;
                if (key0     > row0 + row_lo) s[c*4+0] = -INFINITY;
                if (key0 + 1 > row0 + row_lo) s[c*4+1] = -INFINITY;
                if (key0     > row0 + row_hi) s[c*4+2] = -INFINITY;
                if (key0 + 1 > row0 + row_hi) s[c*4+3] = -INFINITY;
            }
        }
        softmax_row(s, acc_o, m_r, l_r, 0, nb, s2);
        softmax_row(s, acc_o, m_r, l_r, 1, nb, s2);
        uint32_t P[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) P[i] = pack_bf16(s[2*i], s[2*i+1]);
        pv_gemm(acc_o, P, sVt_base);
        __syncthreads();   // sVt reused next block
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l_r[r] = quad_sum(l_r[r]);
        float inv = (l_r[r] == 0.f || l_r[r] != l_r[r]) ? 0.f : 1.f / l_r[r];
        int qr = row0 + (r == 0 ? row_lo : row_hi);
#pragma unroll
        for (int c = 0; c < HD / 8; ++c) {
            int dv0 = 8 * c + 2 * tg;
            uint32_t o = pack_bf16(acc_o[c*4+r*2+0] * inv, acc_o[c*4+r*2+1] * inv);
            if (qr < S) *reinterpret_cast<uint32_t*>(&Ob[(size_t)qr * HD + dv0]) = o;
        }
        if (tg == 0 && qr < S)
            LSEb[qr] = (l_r[r] == 0.f || l_r[r] != l_r[r]) ? -INFINITY : m_r[r] * scale + __logf(l_r[r]);
    }
}

// ===================== Overlap + prefetch variant (FA3's core software pipeline) =====================
// Combines the two independently-validated wins:
//   (1) cp.async double-buffered K prefetch (K(nb+1) loaded a block ahead) so the QK GEMM never stalls
//       on K's global-load latency (NCU: long-scoreboard was the dominant stall, 6.3/issue).
//   (2) intra-warpgroup PV lag-by-one so softmax(nb) (FFMA/SFU pipes) runs while PV(nb-1) (tensor cores)
//       is in flight, filling the tensor-pipe idle. O-rescale is hoisted out of softmax (rescale_o,
//       before both wgmma issues) so nothing defines acc_o between the QK and PV issues (avoids C7513).
// V is cp.async'd in natural (dv-contiguous) layout (gmma_mn_off) and consumed by the PV wgmma with
// imm-trans-b=1 -> no LDG->STS scatter, fully decoupled from the math warps. Both K and V double-buffered:
// K prefetched one block ahead; V(nb) loaded at iter nb and consumed at iter nb+1 (the PV-lag hides its
// cp.async latency). One cp.async group per iter holds {K(nb+1),V(nb)}; wait_group 1 keeps one in flight.
__global__ __launch_bounds__(NThreads) void fused_attn_fwd_ovl(
        const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
        const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
        float* __restrict__ LSE, int S, int causal, float scale, int tiles_m) {
    const int m_block = blockIdx.x;
    if (m_block >= tiles_m) return;
    const int bh = blockIdx.y;
    const int row0 = m_block * CTA_M;
    const __nv_bfloat16* Qb = Q + (size_t)bh * S * HD;
    const __nv_bfloat16* Kb = K + (size_t)bh * S * HD;
    const __nv_bfloat16* Vb = V + (size_t)bh * S * HD;
    __nv_bfloat16* Ob = O + (size_t)bh * S * HD;
    float* LSEb = LSE + (size_t)bh * S;

    const int tid = threadIdx.x;
    const int wg = tid / WG, wtid = tid % WG, warp = wtid >> 5, lane = wtid & 31;
    const int group = lane >> 2, tg = lane & 3;
    const int row_lo = wg * 64 + warp * 16 + group, row_hi = row_lo + 8;
    const float s2 = scale * float(M_LOG2E);

    __nv_bfloat16* sQ = smem;
    __nv_bfloat16* sK[2] = {smem + CTA_M * HD, smem + CTA_M * HD + CTA_N * HD};
    __nv_bfloat16* sV[2] = {smem + CTA_M * HD + 2 * CTA_N * HD,
                            smem + CTA_M * HD + 3 * CTA_N * HD};
#ifdef DIAG_WG0
    const uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ);  // DIAG: drop wg term (WRONG result)
#else
    const uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ) + (uint32_t)(wg * 1024 * 2);
#endif
    const uint32_t sK_base[2] = {(uint32_t)__cvta_generic_to_shared(sK[0]),
                                 (uint32_t)__cvta_generic_to_shared(sK[1])};
    const uint32_t sV_base[2] = {(uint32_t)__cvta_generic_to_shared(sV[0]),
                                 (uint32_t)__cvta_generic_to_shared(sV[1])};

    float acc_o[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc_o[i] = 0.f;
    float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {0.f, 0.f};
    const int n_max = causal ? (m_block + 1) : tiles_m;
    const int qlo = row0 + row_lo, qhi = row0 + row_hi;

    float s[64];
    uint32_t P[32];
    float sc[2];

    // ---- prologue: cp.async Q,K(0),V(0) (group Ga) + prefetch K(1) (group Gb); QK(0)->softmax(0) ----
    load_kmajor_async(Qb, sQ, row0, CTA_M, HD, tid, NThreads);
    load_kmajor_async(Kb, sK[0], 0, CTA_N, HD, tid, NThreads);
    load_vnat_async(Vb, sV[0], tid, NThreads);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    if (n_max > 1) {   // prefetch K(1) for iter 1
        load_kmajor_async(Kb, sK[1], CTA_N, CTA_N, HD, tid, NThreads);
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;\n" ::: "memory");  // Ga done
    } else {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#pragma unroll
    for (int i = 0; i < 64; ++i) s[i] = 0.f;
    qk_gemm(s, sQ_wg, sK_base[0]);
    if (causal && m_block == 0) mask_causal(s, 0, qlo, qhi, tg);
    sc[0] = softmax_stats(s, m_r, l_r, 0, 0, s2);
    sc[1] = softmax_stats(s, m_r, l_r, 1, 0, s2);
#pragma unroll
    for (int i = 0; i < 32; ++i) P[i] = pack_bf16(s[2*i], s[2*i+1]);

    // ---- pipelined mainloop: iter nb computes QK(nb)+softmax(nb) while PV(nb-1) runs ----
    for (int nb = 1; nb < n_max; ++nb) {
        const int stage = nb & 1;
        // prefetch group G_nb = {K(nb+1) (if any), V(nb)}; wait_group 1 keeps G_nb in flight, drains
        // G_{nb-1} which holds K(nb) and V(nb-1) -- both needed this iter.
        if (nb + 1 < n_max)
            load_kmajor_async(Kb, sK[(nb + 1) & 1], (nb + 1) * CTA_N, CTA_N, HD, tid, NThreads);
        load_vnat_async(Vb + (size_t)(nb * CTA_N) * HD, sV[nb & 1], tid, NThreads);
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;\n" ::: "memory");
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

#pragma unroll
        for (int i = 0; i < 64; ++i) s[i] = 0.f;
        rescale_o(acc_o, sc);                             // apply softmax(nb-1) O-rescale (acc_o settled)
        qk_gemm_issue(s, sQ_wg, sK_base[stage]);          // GEMM0(nb) async
        pv_gemm_issue_t(acc_o, P, sV_base[(nb - 1) & 1]); // GEMM1(nb-1) async (trans-b V) -> acc_o
        wgmma_wait1();                                    // QK(nb) done; PV(nb-1) still in flight
        if (causal && nb == m_block) mask_causal(s, nb * CTA_N, qlo, qhi, tg);
        sc[0] = softmax_stats(s, m_r, l_r, 0, nb, s2);    // overlaps the in-flight PV(nb-1)
        sc[1] = softmax_stats(s, m_r, l_r, 1, nb, s2);
#pragma unroll
        for (int i = 0; i < 32; ++i) P[i] = pack_bf16(s[2*i], s[2*i+1]);
        wgmma_wait0();                                    // PV(nb-1) done -> acc_o settled
        __syncthreads();                                  // before sK/sV reuse next iter
    }

    // ---- epilogue: trailing PV for the last block, then normalize/LSE/store ----
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");   // V(n_max-1) (last G) ready
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    rescale_o(acc_o, sc);
    pv_gemm_t(acc_o, P, sV_base[(n_max - 1) & 1]);

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l_r[r] = quad_sum(l_r[r]);
        float inv = (l_r[r] == 0.f || l_r[r] != l_r[r]) ? 0.f : 1.f / l_r[r];
        int qr = row0 + (r == 0 ? row_lo : row_hi);
#pragma unroll
        for (int c = 0; c < HD / 8; ++c) {
            int dv0 = 8 * c + 2 * tg;
            uint32_t o = pack_bf16(acc_o[c*4+r*2+0] * inv, acc_o[c*4+r*2+1] * inv);
            if (qr < S) *reinterpret_cast<uint32_t*>(&Ob[(size_t)qr * HD + dv0]) = o;
        }
        if (tg == 0 && qr < S)
            LSEb[qr] = (l_r[r] == 0.f || l_r[r] != l_r[r]) ? -INFINITY : m_r[r] * scale + __logf(l_r[r]);
    }
}

// ===================== Warp-specialized variant (producer WG + 2-consumer ping-pong) =====================
// FA3's full structure: a dedicated PRODUCER warpgroup (128 thr, setmaxnreg->40) cp.async-loads K/V into a
// 2-stage smem ring and signals readiness via mbarrier (cp.async.mbarrier.arrive); TWO CONSUMER warpgroups
// (256 thr, setmaxnreg->232) each own 64 query rows and run the QK/softmax/PV math. The consumers ping-pong
// a scheduler token (2 named barriers, 256-thr) so that exactly one WG issues GEMMs at a time and each WG's
// softmax (FFMA/SFU) overlaps the OTHER WG's GEMMs (tensor pipe) -> ~2x tensor-pipe occupancy. Decoupling
// loads into a producer raises occupancy (12->18.75% warps) and removes the per-WG load stalls.
//   full[2]/empty[2]/barQ : producer->consumer (data ready) / consumer->producer (buffer free) / Q ready.
//   token protocol (per FA3 no-overlap mainloop): sync(self) once; per iter {QK; arrive(other); softmax;
//   sync(self); PV}; arrive(other) after loop; kickoff = cwg0 arrive(self) in "mma_init".
__global__ __launch_bounds__(NThreadsWS, 1) void fused_attn_fwd_ws(
        const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
        const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
        float* __restrict__ LSE, int S, int causal, float scale, int tiles_m) {
    const int m_block = blockIdx.x;
    const int bh = blockIdx.y;
    const int row0 = m_block * CTA_M;
    const __nv_bfloat16* Qb = Q + (size_t)bh * S * HD;
    const __nv_bfloat16* Kb = K + (size_t)bh * S * HD;
    const __nv_bfloat16* Vb = V + (size_t)bh * S * HD;
    __nv_bfloat16* Ob = O + (size_t)bh * S * HD;
    float* LSEb = LSE + (size_t)bh * S;

    const int tid = threadIdx.x;
    const int n_max = causal ? (m_block + 1) : tiles_m;
    const float s2 = scale * float(M_LOG2E);

    __nv_bfloat16* sQ = smem;
    __nv_bfloat16* sK[2] = {smem + CTA_M * HD, smem + CTA_M * HD + CTA_N * HD};
    __nv_bfloat16* sV[2] = {smem + CTA_M * HD + 2 * CTA_N * HD, smem + CTA_M * HD + 3 * CTA_N * HD};
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + (CTA_M + 4 * CTA_N) * HD);
    uint64_t* full = bars;        // [2] producer signals K(nb),V(nb) loaded
    uint64_t* empty = bars + 2;   // [2] consumers signal buffer free
    uint64_t* barQ = bars + 4;    // [1] Q loaded

    if (tid == 0) {
        mbar_init(&full[0], NumProd);  mbar_init(&full[1], NumProd);
        mbar_init(&empty[0], NumCons); mbar_init(&empty[1], NumCons);
        mbar_init(&barQ[0], NumProd);
    }
    __syncthreads();   // all 384 threads present (only sync before warp specialization)

    const int wg = tid / WG;   // 0 = producer, 1/2 = consumers

    if (wg == 0) {
        // ---------------- producer warpgroup ----------------
        producer_regs();
        const int ptid = tid;   // 0..127
        load_kmajor_async(Qb, sQ, row0, CTA_M, HD, ptid, NumProd);   // Q resident
        cpasync_mbar_arrive_noinc(&barQ[0]);
        for (int nb = 0; nb < n_max; ++nb) {
            const int st = nb & 1;
            if (nb >= 2) {   // wait until consumers freed buffer st (filled at nb-2)
                const int ep = ((nb >> 1) - 1) & 1;
                while (!mbar_try_wait(&empty[st], ep)) {}
            }
            load_kmajor_async(Kb, sK[st], nb * CTA_N, CTA_N, HD, ptid, NumProd);
            load_vnat_async(Vb + (size_t)(nb * CTA_N) * HD, sV[st], ptid, NumProd);
            cpasync_mbar_arrive_noinc(&full[st]);
        }
        return;
    }

    // ---------------- consumer warpgroups (wg 1,2) ----------------
    consumer_regs();
    const int cwg = wg - 1;   // 0 or 1
    const int wtid = tid % WG, warp = wtid >> 5, lane = wtid & 31;
    const int group = lane >> 2, tg = lane & 3;
    const int row_lo = cwg * 64 + warp * 16 + group, row_hi = row_lo + 8;
    const int qlo = row0 + row_lo, qhi = row0 + row_hi;
    const uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ) + (uint32_t)(cwg * 1024 * 2);
    const uint32_t sK_base[2] = {(uint32_t)__cvta_generic_to_shared(sK[0]),
                                 (uint32_t)__cvta_generic_to_shared(sK[1])};
    const uint32_t sV_base[2] = {(uint32_t)__cvta_generic_to_shared(sV[0]),
                                 (uint32_t)__cvta_generic_to_shared(sV[1])};
    const int SB_self = 1 + cwg, SB_other = 2 - cwg;   // named-barrier ids 1,2

    float acc_o[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc_o[i] = 0.f;
    float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {0.f, 0.f};
    float s[64]; uint32_t P[32]; float sc[2];

    while (!mbar_try_wait(&barQ[0], 0)) {}   // Q ready
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

    if (cwg == 0) nbar_arrive256(SB_self);   // kickoff: cwg0 takes the token first
    nbar_sync256(SB_self);                    // acquire token

    for (int nb = 0; nb < n_max; ++nb) {
        const int st = nb & 1;
        const int fp = (nb >> 1) & 1;
        while (!mbar_try_wait(&full[st], fp)) {}   // K(nb),V(nb) ready
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        // GEMM0: QK (token held) -> release -> wait
#pragma unroll
        for (int i = 0; i < 64; ++i) s[i] = 0.f;
        qk_gemm_issue(s, sQ_wg, sK_base[st]);
        nbar_arrive256(SB_other);     // pass token to the other WG
        wgmma_wait0();                // QK done

        if (causal && nb == m_block) mask_causal(s, nb * CTA_N, qlo, qhi, tg);
        sc[0] = softmax_stats(s, m_r, l_r, 0, nb, s2);   // overlaps other WG's GEMM
        sc[1] = softmax_stats(s, m_r, l_r, 1, nb, s2);
#pragma unroll
        for (int i = 0; i < 32; ++i) P[i] = pack_bf16(s[2*i], s[2*i+1]);
        rescale_o(acc_o, sc);

        // GEMM1: PV (re-acquire token)
        nbar_sync256(SB_self);
        pv_gemm_issue_t(acc_o, P, sV_base[st]);
        wgmma_wait0();                // PV done
        mbar_arrive(&empty[st]);      // K(nb),V(nb) consumed -> free buffer st
    }
    nbar_arrive256(SB_other);         // final release so the peer WG can exit

    // ---- epilogue: normalize, LSE, store ----
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l_r[r] = quad_sum(l_r[r]);
        float inv = (l_r[r] == 0.f || l_r[r] != l_r[r]) ? 0.f : 1.f / l_r[r];
        int qr = row0 + (r == 0 ? row_lo : row_hi);
#pragma unroll
        for (int c = 0; c < HD / 8; ++c) {
            int dv0 = 8 * c + 2 * tg;
            uint32_t o = pack_bf16(acc_o[c*4+r*2+0] * inv, acc_o[c*4+r*2+1] * inv);
            if (qr < S) *reinterpret_cast<uint32_t*>(&Ob[(size_t)qr * HD + dv0]) = o;
        }
        if (tg == 0 && qr < S)
            LSEb[qr] = (l_r[r] == 0.f || l_r[r] != l_r[r]) ? -INFINITY : m_r[r] * scale + __logf(l_r[r]);
    }
}

// ===================== Warp-specialized + intra-WG overlap (FA3's full mainloop) =====================
// Adds FA3's intra-warpgroup software pipeline on top of the producer+ping-pong WS kernel: each consumer
// keeps TWO wgmma groups in flight — QK(nb) and PV(nb-1) — and runs softmax(nb) while PV(nb-1) accumulates
// into acc_o (tensor pipe stays busy within a WG, not just across WGs). 3-stage smem ring (K+V per stage)
// so blocks nb (QK), nb-1 (PV) and the producer's prefetch nb+1 are all live. The scheduler token wraps
// BOTH GEMM issues (sync; QK; PV; arrive) so softmax overlaps the peer WG's GEMMs too. rescale_o is hoisted
// before the issues (acc_o not redefined between QK and PV) to avoid wgmma serialization (C7513).
constexpr int WS_STAGES = 3;
__global__ __launch_bounds__(NThreadsWS, 1) void fused_attn_fwd_ws_ovl(
        const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
        const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
        float* __restrict__ LSE, int S, int causal, float scale, int tiles_m) {
    const int m_block = blockIdx.x;
    const int bh = blockIdx.y;
    const int row0 = m_block * CTA_M;
    const __nv_bfloat16* Qb = Q + (size_t)bh * S * HD;
    const __nv_bfloat16* Kb = K + (size_t)bh * S * HD;
    const __nv_bfloat16* Vb = V + (size_t)bh * S * HD;
    __nv_bfloat16* Ob = O + (size_t)bh * S * HD;
    float* LSEb = LSE + (size_t)bh * S;

    const int tid = threadIdx.x;
    const int n_max = causal ? (m_block + 1) : tiles_m;
    const float s2 = scale * float(M_LOG2E);
    constexpr int KVSTRIDE = CTA_N * HD;   // bf16 elems between consecutive K (or V) stages

    __nv_bfloat16* sQ = smem;
    __nv_bfloat16* sK0 = smem + CTA_M * HD;                       // sK stage s = sK0 + s*KVSTRIDE
    __nv_bfloat16* sV0 = smem + CTA_M * HD + WS_STAGES * KVSTRIDE;// sV stage s = sV0 + s*KVSTRIDE
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + (CTA_M + 2 * WS_STAGES * CTA_N) * HD);
    uint64_t* full = bars;                 // [WS_STAGES]
    uint64_t* empty = bars + WS_STAGES;    // [WS_STAGES]
    uint64_t* barQ = bars + 2 * WS_STAGES; // [1]

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < WS_STAGES; ++s) { mbar_init(&full[s], NumProd); mbar_init(&empty[s], NumCons); }
        mbar_init(&barQ[0], NumProd);
    }
    __syncthreads();

    const int wg = tid / WG;
    if (wg == 0) {
        // ---------------- producer ----------------
        producer_regs();
        const int ptid = tid;
#if defined(ABLATE_NOLOAD)
        // diagnostic: signal readiness WITHOUT loading (garbage data, identical consumer tensor work).
        // full[]/barQ init count == NumProd, so all producer threads must arrive to flip the phase.
        // If tensor-pipe util rises vs the real loads, the producer is load-issue bound.
        mbar_arrive(&barQ[0]);
        for (int nb = 0; nb < n_max; ++nb) {
            const int st = nb % WS_STAGES;
            if (nb >= WS_STAGES) {
                const int ep = ((nb / WS_STAGES) - 1) & 1;
                while (!mbar_try_wait(&empty[st], ep)) {}
            }
            mbar_arrive(&full[st]);
        }
        return;
#endif
        load_kmajor_async(Qb, sQ, row0, CTA_M, HD, ptid, NumProd);
        cpasync_mbar_arrive_noinc(&barQ[0]);
        for (int nb = 0; nb < n_max; ++nb) {
            const int st = nb % WS_STAGES;
            if (nb >= WS_STAGES) {
                const int ep = ((nb / WS_STAGES) - 1) & 1;
                while (!mbar_try_wait(&empty[st], ep)) {}
            }
            load_kmajor_async(Kb, sK0 + st * KVSTRIDE, nb * CTA_N, CTA_N, HD, ptid, NumProd);
            load_vnat_async(Vb + (size_t)(nb * CTA_N) * HD, sV0 + st * KVSTRIDE, ptid, NumProd);
            cpasync_mbar_arrive_noinc(&full[st]);
        }
        return;
    }

    // ---------------- consumers ----------------
    consumer_regs();
    // cute canonical_warp_group_idx: __shfl from lane0 so ptxas's uniform-datapath analysis proves cwg (and
    // every smem descriptor derived from it) WARP-UNIFORM -> descriptors land NATIVELY in distinct uniform
    // regs (gdesc[UR8],gdesc[UR12],...) instead of being R2UR-recycled into ONE UR per k-step (the WAR
    // hazard that forces a DEPBAR after every HGMMA). This is the actual ptxas-circumvention cute relies on.
    const int cwg = __shfl_sync(0xffffffffu, wg - 1, 0);
    const int wtid = tid % WG, warp = wtid >> 5, lane = wtid & 31;
    const int group = lane >> 2, tg = lane & 3;
    const int row_lo = cwg * 64 + warp * 16 + group, row_hi = row_lo + 8;
    const int qlo = row0 + row_lo, qhi = row0 + row_hi;
    const uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ) + (uint32_t)(cwg * 1024 * 2);
    const uint32_t sK_a = (uint32_t)__cvta_generic_to_shared(sK0);
    const uint32_t sV_a = (uint32_t)__cvta_generic_to_shared(sV0);
    const uint32_t st_bytes = (uint32_t)(KVSTRIDE * 2);
    const int SB_self = 1 + cwg, SB_other = 2 - cwg;

    float acc_o[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc_o[i] = 0.f;
    float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {0.f, 0.f};
    float s[64]; uint32_t P[32]; float sc[2];

    while (!mbar_try_wait(&barQ[0], 0)) {}
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

    // ---- prologue: QK(0)+softmax(0) (no token; both WGs issue QK(0) concurrently) ----
    while (!mbar_try_wait(&full[0], 0)) {}   // K(0),V(0) ready
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#pragma unroll
    for (int i = 0; i < 64; ++i) s[i] = 0.f;
    qk_gemm(s, sQ_wg, sK_a);                  // issue+wait
    if (causal && m_block == 0) mask_causal(s, 0, qlo, qhi, tg);
    sc[0] = softmax_stats(s, m_r, l_r, 0, 0, s2);
    sc[1] = softmax_stats(s, m_r, l_r, 1, 0, s2);
#pragma unroll
    for (int i = 0; i < 32; ++i) P[i] = pack_bf16(s[2*i], s[2*i+1]);

    if (cwg == 0) nbar_arrive256(SB_self);    // ping-pong kickoff

    // ---- mainloop: QK(nb)+softmax(nb) overlap in-flight PV(nb-1) ----
    for (int nb = 1; nb < n_max; ++nb) {
        const int st = nb % WS_STAGES, pst = (nb - 1) % WS_STAGES;
        const int fp = (nb / WS_STAGES) & 1;
        while (!mbar_try_wait(&full[st], fp)) {}   // K(nb),V(nb) ready
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#pragma unroll
        for (int i = 0; i < 64; ++i) s[i] = 0.f;
        nbar_sync256(SB_self);                      // acquire token
        rescale_o(acc_o, sc);                       // settle acc_o from softmax(nb-1) (before fence)
        qkpv_issue(s, acc_o, P, sQ_wg, sK_a + st * st_bytes, sV_a + pst * st_bytes);
        nbar_arrive256(SB_other);                   // release token
        wgmma_wait1();                              // QK(nb) done; PV(nb-1) in flight
        if (causal && nb == m_block) mask_causal(s, nb * CTA_N, qlo, qhi, tg);
        sc[0] = softmax_stats(s, m_r, l_r, 0, nb, s2);   // overlaps in-flight PV(nb-1)
        sc[1] = softmax_stats(s, m_r, l_r, 1, nb, s2);
#pragma unroll
        for (int i = 0; i < 32; ++i) P[i] = pack_bf16(s[2*i], s[2*i+1]);
        wgmma_wait0();                              // PV(nb-1) done -> acc_o settled
        mbar_arrive(&empty[pst]);                   // stage pst (K,V of nb-1) free
    }
    if (cwg == 0) { /* keep token balance handled by mainloop arrives */ }

    // ---- epilogue: trailing PV(n_max-1) ----
    rescale_o(acc_o, sc);
    pv_gemm_t(acc_o, P, sV_a + ((n_max - 1) % WS_STAGES) * st_bytes);
    mbar_arrive(&empty[(n_max - 1) % WS_STAGES]);

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l_r[r] = quad_sum(l_r[r]);
        float inv = (l_r[r] == 0.f || l_r[r] != l_r[r]) ? 0.f : 1.f / l_r[r];
        int qr = row0 + (r == 0 ? row_lo : row_hi);
#pragma unroll
        for (int c = 0; c < HD / 8; ++c) {
            int dv0 = 8 * c + 2 * tg;
            uint32_t o = pack_bf16(acc_o[c*4+r*2+0] * inv, acc_o[c*4+r*2+1] * inv);
            if (qr < S) *reinterpret_cast<uint32_t*>(&Ob[(size_t)qr * HD + dv0]) = o;
        }
        if (tg == 0 && qr < S)
            LSEb[qr] = (l_r[r] == 0.f || l_r[r] != l_r[r]) ? -INFINITY : m_r[r] * scale + __logf(l_r[r]);
    }
}

// ===================== WST: -DWSO structure with FA3's TMA + 128B-swizzle load path =====================
// Identical consumer schedule to fused_attn_fwd_ws_ovl (producer + 2-consumer ping-pong + intra-WG PV
// overlap, 3-stage ring), but: (1) the producer is a SINGLE thread issuing cp.async.bulk.tensor (TMA)
// per box with mbarrier complete-tx (FA3's load path) instead of a 128-thread cp.async warpgroup; (2)
// smem K/V/Q tiles are 128B-swizzled and the consumer GEMMs use B128 wgmma descriptors (qk_gemm_sw /
// qkpv_issue_sw / pv_gemm_t_sw). full[]/barQ mbarriers have init count 1 (the expect_tx arrive); the
// TMA byte transaction flips the phase. Q,K,V are passed as CUtensorMaps over the [BH*S, HD] tensors.
__global__ __launch_bounds__(NThreadsWS, 1) void fused_attn_fwd_wst(
        const __grid_constant__ CUtensorMap mapQ, const __grid_constant__ CUtensorMap mapK,
        const __grid_constant__ CUtensorMap mapV, __nv_bfloat16* __restrict__ O,
        float* __restrict__ LSE, int S, int causal, float scale, int tiles_m) {
    const int m_block = blockIdx.x;
    const int bh = blockIdx.y;
    const int row0 = m_block * CTA_M;
    __nv_bfloat16* Ob = O + (size_t)bh * S * HD;
    float* LSEb = LSE + (size_t)bh * S;

    const int tid = threadIdx.x;
    const int n_max = causal ? (m_block + 1) : tiles_m;
    const float s2 = scale * float(M_LOG2E);
    constexpr int KVSTRIDE = CTA_N * HD;

    __nv_bfloat16* sQ = smem;
    __nv_bfloat16* sK0 = smem + CTA_M * HD;
    __nv_bfloat16* sV0 = smem + CTA_M * HD + WS_STAGES * KVSTRIDE;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + (CTA_M + 2 * WS_STAGES * CTA_N) * HD);
    uint64_t* full = bars;
    uint64_t* empty = bars + WS_STAGES;
    uint64_t* barQ = bars + 2 * WS_STAGES;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < WS_STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], NumCons); }
        mbar_init(&barQ[0], 1);
    }
    __syncthreads();

    const int wg = tid / WG;
    if (wg == 0) {
        // ---------------- producer: single thread issues TMA (FA3's SingleProducerWarp/elect_one) -------
        producer_regs();
        if (tid != 0) return;
        const int qrow = bh * S + row0;
        mbar_expect_tx(&barQ[0], (uint32_t)(CTA_M * HD * 2));
        tma_tile(sQ, &mapQ, qrow, &barQ[0]);
        for (int nb = 0; nb < n_max; ++nb) {
            const int st = nb % WS_STAGES;
            if (nb >= WS_STAGES) {
                const int ep = ((nb / WS_STAGES) - 1) & 1;
                while (!mbar_try_wait(&empty[st], ep)) {}
            }
            const int krow = bh * S + nb * CTA_N;
            mbar_expect_tx(&full[st], (uint32_t)(2 * CTA_N * HD * 2));   // K + V tiles
            tma_tile(sK0 + st * KVSTRIDE, &mapK, krow, &full[st]);
            tma_tile(sV0 + st * KVSTRIDE, &mapV, krow, &full[st]);
        }
        return;
    }

    // ---------------- consumers ----------------
    consumer_regs();
    const int cwg = wg - 1;
    const int wtid = tid % WG, warp = wtid >> 5, lane = wtid & 31;
    const int group = lane >> 2, tg = lane & 3;
    const int row_lo = cwg * 64 + warp * 16 + group, row_hi = row_lo + 8;
    const int qlo = row0 + row_lo, qhi = row0 + row_hi;
#ifdef DIAG_ST0
    const uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ);  // DIAG: drop cwg (uniform base, WRONG result)
#else
    const uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ) + (uint32_t)(cwg * 4096 * 2);  // swizzle row64
#endif
    const uint32_t sK_a = (uint32_t)__cvta_generic_to_shared(sK0);
    const uint32_t sV_a = (uint32_t)__cvta_generic_to_shared(sV0);
    const uint32_t st_bytes = (uint32_t)(KVSTRIDE * 2);
    const int SB_self = 1 + cwg, SB_other = 2 - cwg;

    float acc_o[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc_o[i] = 0.f;
    float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {0.f, 0.f};
    float s[64]; uint32_t P[32]; float sc[2];

    while (!mbar_try_wait(&barQ[0], 0)) {}
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

#ifdef SEQB
    // ---- SEQB: each consumer WG runs SEQUENTIAL async 8-wgmma batches (no intra-WG 2-groups-in-flight,
    // so ptxas keeps wgmma async: DEPBAR=2 not 32). Inter-WG PING-PONG (nbar) hides each WG's softmax
    // under the PEER WG's GEMM: issue QK (tensor), release token, wait QK, softmax(CUDA cores)||peer GEMM,
    // re-acquire token, issue PV, wait. Mirrors FA3 warp_scheduler_barrier ordering. ----
#ifndef NOPP
    if (cwg == 0) nbar_arrive256(SB_self);   // kickoff: cwg0 holds the tensor token first
    nbar_sync256(SB_self);                    // acquire token
#endif
    for (int nb = 0; nb < n_max; ++nb) {
        const int st = nb % WS_STAGES;
        const int fp = (nb / WS_STAGES) & 1;
        while (!mbar_try_wait(&full[st], fp)) {}
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#pragma unroll
        for (int i = 0; i < 64; ++i) s[i] = 0.f;
        qk_issue_sw(s, sQ_wg, sK_a + st * st_bytes);   // QK async (token held)
#ifndef NOPP
        nbar_arrive256(SB_other);                       // release token to peer WG
#endif
        wgmma_wait0();                                   // QK done
        if (causal && nb == m_block) mask_causal(s, nb * CTA_N, qlo, qhi, tg);
        sc[0] = softmax_stats(s, m_r, l_r, 0, nb, s2);   // overlaps peer WG's GEMM
        sc[1] = softmax_stats(s, m_r, l_r, 1, nb, s2);
        rescale_o(acc_o, sc);
#pragma unroll
        for (int i = 0; i < 32; ++i) P[i] = pack_bf16(s[2*i], s[2*i+1]);
#ifndef NOPP
        nbar_sync256(SB_self);                           // re-acquire token for PV
#endif
        pv_issue_sw(acc_o, P, sV_a + st * st_bytes);     // PV async
        wgmma_wait0();                                    // PV done
        mbar_arrive(&empty[st]);
    }
#ifndef NOPP
    nbar_arrive256(SB_other);                            // final release so peer WG can exit
#endif
#else
    // ---- prologue: QK(0)+softmax(0) ----
    while (!mbar_try_wait(&full[0], 0)) {}
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#pragma unroll
    for (int i = 0; i < 64; ++i) s[i] = 0.f;
    qk_gemm_sw(s, sQ_wg, sK_a);
    if (causal && m_block == 0) mask_causal(s, 0, qlo, qhi, tg);
    sc[0] = softmax_stats(s, m_r, l_r, 0, 0, s2);
    sc[1] = softmax_stats(s, m_r, l_r, 1, 0, s2);
#pragma unroll
    for (int i = 0; i < 32; ++i) P[i] = pack_bf16(s[2*i], s[2*i+1]);

#ifndef NOPP
    if (cwg == 0) nbar_arrive256(SB_self);
#endif

    // ---- mainloop: QK(nb)+softmax(nb) overlap in-flight PV(nb-1) ----
    for (int nb = 1; nb < n_max; ++nb) {
        const int st = nb % WS_STAGES, pst = (nb - 1) % WS_STAGES;
        const int fp = (nb / WS_STAGES) & 1;
        while (!mbar_try_wait(&full[st], fp)) {}
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#pragma unroll
        for (int i = 0; i < 64; ++i) s[i] = 0.f;
#ifndef NOPP
        nbar_sync256(SB_self);
#endif
        rescale_o(acc_o, sc);
#ifdef DIAG_ST0
        qkpv_issue_sw(s, acc_o, P, sQ_wg, sK_a, sV_a);  // DIAG: compile-time uniform base (WRONG result)
#elif defined(CUTE)
        qkpv_issue_cute(s, acc_o, P, sQ_wg, sK_a + st * st_bytes, sV_a + pst * st_bytes);
#elif defined(BLK)
        qkpv_issue_blk(s, acc_o, P, sQ_wg, sK_a + st * st_bytes, sV_a + pst * st_bytes);
#else
        qkpv_issue_sw(s, acc_o, P, sQ_wg, sK_a + st * st_bytes, sV_a + pst * st_bytes);
#endif
#ifndef NOPP
        nbar_arrive256(SB_other);
#endif
        wgmma_wait1();
        if (causal && nb == m_block) mask_causal(s, nb * CTA_N, qlo, qhi, tg);
        sc[0] = softmax_stats(s, m_r, l_r, 0, nb, s2);   // overlaps the in-flight PV(nb-1)
        sc[1] = softmax_stats(s, m_r, l_r, 1, nb, s2);
        wgmma_wait0();                                    // PV(nb-1) done: it no longer reads P
        // pack P for next iter's PV AFTER wait0. Overwriting P before wait0 is a WAR hazard vs the
        // in-flight PV(nb-1) (which reads P) -> ptxas serializes every wgmma (C7513, DEPBAR per HGMMA).
        // This single reorder is what keeps the intra-WG QK(nb)||PV(nb-1) overlap async, matching FA3
        // (which writes tOrP via convert_type_out only after its warpgroup_wait<0>).
#pragma unroll
        for (int i = 0; i < 32; ++i) P[i] = pack_bf16(s[2*i], s[2*i+1]);
        mbar_arrive(&empty[pst]);
    }

    // ---- epilogue: trailing PV(n_max-1) ----
    rescale_o(acc_o, sc);
    pv_gemm_t_sw(acc_o, P, sV_a + ((n_max - 1) % WS_STAGES) * st_bytes);
    mbar_arrive(&empty[(n_max - 1) % WS_STAGES]);
#endif  // SEQB

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l_r[r] = quad_sum(l_r[r]);
        float inv = (l_r[r] == 0.f || l_r[r] != l_r[r]) ? 0.f : 1.f / l_r[r];
        int qr = row0 + (r == 0 ? row_lo : row_hi);
#pragma unroll
        for (int c = 0; c < HD / 8; ++c) {
            int dv0 = 8 * c + 2 * tg;
            uint32_t o = pack_bf16(acc_o[c*4+r*2+0] * inv, acc_o[c*4+r*2+1] * inv);
            if (qr < S) *reinterpret_cast<uint32_t*>(&Ob[(size_t)qr * HD + dv0]) = o;
        }
        if (tg == 0 && qr < S)
            LSEb[qr] = (l_r[r] == 0.f || l_r[r] != l_r[r]) ? -INFINITY : m_r[r] * scale + __logf(l_r[r]);
    }
}

// ===================== WSTP: WST + FA3's persistent scheduler (StaticPersistentTileScheduler) ==========
// grid = #SMs (1-D). Each CTA strides over work tiles: work = blockIdx.x, += gridDim.x (tile_scheduler.hpp
// StaticPersistentTileScheduler::get_next_work). work -> (m_block = work % tiles_m, bh = work / tiles_m).
// The K/V smem ring phase is carried CONTINUOUSLY across tiles via a global block counter (gp/gc) so the
// pipeline never drains between tiles (flash_fwd_kernel_sm90.h: smem_pipe_read/write live outside the
// work loop). Q is single-buffered with a producer<-consumer handshake (barQ ready / qEmpty free), the
// analogue of FA3's QueryEmpty named barrier. Consumer ping-pong token is re-kicked per tile (named
// barriers are stateless; both consumer WGs run the same work sequence so they stay in lockstep).
__global__ __launch_bounds__(NThreadsWS, 1) void fused_attn_fwd_wstp(
        const __grid_constant__ CUtensorMap mapQ, const __grid_constant__ CUtensorMap mapK,
        const __grid_constant__ CUtensorMap mapV, __nv_bfloat16* __restrict__ O,
        float* __restrict__ LSE, int S, int causal, float scale, int tiles_m, int total_tiles) {
    const int tid = threadIdx.x;
    const float s2 = scale * float(M_LOG2E);
    constexpr int KVSTRIDE = CTA_N * HD;

    __nv_bfloat16* sQ = smem;
    __nv_bfloat16* sK0 = smem + CTA_M * HD;
    __nv_bfloat16* sV0 = smem + CTA_M * HD + WS_STAGES * KVSTRIDE;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + (CTA_M + 2 * WS_STAGES * CTA_N) * HD);
    uint64_t* full = bars;                  // [WS_STAGES]
    uint64_t* empty = bars + WS_STAGES;     // [WS_STAGES]
    uint64_t* barQ = bars + 2 * WS_STAGES;  // [1] Q ready
    uint64_t* qEmpty = bars + 2 * WS_STAGES + 1;  // [1] Q free (consumers done with QK)

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < WS_STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], NumCons); }
        mbar_init(&barQ[0], 1);
        mbar_init(&qEmpty[0], NumCons);
    }
    __syncthreads();

    const int wg = tid / WG;
    if (wg == 0) {
        // ---------------- producer: single thread issues all TMA, ring carried across tiles ----------
        producer_regs();
        if (tid != 0) return;
        int gp = 0;   // global K/V block counter (carries the ring phase across tiles)
        int ti = 0;
        for (int work = blockIdx.x; work < total_tiles; work += gridDim.x, ++ti) {
            const int m_block = work % tiles_m, bh = work / tiles_m;
            const int n_max = causal ? (m_block + 1) : tiles_m;
            if (ti > 0) { while (!mbar_try_wait(&qEmpty[0], (ti - 1) & 1)) {} }   // prev tile's Q free
            mbar_expect_tx(&barQ[0], (uint32_t)(CTA_M * HD * 2));
            tma_tile(sQ, &mapQ, bh * S + m_block * CTA_M, &barQ[0]);
            for (int nb = 0; nb < n_max; ++nb, ++gp) {
                const int st = gp % WS_STAGES;
                if (gp >= WS_STAGES) { while (!mbar_try_wait(&empty[st], ((gp / WS_STAGES) - 1) & 1)) {} }
                const int krow = bh * S + nb * CTA_N;
                mbar_expect_tx(&full[st], (uint32_t)(2 * CTA_N * HD * 2));
                tma_tile(sK0 + st * KVSTRIDE, &mapK, krow, &full[st]);
                tma_tile(sV0 + st * KVSTRIDE, &mapV, krow, &full[st]);
            }
        }
        return;
    }

    // ---------------- consumers ----------------
    consumer_regs();
    const int cwg = wg - 1;
    const int wtid = tid % WG, warp = wtid >> 5, lane = wtid & 31;
    const int group = lane >> 2, tg = lane & 3;
    const int row_lo = cwg * 64 + warp * 16 + group, row_hi = row_lo + 8;
    const uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ) + (uint32_t)(cwg * 4096 * 2);
    const uint32_t sK_a = (uint32_t)__cvta_generic_to_shared(sK0);
    const uint32_t sV_a = (uint32_t)__cvta_generic_to_shared(sV0);
    const uint32_t st_bytes = (uint32_t)(KVSTRIDE * 2);
    const int SB_self = 1 + cwg, SB_other = 2 - cwg;

    int gc = 0;   // global K/V block counter (matches producer's gp in lockstep)
    int ti = 0;
    for (int work = blockIdx.x; work < total_tiles; work += gridDim.x, ++ti) {
        const int m_block = work % tiles_m, bh = work / tiles_m;
        const int row0 = m_block * CTA_M;
        const int n_max = causal ? (m_block + 1) : tiles_m;
        const int qlo = row0 + row_lo, qhi = row0 + row_hi;
        __nv_bfloat16* Ob = O + (size_t)bh * S * HD;
        float* LSEb = LSE + (size_t)bh * S;

        float acc_o[64];
#pragma unroll
        for (int i = 0; i < 64; ++i) acc_o[i] = 0.f;
        float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {0.f, 0.f};
        float s[64]; uint32_t P[32]; float sc[2];

        while (!mbar_try_wait(&barQ[0], ti & 1)) {}   // Q for this tile ready
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

#ifdef SEQB
        // ---- SEQB: sequential async batches + inter-WG ping-pong (softmax || peer GEMM). One token
        // acquire/release pair per GEMM phase (QK, PV): 2 sync + 2 arrive per iter, balanced. ----
#ifndef NOPP
        if (cwg == 0) nbar_arrive256(SB_self);   // kickoff: cwg0 holds token first
        nbar_sync256(SB_self);                    // acquire for the first QK
#endif
        for (int nb = 0; nb < n_max; ++nb, ++gc) {
            const int st = gc % WS_STAGES;
            const int fp = (gc / WS_STAGES) & 1;
            while (!mbar_try_wait(&full[st], fp)) {}
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#pragma unroll
            for (int i = 0; i < 64; ++i) s[i] = 0.f;
            qk_issue_sw(s, sQ_wg, sK_a + st * st_bytes);   // QK async (token held)
#ifndef NOPP
            nbar_arrive256(SB_other);                       // release token -> peer GEMM
#endif
            wgmma_wait0();
            if (causal && nb == m_block) mask_causal(s, nb * CTA_N, qlo, qhi, tg);
            sc[0] = softmax_stats(s, m_r, l_r, 0, nb, s2);  // overlaps peer WG's GEMM
            sc[1] = softmax_stats(s, m_r, l_r, 1, nb, s2);
            rescale_o(acc_o, sc);
#pragma unroll
            for (int i = 0; i < 32; ++i) P[i] = pack_bf16(s[2*i], s[2*i+1]);
#ifndef NOPP
            nbar_sync256(SB_self);                           // re-acquire for PV
#endif
            pv_issue_sw(acc_o, P, sV_a + st * st_bytes);
            wgmma_wait0();
            mbar_arrive(&empty[st]);
        }
        mbar_arrive(&qEmpty[0]);   // Q free for next tile
#ifndef NOPP
        nbar_arrive256(SB_other);  // final release so peer WG can exit this tile
#endif
#else
        // ---- prologue: QK(0)+softmax(0) ----
        const int st0 = gc % WS_STAGES; const int fp0 = (gc / WS_STAGES) & 1;
        while (!mbar_try_wait(&full[st0], fp0)) {}
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#pragma unroll
        for (int i = 0; i < 64; ++i) s[i] = 0.f;
        qk_gemm_sw(s, sQ_wg, sK_a + st0 * st_bytes);
        if (causal && m_block == 0) mask_causal(s, 0, qlo, qhi, tg);
        sc[0] = softmax_stats(s, m_r, l_r, 0, 0, s2);
        sc[1] = softmax_stats(s, m_r, l_r, 1, 0, s2);
#pragma unroll
        for (int i = 0; i < 32; ++i) P[i] = pack_bf16(s[2*i], s[2*i+1]);
#ifndef NOPP
        if (cwg == 0) nbar_arrive256(SB_self);
#endif
        int gprev = gc;   // stage holding K/V of the block whose PV is still pending
        ++gc;

        // ---- mainloop: QK(nb)+softmax(nb) overlap in-flight PV(nb-1) ----
        for (int nb = 1; nb < n_max; ++nb, ++gc) {
            const int st = gc % WS_STAGES, pst = gprev % WS_STAGES;
            const int fp = (gc / WS_STAGES) & 1;
            while (!mbar_try_wait(&full[st], fp)) {}
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#pragma unroll
            for (int i = 0; i < 64; ++i) s[i] = 0.f;
#ifndef NOPP
            nbar_sync256(SB_self);
#endif
            rescale_o(acc_o, sc);
#if defined(CUTE)
            qkpv_issue_cute(s, acc_o, P, sQ_wg, sK_a + st * st_bytes, sV_a + pst * st_bytes);
#else
            qkpv_issue_sw(s, acc_o, P, sQ_wg, sK_a + st * st_bytes, sV_a + pst * st_bytes);
#endif
#ifndef NOPP
            nbar_arrive256(SB_other);
#endif
            wgmma_wait1();
            if (causal && nb == m_block) mask_causal(s, nb * CTA_N, qlo, qhi, tg);
            sc[0] = softmax_stats(s, m_r, l_r, 0, nb, s2);
            sc[1] = softmax_stats(s, m_r, l_r, 1, nb, s2);
            wgmma_wait0();                 // PV(nb-1) done before overwriting P (avoid WAR -> serialization)
#pragma unroll
            for (int i = 0; i < 32; ++i) P[i] = pack_bf16(s[2*i], s[2*i+1]);
            mbar_arrive(&empty[pst]);
            gprev = gc;
        }
        // Q no longer needed (all QKs issued) -> let the producer load the next tile's Q.
        mbar_arrive(&qEmpty[0]);

        // ---- epilogue: trailing PV(n_max-1) ----
        rescale_o(acc_o, sc);
        pv_gemm_t_sw(acc_o, P, sV_a + (gprev % WS_STAGES) * st_bytes);
        mbar_arrive(&empty[gprev % WS_STAGES]);
#endif  // SEQB

#pragma unroll
        for (int r = 0; r < 2; ++r) {
            l_r[r] = quad_sum(l_r[r]);
            float inv = (l_r[r] == 0.f || l_r[r] != l_r[r]) ? 0.f : 1.f / l_r[r];
            int qr = row0 + (r == 0 ? row_lo : row_hi);
#pragma unroll
            for (int c = 0; c < HD / 8; ++c) {
                int dv0 = 8 * c + 2 * tg;
                uint32_t o = pack_bf16(acc_o[c*4+r*2+0] * inv, acc_o[c*4+r*2+1] * inv);
                if (qr < S) *reinterpret_cast<uint32_t*>(&Ob[(size_t)qr * HD + dv0]) = o;
            }
            if (tg == 0 && qr < S)
                LSEb[qr] = (l_r[r] == 0.f || l_r[r] != l_r[r]) ? -INFINITY : m_r[r] * scale + __logf(l_r[r]);
        }
        // Token state reset for the next tile: the per-tile kickoff (cwg0 arrive SB_self) leaves exactly
        // 128 pending arrivals on barrier 1 (= cwg0 self / cwg1 other); barrier 2 ends clean. cwg1 arrives
        // barrier 1 once to clear it (without this the residual accumulates across tiles and the ping-pong
        // deadlocks). The 256-thread consumer rendezvous (id 3) is needed in BOTH modes: it keeps the two
        // consumer WGs in lockstep across tiles so the per-tile qEmpty arrive (count 256) never mixes
        // arrivals from two different tiles.
#ifndef NOPP
        if (cwg == 1) nbar_arrive256(SB_other);   // SB_other(cwg1) == barrier 1
#endif
        nbar_sync256(3);
    }
}

// Build a CUtensorMap over a [BH*S, HD] bf16 tensor (HD contiguous), 2 swizzle boxes of {64 inner, 128
// rows}. Same shape works for Q/K/V (all [rows, HD] row-major); the wgmma descriptor decides majorness.
static CUtensorMap make_tma_map(const __nv_bfloat16* ptr, uint64_t rows) {
    CUtensorMap map{};
    uint64_t gdim[2] = {(uint64_t)HD, rows};
    uint64_t gstride[1] = {(uint64_t)HD * sizeof(__nv_bfloat16)};
    uint32_t bdim[2] = {64u, (uint32_t)CTA_N};
    uint32_t estride[2] = {1, 1};
    CUresult cr = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (cr != CUDA_SUCCESS) { const char* s; cuGetErrorString(cr, &s);
        std::fprintf(stderr, "cuTensorMapEncodeTiled: %s\n", s); std::exit(2); }
    return map;
}

int main(int argc, char** argv) {
    int Slen = argc > 1 ? std::atoi(argv[1]) : 2048;
    int D    = argc > 2 ? std::atoi(argv[2]) : 128;
    int causal = argc > 3 ? std::atoi(argv[3]) : 1;
    int B = argc > 4 ? std::atoi(argv[4]) : 1;
    int H = argc > 5 ? std::atoi(argv[5]) : 1;
    int iters  = argc > 6 ? std::atoi(argv[6]) : 100;
    int trials = argc > 7 ? std::atoi(argv[7]) : 7;
    if (D != 128 || Slen % 128) { std::fprintf(stderr, "need D==128, S%%128==0\n"); return 1; }
    const int BH = B * H;
    const float scale = 1.f / std::sqrt(float(D));

    std::vector<uint16_t> hQ, hK, hV;
    orc::fill_bf16(hQ, size_t(BH) * Slen * D, 11);
    orc::fill_bf16(hK, size_t(BH) * Slen * D, 22);
    orc::fill_bf16(hV, size_t(BH) * Slen * D, 33);

    __nv_bfloat16 *dQ, *dK, *dV, *dO; float* dLSE;
    size_t nelem = size_t(BH) * Slen * D;
    ORC_CHECK(cudaMalloc(&dQ, sizeof(__nv_bfloat16) * nelem));
    ORC_CHECK(cudaMalloc(&dK, sizeof(__nv_bfloat16) * nelem));
    ORC_CHECK(cudaMalloc(&dV, sizeof(__nv_bfloat16) * nelem));
    ORC_CHECK(cudaMalloc(&dO, sizeof(__nv_bfloat16) * nelem));
    ORC_CHECK(cudaMalloc(&dLSE, sizeof(float) * size_t(BH) * Slen));
    ORC_CHECK(cudaMemcpy(dQ, hQ.data(), sizeof(__nv_bfloat16) * nelem, cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dK, hK.data(), sizeof(__nv_bfloat16) * nelem, cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dV, hV.data(), sizeof(__nv_bfloat16) * nelem, cudaMemcpyHostToDevice));

    const int tiles_m = Slen / CTA_M;
    int nthreads = NThreads;
#if defined(WSTP)
    auto kernel = fused_attn_fwd_wstp;
    nthreads = NThreadsWS;
    size_t smem_bytes = sizeof(__nv_bfloat16) * (CTA_M + 2 * WS_STAGES * CTA_N) * HD
                        + (2 * WS_STAGES + 2) * sizeof(uint64_t);
#elif defined(WST)
    auto kernel = fused_attn_fwd_wst;
    nthreads = NThreadsWS;
    size_t smem_bytes = sizeof(__nv_bfloat16) * (CTA_M + 2 * WS_STAGES * CTA_N) * HD
                        + (2 * WS_STAGES + 1) * sizeof(uint64_t);
#elif defined(WSO)
    auto kernel = fused_attn_fwd_ws_ovl;
    nthreads = NThreadsWS;
    // sQ + WS_STAGES*(sK + sV) (bf16) + (2*WS_STAGES+1) mbarriers (8B each)
    size_t smem_bytes = sizeof(__nv_bfloat16) * (CTA_M + 2 * WS_STAGES * CTA_N) * HD
                        + (2 * WS_STAGES + 1) * sizeof(uint64_t);
#elif defined(WS)
    auto kernel = fused_attn_fwd_ws;
    nthreads = NThreadsWS;
    // sQ + 2*sK + 2*sV (bf16) + 5 mbarriers (8B each)
    size_t smem_bytes = sizeof(__nv_bfloat16) * (CTA_M + 4 * CTA_N) * HD + 5 * sizeof(uint64_t);
#elif defined(OVL)
    auto kernel = fused_attn_fwd_ovl;
    size_t smem_bytes = sizeof(__nv_bfloat16) * (CTA_M + 2 * CTA_N + 2 * CTA_N) * HD;  // sQ + 2*sK + 2*sVt
#elif defined(PIPE)
    auto kernel = fused_attn_fwd_pipe;
    size_t smem_bytes = sizeof(__nv_bfloat16) * (CTA_M + 2 * CTA_N + CTA_N) * HD;  // sQ + 2*sK + sVt
#else
    auto kernel = fused_attn_fwd;
    size_t smem_bytes = sizeof(__nv_bfloat16) * (CTA_M + CTA_N + CTA_N) * HD;       // sQ + sK + sVt
#endif
    if (smem_bytes >= 48 * 1024)
        ORC_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
#if defined(WSTP)
    int num_sm = 0;
    ORC_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    const int total_tiles = tiles_m * BH;
    dim3 grid(num_sm);
    CUtensorMap mapQ = make_tma_map(dQ, (uint64_t)BH * Slen);
    CUtensorMap mapK = make_tma_map(dK, (uint64_t)BH * Slen);
    CUtensorMap mapV = make_tma_map(dV, (uint64_t)BH * Slen);
    auto launch = [&]() {
        kernel<<<grid, nthreads, smem_bytes>>>(mapQ, mapK, mapV, dO, dLSE, Slen, causal, scale, tiles_m, total_tiles);
    };
#elif defined(WST)
    dim3 grid(tiles_m, BH);
    CUtensorMap mapQ = make_tma_map(dQ, (uint64_t)BH * Slen);
    CUtensorMap mapK = make_tma_map(dK, (uint64_t)BH * Slen);
    CUtensorMap mapV = make_tma_map(dV, (uint64_t)BH * Slen);
    auto launch = [&]() {
        kernel<<<grid, nthreads, smem_bytes>>>(mapQ, mapK, mapV, dO, dLSE, Slen, causal, scale, tiles_m);
    };
#else
    dim3 grid(tiles_m, BH);
    auto launch = [&]() {
        kernel<<<grid, nthreads, smem_bytes>>>(dQ, dK, dV, dO, dLSE, Slen, causal, scale, tiles_m);
    };
#endif
    launch();
    ORC_CHECK(cudaDeviceSynchronize());
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "launch: %s\n", cudaGetErrorString(e)); return 3; }

    // correctness (bh=0 slice) vs CPU fp32 reference
    double o_max = -1, o_mean = -1, l_max = -1, l_mean = -1;
    {
        attn::Problem p{Slen, D, D, (bool)causal, scale};
        std::vector<uint16_t> Q0(hQ.begin(), hQ.begin() + size_t(Slen) * D);
        std::vector<uint16_t> K0(hK.begin(), hK.begin() + size_t(Slen) * D);
        std::vector<uint16_t> V0(hV.begin(), hV.begin() + size_t(Slen) * D);
        std::vector<float> O_ref, LSE_ref;
        attn::ref_cpu(p, Q0, K0, V0, O_ref, LSE_ref);
        std::vector<uint16_t> hO(size_t(Slen) * D); std::vector<float> hLSE(Slen);
        ORC_CHECK(cudaMemcpy(hO.data(), dO, sizeof(uint16_t) * hO.size(), cudaMemcpyDeviceToHost));
        ORC_CHECK(cudaMemcpy(hLSE.data(), dLSE, sizeof(float) * Slen, cudaMemcpyDeviceToHost));
        std::vector<float> Of(hO.size());
        for (size_t i = 0; i < hO.size(); ++i) Of[i] = orc::bf16_bits_to_f32(hO[i]);
        attn::diff(Of, O_ref, o_max, o_mean);
        attn::diff(hLSE, LSE_ref, l_max, l_mean);
    }

    float median = 0.f;
    float best = orc::benchmark_ms(launch, 50, iters, trials, &median);
    // FLOPs: causal ~ 0.5; per (b,h): 2 GEMMs * 2*S*S*D, plus PV 2*S*S*Dv
    double frac = causal ? 0.5 : 1.0;
    double flop = double(BH) * 2.0 * (2.0 * Slen * Slen * D) * frac;   // QK + PV
    std::printf("{\"comp\": \"fused_attn\", \"S\": %d, \"D\": %d, \"causal\": %d, \"B\": %d, \"H\": %d, "
                "\"min_ms\": %.5f, \"median_ms\": %.5f, \"tflops\": %.1f, "
                "\"O_max_abs\": %.3e, \"O_mean_abs\": %.3e, \"LSE_max_abs\": %.3e}\n",
                Slen, D, causal, B, H, best, median, flop / (median * 1e-3) / 1e12,
                o_max, o_mean, l_max);
    return (o_max < 5e-3 && l_max < 5e-3) ? 0 : 5;
}
