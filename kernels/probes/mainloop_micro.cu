// Isolated hd128 bf16 consumer-mainloop microbenchmark (torch-free, NCU-able directly).
// Reproduces the EXACT instruction schedule of fa3_hopper.cu's consumer mainloop (QK m64n128k16 ->
// softmax(exp2) -> PV m64n64k16 x2, register-resident acc_o, RS-P, optional FA3 ping-pong), with
// the producer/TMA/scheduler removed and K/V pre-staged in smem. Goal: measure the mainloop's
// tensor-pipe utilization + register count in isolation, to localize the chunk16k 52% vs FA3 76% gap.
//
// Build:  nvcc -O3 -arch=sm_90a -o /tmp/mlmicro kernels/mainloop_micro.cu
// Run:    /tmp/mlmicro [N_tiles=128] [blocks=132]
// NCU:    ncu --set full -c 1 -s 1 /tmp/mlmicro 128 132
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

#ifndef PINGPONG
#define PINGPONG 1
#endif
// SCHED selects the mainloop variant: 0 = current fa3_hopper.cu schedule (baseline).
#ifndef SCHED
#define SCHED 0
#endif

constexpr int CTA_M = 128, CTA_N = 128, HD = 128, HDV = 128, CN = 128, CM = 128, WG = 128, STAGES = 3;
constexpr int HDsm = 128, HDVsm = 128, KSTRIDE = CN * HDsm, VSTRIDE = CN * HDVsm;

// ---------- exact helpers copied from fa3_hopper.cu (hd128 bf16 subset) ----------
template <int ROWS> __device__ __forceinline__ uint64_t qk_doffR(int kb) { return (uint64_t)((kb & 3) * 2 + (kb >> 2) * (ROWS * 8)); }
__device__ __forceinline__ uint64_t v_doff(int kb) { return (uint64_t)(kb * 128); }
constexpr uint32_t SW_QK_LBO = 16, SW_QK_SBO = 1024;
constexpr uint32_t SW_V_LBO = 16384, SW_V_SBO = 1024;

__device__ __forceinline__ uint64_t make_smem_desc(uint32_t a, uint32_t lbo, uint32_t sbo, uint32_t lt) {
    uint64_t d = 0;
    d |= (uint64_t)((a >> 4) & 0x3FFFu);
    d |= (uint64_t)((lbo >> 4) & 0x3FFFu) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFFu) << 32;
    d |= (uint64_t)(lt & 0x3u) << 62;
    return d;
}
__device__ __forceinline__ uint64_t desc_add(uint64_t base, uint32_t off) {
    uint32_t lo, hi;
    asm("mov.b64 {%0,%1}, %2;\n" : "=r"(lo), "=r"(hi) : "l"(base));
    lo += off;
    uint64_t r; asm("mov.b64 %0, {%1,%2};\n" : "=l"(r) : "r"(lo), "r"(hi));
    return r;
}
__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat16 a = __float2bfloat16_rn(lo), b = __float2bfloat16_rn(hi);
    uint16_t ua, ub; memcpy(&ua, &a, 2); memcpy(&ub, &b, 2);
    return uint32_t(ua) | (uint32_t(ub) << 16);
}
__device__ __forceinline__ float ex2_approx(float x) {
    float r; asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait1() { asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory"); }

__device__ __forceinline__ float softmax_stats(float (&s)[CN/2], float* m_r, float* l_r, int r, int nb, float s2) {
    float b0 = -INFINITY, b1 = -INFINITY, b2 = -INFINITY, b3 = -INFINITY;
#pragma unroll
    for (int c = 0; c < CN / 8; c += 2) {
        b0 = fmaxf(b0, s[c*4+r*2+0]);     b1 = fmaxf(b1, s[c*4+r*2+1]);
        b2 = fmaxf(b2, s[(c+1)*4+r*2+0]); b3 = fmaxf(b3, s[(c+1)*4+r*2+1]);
    }
    float bm = fmaxf(fmaxf(b0, b1), fmaxf(b2, b3));
    bm = quad_max(bm);
    float scale_f;
    if (nb == 0) { m_r[r] = bm; scale_f = 1.f; }
    else { float prev = m_r[r]; m_r[r] = fmaxf(m_r[r], bm);
           float cur = (m_r[r] == -INFINITY) ? 0.f : m_r[r];
           scale_f = ex2_approx((prev - cur) * s2); l_r[r] *= scale_f; }
    float max_scaled = (m_r[r] == -INFINITY) ? 0.f : m_r[r] * s2;
    float ls0 = 0.f, ls1 = 0.f, ls2 = 0.f, ls3 = 0.f;
#pragma unroll
    for (int c = 0; c < CN / 8; c += 2) {
        float p0 = ex2_approx(s[c*4+r*2+0]     * s2 - max_scaled);
        float p1 = ex2_approx(s[c*4+r*2+1]     * s2 - max_scaled);
        float p2 = ex2_approx(s[(c+1)*4+r*2+0] * s2 - max_scaled);
        float p3 = ex2_approx(s[(c+1)*4+r*2+1] * s2 - max_scaled);
        s[c*4+r*2+0] = p0;     s[c*4+r*2+1] = p1;
        s[(c+1)*4+r*2+0] = p2; s[(c+1)*4+r*2+1] = p3;
        ls0 += p0; ls1 += p1; ls2 += p2; ls3 += p3;
    }
    float lsum = (ls0 + ls1) + (ls2 + ls3);
    l_r[r] = (nb == 0) ? lsum : l_r[r] + lsum;
    return scale_f;
}
__device__ __forceinline__ void pf_mask(float (&s)[CN/2], int col0, int qlo, int qhi, int seqlen_k, int causal, int tg) {
#pragma unroll
    for (int c = 0; c < CN / 8; ++c) {
        int k0 = col0 + 8 * c + 2 * tg;
        if (k0     >= seqlen_k || (causal && k0     > qlo)) s[c*4+0] = -INFINITY;
        if (k0 + 1 >= seqlen_k || (causal && k0 + 1 > qlo)) s[c*4+1] = -INFINITY;
        if (k0     >= seqlen_k || (causal && k0     > qhi)) s[c*4+2] = -INFINITY;
        if (k0 + 1 >= seqlen_k || (causal && k0 + 1 > qhi)) s[c*4+3] = -INFINITY;
    }
}
__device__ __forceinline__ void producer_regs() { asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory"); }
template <int R = 232>
__device__ __forceinline__ void consumer_regs() {
    if constexpr (R <= 128)      asm volatile("setmaxnreg.inc.sync.aligned.u32 128;\n" ::: "memory");
    else if constexpr (R <= 168) asm volatile("setmaxnreg.inc.sync.aligned.u32 168;\n" ::: "memory");
    else                         asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
}
__device__ __forceinline__ void nb_sync(int id, int cnt) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(cnt) : "memory"); }
__device__ __forceinline__ void nb_arrive(int id, int cnt) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(cnt) : "memory"); }

#define PF_ACC64 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
"%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31," \
"%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47," \
"%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}"
#define PF_ACC32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
"%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
#define PF_OUT64(a) \
        "+f"(a[0]),"+f"(a[1]),"+f"(a[2]),"+f"(a[3]),"+f"(a[4]),"+f"(a[5]),"+f"(a[6]),"+f"(a[7]), \
        "+f"(a[8]),"+f"(a[9]),"+f"(a[10]),"+f"(a[11]),"+f"(a[12]),"+f"(a[13]),"+f"(a[14]),"+f"(a[15]), \
        "+f"(a[16]),"+f"(a[17]),"+f"(a[18]),"+f"(a[19]),"+f"(a[20]),"+f"(a[21]),"+f"(a[22]),"+f"(a[23]), \
        "+f"(a[24]),"+f"(a[25]),"+f"(a[26]),"+f"(a[27]),"+f"(a[28]),"+f"(a[29]),"+f"(a[30]),"+f"(a[31]), \
        "+f"(a[32]),"+f"(a[33]),"+f"(a[34]),"+f"(a[35]),"+f"(a[36]),"+f"(a[37]),"+f"(a[38]),"+f"(a[39]), \
        "+f"(a[40]),"+f"(a[41]),"+f"(a[42]),"+f"(a[43]),"+f"(a[44]),"+f"(a[45]),"+f"(a[46]),"+f"(a[47]), \
        "+f"(a[48]),"+f"(a[49]),"+f"(a[50]),"+f"(a[51]),"+f"(a[52]),"+f"(a[53]),"+f"(a[54]),"+f"(a[55]), \
        "+f"(a[56]),"+f"(a[57]),"+f"(a[58]),"+f"(a[59]),"+f"(a[60]),"+f"(a[61]),"+f"(a[62]),"+f"(a[63])
#define PF_OUT32(a, O) \
        "+f"(a[(O)+0]),"+f"(a[(O)+1]),"+f"(a[(O)+2]),"+f"(a[(O)+3]),"+f"(a[(O)+4]),"+f"(a[(O)+5]),"+f"(a[(O)+6]),"+f"(a[(O)+7]), \
        "+f"(a[(O)+8]),"+f"(a[(O)+9]),"+f"(a[(O)+10]),"+f"(a[(O)+11]),"+f"(a[(O)+12]),"+f"(a[(O)+13]),"+f"(a[(O)+14]),"+f"(a[(O)+15]), \
        "+f"(a[(O)+16]),"+f"(a[(O)+17]),"+f"(a[(O)+18]),"+f"(a[(O)+19]),"+f"(a[(O)+20]),"+f"(a[(O)+21]),"+f"(a[(O)+22]),"+f"(a[(O)+23]), \
        "+f"(a[(O)+24]),"+f"(a[(O)+25]),"+f"(a[(O)+26]),"+f"(a[(O)+27]),"+f"(a[(O)+28]),"+f"(a[(O)+29]),"+f"(a[(O)+30]),"+f"(a[(O)+31])

template <int N> __device__ __forceinline__ void wg_fence_c(float (&c)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+f"(c[i]) :: "memory");
}
template <int M = 32> __device__ __forceinline__ void wg_fence_a(const uint32_t (&a)[M]) {
#pragma unroll
    for (int i = 0; i < M; ++i) asm volatile("" : "+r"(const_cast<uint32_t&>(a[i])) :: "memory");
}
// QK: m64n128k16 over HD/16 k-steps (CM=CN=128 -> qk_doffR<128> for both A and B).
__device__ __forceinline__ void qk128(float (&s)[64], uint64_t baseA, uint64_t baseB, int p0) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < HD / 16; ++kb) {
        uint64_t dA = desc_add(baseA, (uint32_t)qk_doffR<128>(kb));
        uint64_t dB = desc_add(baseB, (uint32_t)qk_doffR<128>(kb));
        int scaleD = (kb == 0) ? p0 : 1;
        asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " PF_ACC64 ",%64,%65,p,%67,%68,%69,%70;\n}\n"
        : PF_OUT64(s) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(0));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
// PV box: 8x m64n64k16 accumulating into acc[O..O+31].
template <int O> __device__ __forceinline__ void pvbox(float (&acc)[64], const uint32_t (&P)[32], uint64_t baseV) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < CN / 16; ++kb) {
        uint64_t dB = desc_add(baseV, (uint32_t)v_doff(kb));
        asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " PF_ACC32 ",{%32,%33,%34,%35},%36,p,%38,%39,%40;\n}\n"
        : PF_OUT32(acc, O)
        : "r"(P[kb*4+0]),"r"(P[kb*4+1]),"r"(P[kb*4+2]),"r"(P[kb*4+3]),"l"(dB),"r"(1),"n"(1),"n"(1),"n"(1));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void pvboxes(float (&acc_o)[64], const uint32_t (&P)[32], uint32_t sV_base) {
    constexpr uint32_t LBO = (uint32_t)(CN * 128), BOX = (uint32_t)(CN * 128);
    pvbox<0>(acc_o, P, make_smem_desc(sV_base, LBO, SW_V_SBO, 1));
    pvbox<32>(acc_o, P, make_smem_desc(sV_base + 1 * BOX, LBO, SW_V_SBO, 1));
}
__device__ __forceinline__ void rescale_o_g(float (&acc_o)[64], const float* sc) {
#pragma unroll
    for (int c = 0; c < 64 / 4; ++c) {
        acc_o[c*4+0] *= sc[0]; acc_o[c*4+1] *= sc[0];
        acc_o[c*4+2] *= sc[1]; acc_o[c*4+3] *= sc[1];
    }
}
__device__ __forceinline__ void qkgemm(float (&s)[64], uint32_t sQ_wg, uint32_t sK_base) {
    uint64_t baseA = make_smem_desc(sQ_wg, SW_QK_LBO, SW_QK_SBO, 1);
    uint64_t baseB = make_smem_desc(sK_base, SW_QK_LBO, SW_QK_SBO, 1);
    qk128(s, baseA, baseB, 0);
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    wg_fence_c(s);
}
__device__ __forceinline__ void qkrspv(float (&s)[64], float (&acc_o)[64], const uint32_t (&P)[32],
                                       uint32_t sQ_wg, uint32_t sK_base, uint32_t sV_base, const float* sc) {
    uint64_t baseA = make_smem_desc(sQ_wg, SW_QK_LBO, SW_QK_SBO, 1);
    uint64_t baseB = make_smem_desc(sK_base, SW_QK_LBO, SW_QK_SBO, 1);
    qk128(s, baseA, baseB, 0);
    rescale_o_g(acc_o, sc);
    pvboxes(acc_o, P, sV_base);
}
__device__ __forceinline__ void pvfinal(float (&acc_o)[64], const uint32_t (&P)[32], uint32_t sV_base) {
    pvboxes(acc_o, P, sV_base);
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    wg_fence_c(acc_o);
    wg_fence_a(P);
}

extern __shared__ __nv_bfloat16 smem[];

__global__ __launch_bounds__(384, 1) void micro(float* __restrict__ gO, int N) {
    __nv_bfloat16* sQ = smem;
    __nv_bfloat16* sK0 = sQ + CM * HDsm;
    __nv_bfloat16* sV0 = sK0 + STAGES * KSTRIDE;
    const int tid = threadIdx.x;
    const int total = CM * HDsm + STAGES * KSTRIDE + STAGES * VSTRIDE;
    for (int i = tid; i < total; i += 384) smem[i] = __float2bfloat16_rn(0.0078125f * ((i & 31) - 16));
    __syncthreads();

    const int wg = tid / WG;
    if (wg == 0) { producer_regs(); return; }
    consumer_regs<232>();
    const int cwg = wg - 1;
    const int wtid = tid % WG, warp = wtid >> 5, lane = wtid & 31, group = lane >> 2, tg = lane & 3;
    const uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ) + (uint32_t)(cwg * 4096 * 2);
    const uint32_t sK_a = (uint32_t)__cvta_generic_to_shared(sK0);
    const uint32_t sV_a = (uint32_t)__cvta_generic_to_shared(sV0);
    const uint32_t stK = (uint32_t)(KSTRIDE * 2), stV = (uint32_t)(VSTRIDE * 2);
    (void)warp; (void)group;

    float acc_o[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc_o[i] = 0.f;
    float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {0.f, 0.f};
    float s[64]; uint32_t P[32]; float sc[2];
    const int seqlen_k = 1 << 20, causal = 1, qlo = 1 << 20, qhi = 1 << 20;  // interior tile: nothing masked
    const float s2 = 0.0883883f;

#if PINGPONG
    if (cwg == 0) nb_arrive(8, 256);
#endif
    // first tile (QK-only)
#pragma unroll
    for (int i = 0; i < 64; ++i) s[i] = 0.f;
    qkgemm(s, sQ_wg, sK_a + 0 * stK);
    pf_mask(s, 0, qlo, qhi, seqlen_k, causal, tg);
    sc[0] = softmax_stats(s, m_r, l_r, 0, 0, s2);
    sc[1] = softmax_stats(s, m_r, l_r, 1, 0, s2);
#pragma unroll
    for (int i = 0; i < 32; ++i) P[i] = pack_bf16(s[2*i], s[2*i+1]);

    for (int li = 1; li < N; ++li) {
        const int st = li % STAGES, pst = (li - 1) % STAGES;
#if PINGPONG
        nb_sync(8 + cwg, 256);
#endif
        qkrspv(s, acc_o, P, sQ_wg, sK_a + st * stK, sV_a + pst * stV, sc);
#if PINGPONG
        nb_arrive(8 + (1 - cwg), 256);
#endif
        wgmma_wait1();
        pf_mask(s, li * 128, qlo, qhi, seqlen_k, causal, tg);
        sc[0] = softmax_stats(s, m_r, l_r, 0, li, s2);
        sc[1] = softmax_stats(s, m_r, l_r, 1, li, s2);
        wgmma_wait0();
#pragma unroll
        for (int i = 0; i < 32; ++i) P[i] = pack_bf16(s[2*i], s[2*i+1]);
    }
    rescale_o_g(acc_o, sc);
    pvfinal(acc_o, P, sV_a + ((N - 1) % STAGES) * stV);

    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) acc += acc_o[i];
    acc += m_r[0] + m_r[1] + l_r[0] + l_r[1];
    gO[blockIdx.x * 384 + tid] = acc;
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 128;
    int blocks = argc > 2 ? atoi(argv[2]) : 132;
    int smem = (CM * HDsm + STAGES * KSTRIDE + STAGES * VSTRIDE) * (int)sizeof(__nv_bfloat16);
    cudaError_t e = cudaFuncSetAttribute(micro, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) { printf("setattr smem=%d: %s\n", smem, cudaGetErrorString(e)); return 1; }
    float* gO; cudaMalloc(&gO, sizeof(float) * (size_t)blocks * 384);

    micro<<<blocks, 384, smem>>>(gO, N);     // warmup
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("launch: %s\n", cudaGetErrorString(e)); return 1; }

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int iters = 300;
    cudaEventRecord(a);
    for (int it = 0; it < iters; ++it) micro<<<blocks, 384, smem>>>(gO, N);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    printf("micro N=%d blocks=%d smem=%dKB PINGPONG=%d SCHED=%d: %.3f us/launch\n",
           N, blocks, smem >> 10, PINGPONG, SCHED, ms / iters * 1000.0);
    cudaDeviceSynchronize();
    return 0;
}
