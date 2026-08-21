// fa4_qk_gemm.cu — Blackwell SM100 QK GEMM building block: S = Q·Kᵀ via tcgen05.mma into tmem.
//
// Validates the hardest FA4 primitive: bf16 SS-form UMMA (both operands from 128B-swizzled smem
// descriptors) accumulating F32 into tensor memory, then reading S back and checking vs an fp32 ref.
//
// Reuses FA3's proven, byte-exact 128B swizzle (koff) + per-k-tile descriptor advance (qk_doff),
// which — verified against the reference PTX — is IDENTICAL for SM90 wgmma and SM100 tcgen05
// (same Swizzle<3,4,3> atom). The MMA descriptor constants were extracted from the compiled FA4
// reference:  idesc(QK)=0x8200490, hi=0x40004040 (SBO=1024>>4=64, ver=1, layout_type=2=128B),
// lo = (addr>>4 & 0x3FC0) | (1<<16) [LBO=16>>4=1].
//
// Since the 128B swizzle is address-based and self-consistent, we fill smem with plain swizzled
// stores (no TMA) and the MMA reads it back correctly. Tile: M=N=128, K=head_dim=128 (8 k-tiles).
//
// Build: nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 fa4_qk_gemm.cu -o /tmp/qk
#include "probe_common.cuh"

// ---------- tcgen05 tmem intrinsics ----------
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void tmem_alloc(uint32_t s, int n){ asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(s),"r"(n)); }
__device__ __forceinline__ void tmem_relinquish(){ asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int n){ asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(n)); }
__device__ __forceinline__ uint32_t tmem_ld_1(uint32_t a){ uint32_t v; asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];":"=r"(v):"r"(a)); return v; }
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;"); }

// ---------- 128B swizzle (K-major, from FA3, ROWS=128) ----------
__device__ __forceinline__ int koff(int row, int c){ return (c>>6)*8192 + row*64 + ((c&63) ^ ((row&7)<<3)); }

// ---------- SM100 64-bit smem matrix descriptor ----------
__device__ __forceinline__ uint64_t make_desc_sm100(uint32_t addr, uint32_t lbo_bytes, uint32_t sbo_bytes, int lt){
    uint64_t d = 0;
    d |= (uint64_t)((addr>>4) & 0x3FC0u);            // start_address [0,14) (1024B-atom aligned)
    d |= (uint64_t)((lbo_bytes>>4) & 0x3FFFu) << 16; // leading_byte_offset [16,30)
    d |= (uint64_t)((sbo_bytes>>4) & 0x3FFFu) << 32; // stride_byte_offset  [32,46)
    d |= (uint64_t)1u << 46;                          // version = 1
    d |= (uint64_t)(lt & 0x7u) << 61;                 // layout_type [61,64)
    return d;
}
__device__ __forceinline__ uint64_t desc_add(uint64_t base, uint32_t off){
    uint32_t lo,hi; asm("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"l"(base));
    lo += off; uint64_t r; asm("mov.b64 %0, {%1,%2};":"=l"(r):"r"(lo),"r"(hi)); return r;
}
// per-k-tile descriptor advance (16-byte units), ROWS=128: {0,2,4,6,0x400,0x402,0x404,0x406}
__device__ __forceinline__ uint32_t qk_doff(int kb){ return (uint32_t)((kb&3)*2 + (kb>>2)*1024); }

constexpr int M=128, N=128, K=128, NKT=K/16;      // 8 k-tiles of MMA-K=16
constexpr uint32_t QK_IDESC = 0x8200490u;          // bf16 A/B, F32 acc, K-major, M=128,N=128
constexpr uint32_t LBO=16, SBO=1024;

__global__ void __launch_bounds__(128) qk_kernel(const __nv_bfloat16* Qg, const __nv_bfloat16* Kg, float* Sout){
    __shared__ __align__(1024) __nv_bfloat16 sQ[M*K];
    __shared__ __align__(1024) __nv_bfloat16 sK[N*K];
    __shared__ __align__(8) uint32_t s_base;
    __shared__ __align__(8) uint64_t s_mbar;
    const int tid = threadIdx.x, warp = tid>>5, lane = tid&31;

    // 1) fill swizzled smem (each thread strides over the tile).
    for (int i = tid; i < M*K; i += 128) { int r=i/K, c=i%K; sQ[koff(r,c)] = Qg[r*K+c]; }
    for (int i = tid; i < N*K; i += 128) { int r=i/K, c=i%K; sK[koff(r,c)] = Kg[r*K+c]; }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();

    // 2) tmem alloc + mbarrier init (warp 0 / thread 0).
    if (warp==0) { tmem_alloc(smem_u32(&s_base), 512); tmem_relinquish(); }
    if (tid==0) asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(smem_u32(&s_mbar)));
    __syncthreads();
    const uint32_t base = s_base;

    // 3) MMA warp (warp 0), elected leader lane issues 8 k-tiles then commits.
    if (warp==0) {
        uint64_t qd0 = make_desc_sm100(smem_u32(sQ), LBO, SBO, 2);
        uint64_t kd0 = make_desc_sm100(smem_u32(sK), LBO, SBO, 2);
        uint32_t mbar = smem_u32(&s_mbar);
        // broadcast descriptors from lane 0 (they must be uniform across the warp)
        // (smem base is already uniform, but mirror the reference's shfl for safety)
        for (int kb=0; kb<NKT; ++kb) {
            uint64_t qd = desc_add(qd0, qk_doff(kb));
            uint64_t kd = desc_add(kd0, qk_doff(kb));
            uint32_t pred = (kb>0) ? 1u : 0u;
            asm volatile(
                "{ .reg .pred lead; .reg .pred p;\n"
                "  elect.sync _|lead, -1;\n"
                "  setp.ne.b32 p, %3, 0;\n"
                "  @lead tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p;\n"
                "}" :: "r"(base), "l"(qd), "l"(kd), "r"(pred), "r"(QK_IDESC));
        }
        // commit: signal MMA completion onto the mbarrier (one arrival).
        asm volatile(
            "{ .reg .pred lead;\n"
            "  elect.sync _|lead, -1;\n"
            "  @lead tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
            "}" :: "r"(mbar));
    }

    // 4) all threads wait for MMA completion (phase 0).
    {
        uint32_t mbar = smem_u32(&s_mbar);
        asm volatile(
            "{ .reg .pred P;\n"
            "LWAIT: mbarrier.try_wait.parity.shared.b64 P, [%0], 0, 10000000;\n"
            "  @P bra DONE; bra LWAIT;\n"
            "DONE: }" :: "r"(mbar));
    }
    __syncthreads();

    // 5) read S back: thread tid owns tmem lane tid (== row); read cols 0..N-1.
    for (int c=0; c<N; ++c) {
        uint32_t taddr = base + ((uint32_t)tid << 16) + (uint32_t)c;
        uint32_t bits = tmem_ld_1(taddr);
        tmem_wait_ld();
        float v; memcpy(&v, &bits, 4);
        Sout[tid*N + c] = v;
    }
    __syncthreads();
    if (warp==0) tmem_dealloc(base, 512);
    (void)lane;
}

int main(){
    ORC_CHECK(cudaSetDevice(0));
    cudaDeviceProp p{}; ORC_CHECK(cudaGetDeviceProperties(&p,0));
    std::printf("device: %s cc %d.%d\n", p.name, p.major, p.minor);

    std::vector<uint16_t> Qh, Kh; orc::fill_bf16(Qh, M*K, 1); orc::fill_bf16(Kh, N*K, 2);
    __nv_bfloat16 *dQ,*dK; float* dS;
    ORC_CHECK(cudaMalloc(&dQ, M*K*2)); ORC_CHECK(cudaMalloc(&dK, N*K*2)); ORC_CHECK(cudaMalloc(&dS, M*N*4));
    ORC_CHECK(cudaMemcpy(dQ, Qh.data(), M*K*2, cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dK, Kh.data(), N*K*2, cudaMemcpyHostToDevice));

    qk_kernel<<<1,128>>>(dQ, dK, dS);
    ORC_CHECK(cudaGetLastError()); ORC_CHECK(cudaDeviceSynchronize());

    std::vector<float> S(M*N); ORC_CHECK(cudaMemcpy(S.data(), dS, M*N*4, cudaMemcpyDeviceToHost));

    // fp32 reference: S[m,n] = sum_k bf16(Q[m,k]) * bf16(K[n,k])
    double max_abs=0, max_rel=0; int nbad=0, shown=0;
    for (int m=0;m<M;++m) for (int n=0;n<N;++n){
        double acc=0;
        for (int k=0;k<K;++k) acc += (double)orc::bf16_bits_to_f32(Qh[m*K+k]) * (double)orc::bf16_bits_to_f32(Kh[n*K+k]);
        double got = S[m*N+n], err = std::fabs(got-acc);
        double rel = err/(std::fabs(acc)+1e-6);
        if (err>max_abs) max_abs=err;
        if (rel>max_rel) max_rel=rel;
        if (err>0.5 && shown<8){ std::printf("  bad m=%d n=%d got=%.4f ref=%.4f\n",m,n,got,acc); shown++; nbad++; }
    }
    std::printf("max_abs=%.5f  max_rel=%.5f\n", max_abs, max_rel);
    bool pass = (max_abs < 0.5);   // bf16 GEMM over K=128, values in [-1,1]
    std::printf("%s: QK GEMM (tcgen05.mma SS bf16 -> tmem -> readback)\n", pass?"PASS":"FAIL");
    ORC_CHECK(cudaFree(dQ)); ORC_CHECK(cudaFree(dK)); ORC_CHECK(cudaFree(dS));
    return pass?0:1;
}
