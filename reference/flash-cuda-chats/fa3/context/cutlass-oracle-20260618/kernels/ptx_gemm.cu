// Pure-CUDA / inline-PTX GEMM oracle (NO cutlass / cute).
//
// Re-implements the two FA3 hd128-bf16 prefill GEMM building blocks using only raw Hopper
// primitives traced from the CUTLASS oracle. Build with -DGEMM_MODE_QK (default) or -DGEMM_MODE_PV
// (mirrors cutlass_gemm.cu):
//   QK: C[M,N] = A[M,K] . B[N,K]^T   (wgmma SS; A=Q, B=K both K-major)          K=head_dim
//   PV: C[M,N] = A[M,K] . B[K,N]     (wgmma RS; A=P in registers, B=V^T in smem)  K=n-block keys
//
// Shared design (matches/beats CUTLASS on these store-bound K=128 fp32-output GEMMs, bit-identical
// output): PERSISTENT grid (occupancy x #SMs) striding the tiles_m x tiles_n output grid; 2
// warpgroups (256 threads) / 128x128 tile; hand-built GMMA descriptor (no-swizzle K-major);
// float2 (64-bit) accumulator->global epilogue straight from registers.
//
// Usage: ./ptx_gemm <M> <N> <K> [out.bin] [iters] [trials]   (K%16==0, M%128==0, N%128==0)

#include "oracle_common.cuh"

// ---- GMMA shared-memory descriptor (see cute/arch/mma_sm90_desc.hpp) --------------------
// Encodes: start_address>>4 [0,14), leading_byte_offset>>4 [16,30), stride_byte_offset>>4
// [32,46), base_offset [49,52), layout_type [62,64). layout_type: 0=INTERLEAVE(no swizzle).
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t smem_addr, uint32_t lbo_bytes,
                                                    uint32_t sbo_bytes, uint32_t layout_type) {
    uint64_t d = 0;
    d |= (uint64_t)((smem_addr >> 4) & 0x3FFFu);
    d |= (uint64_t)((lbo_bytes  >> 4) & 0x3FFFu) << 16;
    d |= (uint64_t)((sbo_bytes  >> 4) & 0x3FFFu) << 32;
    d |= (uint64_t)(layout_type & 0x3u) << 62;
    return d;
}

// Canonical no-swizzle K-major smem layout for a (MN x K) operand. Element (mn,k) is stored at
// this element offset within the operand's smem region. Built from core matrices of 8(MN) x
// 8(K) bf16 (=128B), tiled: [kb (K/16)][a=mn/8][c=(k%16)/8][b=mn%8][d=k%8].
__device__ __forceinline__ int gmma_k_off(int mn, int k, int MN) {
    int a = mn >> 3, b = mn & 7;
    int kb = k >> 4, kk = k & 15;
    int c = kk >> 3, dd = kk & 7;
    return kb * (MN * 16) + a * 128 + c * 64 + b * 8 + dd;
}

// CTA computes a 128(M) x 128(N) tile with 2 warpgroups (256 threads); each WG issues its own
// m64n128k16 wgmma. K (contraction) is loaded whole into smem (K=128 building block).
constexpr int CTA_M = 128;
constexpr int CTA_N = 128;
constexpr int NWG = 2;
constexpr int WG = 128;
constexpr int NThreads = NWG * WG;

extern __shared__ __nv_bfloat16 smem[];

#if defined(GEMM_MODE_PV)
// ====================== PV: O[M,N] = P[M,K] . V[K,N]  (wgmma RS) ======================
static const char* kModeName = "PV";

// Load V[K,N] (row-major) transposed into smem as V^T in the K-major gmma layout:
//   sVt[gmma_k_off(n, k, CTA_N)] = V[k*N + (col0+n)].
// Coalesced float4 read of 8 consecutive n (global), scattered to 8 k-strided smem slots.
__device__ __forceinline__ void load_vt(const __nv_bfloat16* __restrict__ V, __nv_bfloat16* S,
                                        int col0, int N, int K, int tid, int nthreads) {
    const int chunks = (CTA_N / 8) * K;   // groups of 8 n, per k
    for (int idx = tid; idx < chunks; idx += nthreads) {
        int n0 = (idx % (CTA_N / 8)) * 8;
        int k  = idx / (CTA_N / 8);
        float4 v = *reinterpret_cast<const float4*>(&V[(size_t)k * N + col0 + n0]);
        const __nv_bfloat16* vv = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int i = 0; i < 8; ++i) S[gmma_k_off(n0 + i, k, CTA_N)] = vv[i];
    }
}

__global__ __launch_bounds__(NThreads) void ptx_gemm_pv(const __nv_bfloat16* __restrict__ P,
                                                  const __nv_bfloat16* __restrict__ V,
                                                  float* __restrict__ C, int M, int N, int K,
                                                  int tiles_m, int tiles_n) {
    const int tid = threadIdx.x;
    const int wg = tid / WG;
    const int wtid = tid % WG;
    const int warp = wtid >> 5, lane = wtid & 31;
    const int group = lane >> 2, tg = lane & 3;
    const int wg_row = wg * 64;

    __nv_bfloat16* sVt = smem;
    const uint32_t sVt_base = (uint32_t)__cvta_generic_to_shared(sVt);
    const int ksteps = K / 16;
    const int total = tiles_m * tiles_n;

    for (int tile = blockIdx.x; tile < total; tile += gridDim.x) {
        const int row0 = (tile / tiles_n) * CTA_M;   // M offset (queries)
        const int col0 = (tile % tiles_n) * CTA_N;   // N offset (head_dim_v)

        load_vt(V, sVt, col0, N, K, tid, NThreads);
        __syncthreads();
        // V^T written via generic STS; make it visible to wgmma's async proxy.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        float d[64];
#pragma unroll
        for (int i = 0; i < 64; ++i) d[i] = 0.f;

        // P row owned by this thread's A-fragment (K-major, in registers).
        const int prow = row0 + wg_row + warp * 16 + group;

        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        for (int kb = 0; kb < ksteps; ++kb) {
            int k0 = 16 * kb + 2 * tg;
            // A operand: 4x b32, each = 2 contiguous bf16 of P (K-major fragment).
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(&P[(size_t)prow       * K + k0]);
            uint32_t a1 = *reinterpret_cast<const uint32_t*>(&P[(size_t)(prow + 8) * K + k0]);
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(&P[(size_t)prow       * K + k0 + 8]);
            uint32_t a3 = *reinterpret_cast<const uint32_t*>(&P[(size_t)(prow + 8) * K + k0 + 8]);
            uint64_t descB = make_smem_desc(sVt_base + (uint32_t)(kb * CTA_N * 16 * 2), 128, 256, 0);
            int scaleD = (kb == 0) ? 0 : 1;
            asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %69, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
            "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
            "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
            "{%64,%65,%66,%67},"
            "%68,p,%70,%71,%72;\n"
            "}\n"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
              "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
              "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
              "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
              "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
              "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
              "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
              "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "l"(descB), "r"(scaleD), "n"(1), "n"(1), "n"(0));
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

        // ---- accumulator -> C (row-major M x N), same fragment layout as QK. ----
#pragma unroll
        for (int c = 0; c < CTA_N / 8; ++c) {
            int col0n = 8 * c + 2 * tg;
            int rlo = wg_row + warp * 16 + group;
            int rhi = rlo + 8;
            int gr_lo = row0 + rlo, gr_hi = row0 + rhi;
            int gc0 = col0 + col0n;
            if (gr_lo < M && gc0 + 1 < N) *reinterpret_cast<float2*>(&C[gr_lo * N + gc0]) = make_float2(d[c*4+0], d[c*4+1]);
            if (gr_hi < M && gc0 + 1 < N) *reinterpret_cast<float2*>(&C[gr_hi * N + gc0]) = make_float2(d[c*4+2], d[c*4+3]);
        }
        __syncthreads();
    }
}

// fp32 reference: O[m,n] = sum_k P[m,k] * V[k,n]  (P row-major MxK, V row-major KxN)
__global__ void ref_gemm_pv(const __nv_bfloat16* P, const __nv_bfloat16* V, float* C,
                            int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += __bfloat162float(P[m * K + k]) * __bfloat162float(V[(size_t)k * N + n]);
    C[m * N + n] = acc;
}

#else
// ====================== QK: C[M,N] = A[M,K] . B[N,K]^T  (wgmma SS) ======================
static const char* kModeName = "QK";

// vectorized global->swizzled-smem copy: 8 bf16 (16B) per chunk. float4 LDG->STS (beats single-
// buffer cp.async here). Within a core matrix the 8 consecutive K-elems are contiguous in both
// global (row-major) and the gmma-K smem layout.
__device__ __forceinline__ void load_kmajor(const __nv_bfloat16* __restrict__ G, __nv_bfloat16* S,
                                            int row0, int MN, int K, int tid, int nthreads) {
    const int chunks = MN * (K / 8);
    for (int idx = tid; idx < chunks; idx += nthreads) {
        int mn = idx / (K / 8);
        int kc = (idx % (K / 8)) * 8;
        const float4* gp = reinterpret_cast<const float4*>(&G[(row0 + mn) * K + kc]);
        float4* sp = reinterpret_cast<float4*>(&S[gmma_k_off(mn, kc, MN)]);
        *sp = *gp;
    }
}

__global__ __launch_bounds__(NThreads) void ptx_gemm_qk(const __nv_bfloat16* __restrict__ A,
                                                  const __nv_bfloat16* __restrict__ B,
                                                  float* __restrict__ C, int M, int N, int K,
                                                  int tiles_m, int tiles_n) {
    const int tid = threadIdx.x;
    const int wg = tid / WG;               // which warpgroup (0..NWG-1)
    const int wtid = tid % WG;             // thread within warpgroup
    const int warp = wtid >> 5, lane = wtid & 31;
    const int gid = lane >> 2, tg = lane & 3;
    const int wg_row = wg * 64;

    __nv_bfloat16* sA = smem;
    __nv_bfloat16* sB = smem + CTA_M * K;
    const uint32_t sA_base = (uint32_t)__cvta_generic_to_shared(sA);
    const uint32_t sB_base = (uint32_t)__cvta_generic_to_shared(sB);
    // this WG's A sub-tile (rows [64*wg, 64*wg+64)) starts g*1024 elems into sA (per gmma_k_off).
    const uint32_t sA_wg = sA_base + (uint32_t)(wg * 1024 * 2);
    const int ksteps = K / 16;
    const int total = tiles_m * tiles_n;

    for (int tile = blockIdx.x; tile < total; tile += gridDim.x) {
        const int row0 = (tile / tiles_n) * CTA_M;   // M offset
        const int col0 = (tile % tiles_n) * CTA_N;   // N offset

        load_kmajor(A, sA, row0, CTA_M, K, tid, NThreads);
        load_kmajor(B, sB, col0, CTA_N, K, tid, NThreads);
        __syncthreads();
        // wgmma reads smem through the async proxy; make the generic STS writes visible to it.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        float d[64];
#pragma unroll
        for (int i = 0; i < 64; ++i) d[i] = 0.f;

        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        for (int kb = 0; kb < ksteps; ++kb) {
            uint64_t descA = make_smem_desc(sA_wg   + (uint32_t)(kb * CTA_M * 16 * 2), 128, 256, 0);
            uint64_t descB = make_smem_desc(sB_base + (uint32_t)(kb * CTA_N * 16 * 2), 128, 256, 0);
            // scaleD=1 (accumulate after first step), scaleA=1, scaleB=1, tnspA=0, tnspB=0
            int scaleD = (kb == 0) ? 0 : 1;
            asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %66, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
            "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
            "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
            "%64,%65,p,%67,%68,%69,%70;\n"
            "}\n"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
              "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
              "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
              "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
              "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
              "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
              "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
              "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
            : "l"(descA), "l"(descB), "r"(scaleD), "n"(1), "n"(1), "n"(0), "n"(0));
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

        // ---- accumulator -> C (row-major M x N). wgmma .f32 m64nNk16 fragment layout. ----
        // this WG owns rows [64*wg, 64*wg+64); each thread holds adjacent column pairs.
#pragma unroll
        for (int c = 0; c < CTA_N / 8; ++c) {
            int col0n = 8 * c + 2 * tg;
            int rlo = wg_row + warp * 16 + gid;
            int rhi = rlo + 8;
            int gr_lo = row0 + rlo, gr_hi = row0 + rhi;
            int gc0 = col0 + col0n;
            // adjacent columns -> one 64-bit store per row half
            if (gr_lo < M && gc0 + 1 < N) *reinterpret_cast<float2*>(&C[gr_lo * N + gc0]) = make_float2(d[c*4+0], d[c*4+1]);
            if (gr_hi < M && gc0 + 1 < N) *reinterpret_cast<float2*>(&C[gr_hi * N + gc0]) = make_float2(d[c*4+2], d[c*4+3]);
        }
        // ensure all warps finished reading smem (wgmma) before next tile overwrites it.
        __syncthreads();
    }
}

// fp32 reference: C = A . B^T (both row-major MxK / NxK)
__global__ void ref_gemm_qk(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                            int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += __bfloat162float(A[m * K + k]) * __bfloat162float(B[n * K + k]);
    C[m * N + n] = acc;
}
#endif

int main(int argc, char** argv) {
    int M = argc > 1 ? std::atoi(argv[1]) : 2048;
    int N = argc > 2 ? std::atoi(argv[2]) : 2048;
    int K = argc > 3 ? std::atoi(argv[3]) : 128;
    std::string out_path = argc > 4 ? argv[4] : "";
    int iters  = argc > 5 ? std::atoi(argv[5]) : 100;
    int trials = argc > 6 ? std::atoi(argv[6]) : 7;
    if (K % 16 || M % CTA_M || N % CTA_N) {
        std::fprintf(stderr, "require K%%16==0, M%%128==0, N%%128==0\n"); return 1;
    }

    std::vector<uint16_t> hA, hB;
    orc::fill_bf16(hA, size_t(M) * K, 1);
    orc::fill_bf16(hB, size_t(N) * K, 2);

    __nv_bfloat16 *dA, *dB; float *dC, *dRef;
    ORC_CHECK(cudaMalloc(&dA, sizeof(__nv_bfloat16) * size_t(M) * K));
    ORC_CHECK(cudaMalloc(&dB, sizeof(__nv_bfloat16) * size_t(N) * K));
    ORC_CHECK(cudaMalloc(&dC, sizeof(float) * size_t(M) * N));
    ORC_CHECK(cudaMalloc(&dRef, sizeof(float) * size_t(M) * N));
    ORC_CHECK(cudaMemcpy(dA, hA.data(), sizeof(__nv_bfloat16) * size_t(M) * K, cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dB, hB.data(), sizeof(__nv_bfloat16) * size_t(N) * K, cudaMemcpyHostToDevice));

    const int tiles_m = M / CTA_M, tiles_n = N / CTA_N;
    const int total_tiles = tiles_m * tiles_n;
#if defined(GEMM_MODE_PV)
    auto kernel = ptx_gemm_pv;
    size_t smem_bytes = sizeof(__nv_bfloat16) * CTA_N * K;          // V^T tile only
#else
    auto kernel = ptx_gemm_qk;
    size_t smem_bytes = sizeof(__nv_bfloat16) * (CTA_M + CTA_N) * K; // A + B tiles
#endif
    if (smem_bytes >= 48 * 1024)
        ORC_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int)smem_bytes));
    // persistent grid: occupancy * #SMs, capped at the number of tiles.
    cudaDeviceProp prop; ORC_CHECK(cudaGetDeviceProperties(&prop, 0));
    int blocks_per_sm = 0;
    ORC_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, NThreads,
                                                            (int)smem_bytes));
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    int grid_blocks = prop.multiProcessorCount * blocks_per_sm;
    if (grid_blocks > total_tiles) grid_blocks = total_tiles;
    dim3 grid(grid_blocks);
    auto launch = [&]() { kernel<<<grid, NThreads, smem_bytes>>>(dA, dB, dC, M, N, K, tiles_m, tiles_n); };
    launch();
    ORC_CHECK(cudaDeviceSynchronize());
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(e)); return 4; }

    // correctness vs device fp32 reference
    dim3 rb(16, 16), rg((N + 15) / 16, (M + 15) / 16);
#if defined(GEMM_MODE_PV)
    ref_gemm_pv<<<rg, rb>>>(dA, dB, dRef, M, N, K);
#else
    ref_gemm_qk<<<rg, rb>>>(dA, dB, dRef, M, N, K);
#endif
    ORC_CHECK(cudaDeviceSynchronize());
    std::vector<float> hC(size_t(M) * N), hRef(size_t(M) * N);
    ORC_CHECK(cudaMemcpy(hC.data(), dC, sizeof(float) * size_t(M) * N, cudaMemcpyDeviceToHost));
    ORC_CHECK(cudaMemcpy(hRef.data(), dRef, sizeof(float) * size_t(M) * N, cudaMemcpyDeviceToHost));
    double max_abs = 0, max_rel = 0; long nbad = 0;
    for (size_t i = 0; i < hC.size(); ++i) {
        double a = hC[i], r = hRef[i], e2 = std::fabs(a - r);
        max_abs = std::max(max_abs, e2);
        double rel = e2 / (std::fabs(r) + 1e-6);
        max_rel = std::max(max_rel, rel);
        if (e2 > 0.5 && rel > 0.05) ++nbad;
    }

    float median = 0.f;
    float best = orc::benchmark_ms(launch, 50, iters, trials, &median);
    double gflop = orc::gemm_gflop(M, N, K);
    std::printf("{\"impl\": \"ptx\", \"mode\": \"%s\", \"M\": %d, \"N\": %d, \"K\": %d, "
                "\"min_ms\": %.6f, \"median_ms\": %.6f, \"tflops\": %.2f, "
                "\"max_abs_err\": %.4g, \"max_rel_err\": %.4g, \"nbad\": %ld}\n",
                kModeName, M, N, K, best, median, gflop / (median * 1e-3) / 1e3, max_abs, max_rel, nbad);

    if (!out_path.empty()) orc::dump_f32(out_path, hC);
    return nbad == 0 ? 0 : 5;
}
