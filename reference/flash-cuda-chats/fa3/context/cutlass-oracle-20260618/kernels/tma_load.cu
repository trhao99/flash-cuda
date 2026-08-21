// Memory-movement component: TMA (Tensor Memory Accelerator) bulk tile load, pure CUDA + inline
// PTX (NO cutlass/cute). Replicates what FA3 uses `cute::SM90_TMA_LOAD` / `make_tma_copy` for:
//   * host: cuTensorMapEncodeTiled  (CUDA driver) builds the tensor-map descriptor
//   * device: cp.async.bulk.tensor.2d.shared::cluster.global ... mbarrier::complete_tx::bytes
//             + mbarrier.init / arrive.expect_tx / try_wait.parity
//
// Validates by TMA-loading every [boxM x boxK] tile of a [M,K] bf16 matrix into smem and writing it
// back to global; checks bitwise equality vs the input and reports effective HBM bandwidth. A plain
// cp.async copy is timed as a baseline. SWIZZLE_NONE here (the swizzle integration with wgmma is a
// separate step); FA3's paged config actually loads K/V via cp.async, Q via TMA.
//
// Usage: ./tma_load <M> <K> [iters] [trials]

#include "oracle_common.cuh"
#include <cuda.h>   // CUtensorMap, cuTensorMapEncodeTiled (driver API; link -lcuda)

constexpr int BOX_M = 128;
constexpr int BOX_K = 64;
constexpr int NThreads = 256;

// ---- inline-PTX TMA / mbarrier helpers ----
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* smem, const CUtensorMap* map,
                                            int x, int y, uint64_t* bar) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t b = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(s), "l"(map), "r"(x), "r"(y), "r"(b) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, int phase) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t ok;
    asm volatile(
        "{\n.reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(ok) : "r"(a), "r"(phase));
    return ok != 0;
}

// TMA-load each box tile into smem, write back to global (round-trip correctness + bandwidth).
__global__ void tma_roundtrip(const __grid_constant__ CUtensorMap map,
                              __nv_bfloat16* __restrict__ out, int M, int K,
                              int tiles_m, int tiles_k) {
    __shared__ alignas(128) __nv_bfloat16 tile_smem[BOX_M * BOX_K];  // TMA dst needs 128B align
    __shared__ alignas(8) uint64_t bar;
    const int tid = threadIdx.x;
    const int total = tiles_m * tiles_k;

    if (tid == 0) mbar_init(&bar, 1);
    __syncthreads();

    for (int tile = blockIdx.x; tile < total; tile += gridDim.x) {
        const int m0 = (tile / tiles_k) * BOX_M;
        const int k0 = (tile % tiles_k) * BOX_K;
        if (tid == 0) {
            tma_load_2d(tile_smem, &map, k0, m0, &bar);      // coords: {inner=K, outer=M}
            mbar_expect_tx(&bar, BOX_M * BOX_K * sizeof(__nv_bfloat16));
        }
        // phase flips every completed tile; track parity = #tiles done so far for this block.
        int phase = ((tile - blockIdx.x) / gridDim.x) & 1;
        while (!mbar_try_wait(&bar, phase)) {}
        // smem holds the box row-major (inner K contiguous): smem[m*BOX_K + k].
        for (int i = tid; i < BOX_M * BOX_K; i += NThreads) {
            int m = i / BOX_K, k = i % BOX_K;
            out[(size_t)(m0 + m) * K + (k0 + k)] = tile_smem[m * BOX_K + k];
        }
        __syncthreads();   // tile reused next iteration
    }
}

// cp.async baseline: same round-trip via float4 global->smem->global.
__global__ void cpasync_roundtrip(const __nv_bfloat16* __restrict__ in,
                                  __nv_bfloat16* __restrict__ out, int M, int K,
                                  int tiles_m, int tiles_k) {
    __shared__ alignas(128) __nv_bfloat16 tile_smem[BOX_M * BOX_K];
    const int tid = threadIdx.x;
    const int total = tiles_m * tiles_k;
    for (int tile = blockIdx.x; tile < total; tile += gridDim.x) {
        const int m0 = (tile / tiles_k) * BOX_M;
        const int k0 = (tile % tiles_k) * BOX_K;
        for (int i = tid; i < BOX_M * (BOX_K / 8); i += NThreads) {
            int m = i / (BOX_K / 8), kc = (i % (BOX_K / 8)) * 8;
            uint32_t s = (uint32_t)__cvta_generic_to_shared(&tile_smem[m * BOX_K + kc]);
            const void* g = &in[(size_t)(m0 + m) * K + (k0 + kc)];
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g));
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        for (int i = tid; i < BOX_M * (BOX_K / 8); i += NThreads) {
            int m = i / (BOX_K / 8), kc = (i % (BOX_K / 8)) * 8;
            *reinterpret_cast<float4*>(&out[(size_t)(m0 + m) * K + (k0 + kc)]) =
                *reinterpret_cast<float4*>(&tile_smem[m * BOX_K + kc]);
        }
        __syncthreads();
    }
}

// ---- load-only bandwidth (isolates the read path; checksum prevents dead-code elim) ----
__global__ void tma_loadonly(const __grid_constant__ CUtensorMap map, float* __restrict__ sink,
                             int tiles_m, int tiles_k) {
    __shared__ alignas(128) __nv_bfloat16 tile_smem[BOX_M * BOX_K];
    __shared__ alignas(8) uint64_t bar;
    const int tid = threadIdx.x, total = tiles_m * tiles_k;
    if (tid == 0) mbar_init(&bar, 1);
    __syncthreads();
    float acc = 0.f;
    for (int tile = blockIdx.x; tile < total; tile += gridDim.x) {
        const int m0 = (tile / tiles_k) * BOX_M, k0 = (tile % tiles_k) * BOX_K;
        if (tid == 0) { tma_load_2d(tile_smem, &map, k0, m0, &bar);
                        mbar_expect_tx(&bar, BOX_M * BOX_K * sizeof(__nv_bfloat16)); }
        int phase = ((tile - blockIdx.x) / gridDim.x) & 1;
        while (!mbar_try_wait(&bar, phase)) {}
        for (int i = tid; i < BOX_M * BOX_K; i += NThreads) acc += __bfloat162float(tile_smem[i]);
        __syncthreads();
    }
    if (acc == -123456.f) sink[tid] = acc;   // never true; keeps load live
}
__global__ void cpasync_loadonly(const __nv_bfloat16* __restrict__ in, float* __restrict__ sink,
                                 int M, int K, int tiles_m, int tiles_k) {
    __shared__ alignas(128) __nv_bfloat16 tile_smem[BOX_M * BOX_K];
    const int tid = threadIdx.x, total = tiles_m * tiles_k;
    float acc = 0.f;
    for (int tile = blockIdx.x; tile < total; tile += gridDim.x) {
        const int m0 = (tile / tiles_k) * BOX_M, k0 = (tile % tiles_k) * BOX_K;
        for (int i = tid; i < BOX_M * (BOX_K / 8); i += NThreads) {
            int m = i / (BOX_K / 8), kc = (i % (BOX_K / 8)) * 8;
            uint32_t s = (uint32_t)__cvta_generic_to_shared(&tile_smem[m * BOX_K + kc]);
            const void* g = &in[(size_t)(m0 + m) * K + (k0 + kc)];
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g));
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n");
        __syncthreads();
        for (int i = tid; i < BOX_M * BOX_K; i += NThreads) acc += __bfloat162float(tile_smem[i]);
        __syncthreads();
    }
    if (acc == -123456.f) sink[tid] = acc;
}

int main(int argc, char** argv) {
    int M = argc > 1 ? std::atoi(argv[1]) : 8192;
    int K = argc > 2 ? std::atoi(argv[2]) : 128;
    int iters  = argc > 3 ? std::atoi(argv[3]) : 200;
    int trials = argc > 4 ? std::atoi(argv[4]) : 9;
    if (M % BOX_M || K % BOX_K) { std::fprintf(stderr, "M%%128, K%%64 required\n"); return 1; }

    std::vector<uint16_t> hIn;
    orc::fill_bf16(hIn, size_t(M) * K, 7);
    __nv_bfloat16 *dIn, *dOut;
    ORC_CHECK(cudaMalloc(&dIn, sizeof(__nv_bfloat16) * size_t(M) * K));
    ORC_CHECK(cudaMalloc(&dOut, sizeof(__nv_bfloat16) * size_t(M) * K));
    ORC_CHECK(cudaMemcpy(dIn, hIn.data(), sizeof(__nv_bfloat16) * size_t(M) * K, cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemset(dOut, 0, sizeof(__nv_bfloat16) * size_t(M) * K));

    // build the TMA descriptor (row-major [M,K], K contiguous): dims inner-first {K, M}.
    CUtensorMap map{};
    uint64_t gdim[2]  = {(uint64_t)K, (uint64_t)M};
    uint64_t gstride[1] = {(uint64_t)K * sizeof(__nv_bfloat16)};   // bytes between rows (dim1)
    uint32_t bdim[2]  = {(uint32_t)BOX_K, (uint32_t)BOX_M};
    uint32_t estride[2] = {1, 1};
    CUresult cr = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dIn, gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (cr != CUDA_SUCCESS) { const char* s; cuGetErrorString(cr, &s);
        std::fprintf(stderr, "cuTensorMapEncodeTiled: %s\n", s); return 2; }

    const int tiles_m = M / BOX_M, tiles_k = K / BOX_K, total = tiles_m * tiles_k;
    cudaDeviceProp prop; ORC_CHECK(cudaGetDeviceProperties(&prop, 0));
    int grid = prop.multiProcessorCount * 2; if (grid > total) grid = total;

    auto tma = [&]() { tma_roundtrip<<<grid, NThreads>>>(map, dOut, M, K, tiles_m, tiles_k); };
    auto cpa = [&]() { cpasync_roundtrip<<<grid, NThreads>>>(dIn, dOut, M, K, tiles_m, tiles_k); };

    // correctness (TMA)
    ORC_CHECK(cudaMemset(dOut, 0, sizeof(__nv_bfloat16) * size_t(M) * K));
    tma(); ORC_CHECK(cudaDeviceSynchronize());
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "tma launch: %s\n", cudaGetErrorString(e)); return 3; }
    std::vector<uint16_t> hOut(size_t(M) * K);
    ORC_CHECK(cudaMemcpy(hOut.data(), dOut, sizeof(uint16_t) * size_t(M) * K, cudaMemcpyDeviceToHost));
    long nbad = 0; for (size_t i = 0; i < hOut.size(); ++i) if (hOut[i] != hIn[i]) ++nbad;

    float* dSink; ORC_CHECK(cudaMalloc(&dSink, sizeof(float) * NThreads));
    auto tma_lo = [&]() { tma_loadonly<<<grid, NThreads>>>(map, dSink, tiles_m, tiles_k); };
    auto cpa_lo = [&]() { cpasync_loadonly<<<grid, NThreads>>>(dIn, dSink, M, K, tiles_m, tiles_k); };

    float tma_med = 0, cpa_med = 0, tma_lo_med = 0, cpa_lo_med = 0;
    orc::benchmark_ms(tma, 30, iters, trials, &tma_med);
    orc::benchmark_ms(cpa, 30, iters, trials, &cpa_med);
    orc::benchmark_ms(tma_lo, 30, iters, trials, &tma_lo_med);
    orc::benchmark_ms(cpa_lo, 30, iters, trials, &cpa_lo_med);
    double rw = 2.0 * double(M) * K * sizeof(__nv_bfloat16);    // round-trip read + write
    double rd = 1.0 * double(M) * K * sizeof(__nv_bfloat16);    // load-only read
    std::printf("{\"comp\": \"tma_load\", \"M\": %d, \"K\": %d, \"nbad\": %ld, "
                "\"rt_tma_GBps\": %.0f, \"rt_cpasync_GBps\": %.0f, "
                "\"load_tma_GBps\": %.0f, \"load_cpasync_GBps\": %.0f}\n",
                M, K, nbad, rw / (tma_med * 1e-3) / 1e9, rw / (cpa_med * 1e-3) / 1e9,
                rd / (tma_lo_med * 1e-3) / 1e9, rd / (cpa_lo_med * 1e-3) / 1e9);
    return nbad == 0 ? 0 : 5;
}
