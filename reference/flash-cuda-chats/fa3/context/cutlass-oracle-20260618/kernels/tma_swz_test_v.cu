// Isolated validation of the MN-major 128B-swizzle <-> wgmma-descriptor <-> TMA integration for the V
// operand (PV GEMM). O[128,128] = P[128,128] @ V[128,128], bf16 operands, fp32 acc.
//   P : A operand, K-major (keys contiguous)   -> K-major B128 desc (LBO=16B, SBO=1024B), trans-a=0
//   V : B operand, MN-major (dv  contiguous)   -> MN-major B128 desc (LBO=1024B, SBO=2048B), trans-b=1
// Both ground-truthed from cute (swz_probe.cu): MN k-step (over keys) = kb*2048 elems = kb*4096 B.
// Path B (the one the real kernel uses): TMA-load P,V with CU_TENSOR_MAP_SWIZZLE_128B (single box,
// inner = contiguous extent = 128 elems = 256B, exactly as cute/FA3 encode it) -> wgmma -> vs CPU ref.
#include "oracle_common.cuh"
#include <cuda.h>

constexpr int M = 128, KEYS = 128, DV = 128;

__device__ __host__ __forceinline__ int swz(int o) { return o ^ (((o >> 7) & 7) << 4); }
// element (mn,k) of a [128,K] K-major SW128 tile (used for the P / A operand)
__device__ __host__ __forceinline__ int koff(int mn, int k) { return (k >> 6) * 8192 + swz(mn * 64 + (k & 63)); }

__device__ __forceinline__ uint64_t make_desc(uint32_t a, uint32_t lbo, uint32_t sbo, uint32_t lt) {
    uint64_t d = 0;
    d |= (uint64_t)((a >> 4) & 0x3FFFu);
    d |= (uint64_t)((lbo >> 4) & 0x3FFFu) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFFu) << 32;
    d |= (uint64_t)(lt & 0x3u) << 62;
    return d;
}

// TMA helpers (identical to tma_swz_test.cu)
__device__ __forceinline__ void mbar_init(uint64_t* bar, int c) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(a), "r"(c));
}
__device__ __forceinline__ void mbar_expect(uint64_t* bar, uint32_t bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma_2d(void* smem, const CUtensorMap* map, int x, int y, uint64_t* bar) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem), b = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];\n" :: "r"(s), "l"(map), "r"(x), "r"(y), "r"(b) : "memory");
}
__device__ __forceinline__ bool mbar_wait(uint64_t* bar, int ph) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar), ok;
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(a), "r"(ph));
    return ok != 0;
}

// SS PV GEMM: A=P (K-major B128), B=V (MN-major B128, trans-b=1). 8 k16 steps over keys=128.
__device__ __forceinline__ void pv_gemm_sw(float (&o)[64], uint32_t sP_wg, uint32_t sV_base) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
        uint32_t kstepP = (uint32_t)(((kb & 3) * 16 + (kb >> 2) * 8192) * 2);  // bytes, K-major (P)
        uint32_t kstepV = (uint32_t)(kb * 2048);                               // bytes, MN-major (V): keys-step=1024 elems
        uint64_t dA = make_desc(sP_wg   + kstepP, 16, 1024, 1);               // K-major B128 (LBO=16,SBO=1024)
        uint64_t dB = make_desc(sV_base + kstepV, 16384, 1024, 1);            // MN-major B128 (LBO=16384,SBO=1024)
        int scaleD = (kb == 0) ? 0 : 1;
        asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "%64,%65,p,%67,%68,%69,%70;\n}\n"
        : "+f"(o[0]),"+f"(o[1]),"+f"(o[2]),"+f"(o[3]),"+f"(o[4]),"+f"(o[5]),"+f"(o[6]),"+f"(o[7]),
          "+f"(o[8]),"+f"(o[9]),"+f"(o[10]),"+f"(o[11]),"+f"(o[12]),"+f"(o[13]),"+f"(o[14]),"+f"(o[15]),
          "+f"(o[16]),"+f"(o[17]),"+f"(o[18]),"+f"(o[19]),"+f"(o[20]),"+f"(o[21]),"+f"(o[22]),"+f"(o[23]),
          "+f"(o[24]),"+f"(o[25]),"+f"(o[26]),"+f"(o[27]),"+f"(o[28]),"+f"(o[29]),"+f"(o[30]),"+f"(o[31]),
          "+f"(o[32]),"+f"(o[33]),"+f"(o[34]),"+f"(o[35]),"+f"(o[36]),"+f"(o[37]),"+f"(o[38]),"+f"(o[39]),
          "+f"(o[40]),"+f"(o[41]),"+f"(o[42]),"+f"(o[43]),"+f"(o[44]),"+f"(o[45]),"+f"(o[46]),"+f"(o[47]),
          "+f"(o[48]),"+f"(o[49]),"+f"(o[50]),"+f"(o[51]),"+f"(o[52]),"+f"(o[53]),"+f"(o[54]),"+f"(o[55]),
          "+f"(o[56]),"+f"(o[57]),"+f"(o[58]),"+f"(o[59]),"+f"(o[60]),"+f"(o[61]),"+f"(o[62]),"+f"(o[63])
        : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(1));  // trans-a=0, trans-b=1
    }
    asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// write m64n128 fragment O[128,128] (row-major) to global
__device__ __forceinline__ void store_O(float (&o)[64], float* Og, int cwg, int warp, int lane) {
    int group = lane >> 2, tg = lane & 3;
    int row_lo = cwg * 64 + warp * 16 + group, row_hi = row_lo + 8;
#pragma unroll
    for (int c = 0; c < 16; ++c)
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            int col = c * 8 + tg * 2 + p;
            Og[row_lo * DV + col] = o[c * 4 + 0 * 2 + p];
            Og[row_hi * DV + col] = o[c * 4 + 1 * 2 + p];
        }
}

// TMA-load P (K-major box, 2x 64-wide) and V (MN-major single 128-wide box) then wgmma.
__global__ void pv_swz_tma(const __grid_constant__ CUtensorMap mapP, const __grid_constant__ CUtensorMap mapV,
                           float* __restrict__ Og) {
    __shared__ alignas(1024) __nv_bfloat16 sP[16384];
    __shared__ alignas(1024) __nv_bfloat16 sV[16384];
    __shared__ alignas(8) uint64_t bar;
    int tid = threadIdx.x;
    if (tid == 0) {
        mbar_init(&bar, 1);
        // P: two 128x64 K-major boxes (k0=0,64). V: two 64x128 MN-major boxes (dv0=0,64) -> smem 0/8192,
        // exactly mirroring K (FA3's SmemLayoutVt has the 2 dv-halves at clean stride 8192).
        mbar_expect(&bar, (2u * 128u * 64u + 2u * 64u * 128u) * sizeof(__nv_bfloat16));
        tma_2d(&sP[0],    &mapP, 0,  0, &bar);
        tma_2d(&sP[8192], &mapP, 64, 0, &bar);
        tma_2d(&sV[0],    &mapV, 0,  0, &bar);   // dv 0..63
        tma_2d(&sV[8192], &mapV, 64, 0, &bar);   // dv 64..127
    }
    __syncthreads();
    while (!mbar_wait(&bar, 0)) {}
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    int wtid = tid % 128, cwg = tid / 128, warp = wtid >> 5, lane = wtid & 31;
    uint32_t sP_wg = (uint32_t)__cvta_generic_to_shared(sP) + (uint32_t)(cwg * 4096 * 2);
    uint32_t sV_a  = (uint32_t)__cvta_generic_to_shared(sV);
    float o[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) o[i] = 0.f;
    pv_gemm_sw(o, sP_wg, sV_a);
    store_O(o, Og, cwg, warp, lane);
}

// K-major map for P: gdim {inner=keys, outer=M}, box {64, 128}, SWIZZLE_128B (same as the K test).
static CUtensorMap make_map_kmajor(const __nv_bfloat16* ptr) {
    CUtensorMap map{};
    uint64_t gdim[2] = {(uint64_t)KEYS, (uint64_t)M};
    uint64_t gstride[1] = {(uint64_t)KEYS * sizeof(__nv_bfloat16)};
    uint32_t bdim[2] = {64u, 128u};
    uint32_t estride[2] = {1, 1};
    CUresult cr = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (cr != CUDA_SUCCESS) { const char* s; cuGetErrorString(cr, &s);
        std::fprintf(stderr, "P map: %s\n", s); std::exit(2); }
    return map;
}
// MN-major map for V: V is [keys, dv] row-major (dv contiguous). gdim {inner=dv, outer=keys},
// box {dv=128, keys=128} single box -> exactly cute/FA3's V box. SWIZZLE_128B.
static CUtensorMap make_map_mnmajor(const __nv_bfloat16* ptr) {
    CUtensorMap map{};
    uint64_t gdim[2] = {(uint64_t)DV, (uint64_t)KEYS};
    uint64_t gstride[1] = {(uint64_t)DV * sizeof(__nv_bfloat16)};
    uint32_t bdim[2] = {64u, 128u};                              // inner = dv = 128B (2 boxes, dv0=0/64)
    uint32_t estride[2] = {1, 1};
    CUresult cr = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (cr != CUDA_SUCCESS) { const char* s; cuGetErrorString(cr, &s);
        std::fprintf(stderr, "V map: %s\n", s); std::exit(2); }
    return map;
}

int main() {
    std::vector<uint16_t> hP, hV;
    orc::fill_bf16(hP, M * KEYS, 1);
    orc::fill_bf16(hV, KEYS * DV, 2);
    // CPU ref O = P @ V (fp32 acc over bf16 operands)
    std::vector<float> ref(M * DV);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < DV; ++j) {
            float acc = 0.f;
            for (int k = 0; k < KEYS; ++k)
                acc += orc::bf16_bits_to_f32(hP[i * KEYS + k]) * orc::bf16_bits_to_f32(hV[k * DV + j]);
            ref[i * DV + j] = acc;
        }
    __nv_bfloat16 *dP, *dV; float *dO;
    ORC_CHECK(cudaMalloc(&dP, sizeof(__nv_bfloat16) * M * KEYS));
    ORC_CHECK(cudaMalloc(&dV, sizeof(__nv_bfloat16) * KEYS * DV));
    ORC_CHECK(cudaMalloc(&dO, sizeof(float) * M * DV));
    ORC_CHECK(cudaMemcpy(dP, hP.data(), sizeof(__nv_bfloat16) * M * KEYS, cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dV, hV.data(), sizeof(__nv_bfloat16) * KEYS * DV, cudaMemcpyHostToDevice));

    CUtensorMap mapP = make_map_kmajor(dP), mapV = make_map_mnmajor(dV);
    ORC_CHECK(cudaMemset(dO, 0, sizeof(float) * M * DV));
    pv_swz_tma<<<1, 256>>>(mapP, mapV, dO);
    ORC_CHECK(cudaDeviceSynchronize());
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "pv_swz_tma: %s\n", cudaGetErrorString(e)); return 4; }

    std::vector<float> hO(M * DV);
    ORC_CHECK(cudaMemcpy(hO.data(), dO, sizeof(float) * M * DV, cudaMemcpyDeviceToHost));
    double md = 0; int nbad = 0;
    for (int i = 0; i < M * DV; ++i) { double d = std::fabs(hO[i] - ref[i]); md = std::max(md, d); if (d > 0.5) nbad++; }
    std::printf("pathB(V MN-major TMA) max_abs_diff=%.4f  nbad(>0.5)=%d\n", md, nbad);
    std::printf("RESULT: %s\n", nbad == 0 ? "PASS (V TMA path bit-exact)" : "FAIL");
    return nbad == 0 ? 0 : 5;
}
