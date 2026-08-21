// Validate the NATIVE fp8 (e4m3) hd256 / CN=64 wgmma path (QK + PV-SS) before touching fa3_hopper.cu.
//   QK: Q[CM=128,HD=256] . K[CN=64,HD=256]^T -> S[128,64]. m64n64k32, 8 k-blocks, A(Q,CM rows) and
//       B(K,CN rows) have DIFFERENT col-block strides (128*CM vs 128*CN) once HD spans 2 128-col blocks.
//   PV: P[128,64] . V[64,256] -> O[128,256]. V transposed to Vt[256,64] (K-major). Two m64n128k32 boxes
//       over the HDV=256 output (box b -> dv[128b,128b+128), acc_o[64b..64b+63]), 2 k-blocks over key.
//   nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a kernels/probes/fp8_probe256.cu -o /tmp/fp8_probe256 && /tmp/fp8_probe256
#include "probe_common.cuh"
#include <cuda_fp8.h>
#include <cstdio>
#include <cmath>
#include <vector>
using fp8 = __nv_fp8_e4m3;
constexpr int HD = 256, HDV = 256, CTA_M = 128, CN = 64, WG = 128;

// fp8 1-byte 128B swizzle, row-count-generic: col-block (c>=128) stride = 128*ROWS (rows in the tile).
template <int ROWS> __device__ __forceinline__ int koff8R(int row, int c) {
    return (c >> 7) * (128 * ROWS) + row * 128 + ((c & 127) ^ ((row & 7) << 4));
}
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t a, uint32_t lbo, uint32_t sbo, uint32_t lt) {
    uint64_t d = 0; d |= (uint64_t)((a >> 4) & 0x3FFFu); d |= (uint64_t)((lbo >> 4) & 0x3FFFu) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFFu) << 32; d |= (uint64_t)(lt & 0x3u) << 62; return d;
}
constexpr uint32_t SW_LBO = 16, SW_SBO = 1024;

#define ACC32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
#define OUT32(a) \
  "+f"(a[0]),"+f"(a[1]),"+f"(a[2]),"+f"(a[3]),"+f"(a[4]),"+f"(a[5]),"+f"(a[6]),"+f"(a[7]),"+f"(a[8]),"+f"(a[9]),"+f"(a[10]),"+f"(a[11]),"+f"(a[12]),"+f"(a[13]),"+f"(a[14]),"+f"(a[15]),"+f"(a[16]),"+f"(a[17]),"+f"(a[18]),"+f"(a[19]),"+f"(a[20]),"+f"(a[21]),"+f"(a[22]),"+f"(a[23]),"+f"(a[24]),"+f"(a[25]),"+f"(a[26]),"+f"(a[27]),"+f"(a[28]),"+f"(a[29]),"+f"(a[30]),"+f"(a[31])
#define ACC64 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}"
#define OUT64O(a,O) \
  "+f"(a[(O)+0]),"+f"(a[(O)+1]),"+f"(a[(O)+2]),"+f"(a[(O)+3]),"+f"(a[(O)+4]),"+f"(a[(O)+5]),"+f"(a[(O)+6]),"+f"(a[(O)+7]),"+f"(a[(O)+8]),"+f"(a[(O)+9]),"+f"(a[(O)+10]),"+f"(a[(O)+11]),"+f"(a[(O)+12]),"+f"(a[(O)+13]),"+f"(a[(O)+14]),"+f"(a[(O)+15]),"+f"(a[(O)+16]),"+f"(a[(O)+17]),"+f"(a[(O)+18]),"+f"(a[(O)+19]),"+f"(a[(O)+20]),"+f"(a[(O)+21]),"+f"(a[(O)+22]),"+f"(a[(O)+23]),"+f"(a[(O)+24]),"+f"(a[(O)+25]),"+f"(a[(O)+26]),"+f"(a[(O)+27]),"+f"(a[(O)+28]),"+f"(a[(O)+29]),"+f"(a[(O)+30]),"+f"(a[(O)+31]),"+f"(a[(O)+32]),"+f"(a[(O)+33]),"+f"(a[(O)+34]),"+f"(a[(O)+35]),"+f"(a[(O)+36]),"+f"(a[(O)+37]),"+f"(a[(O)+38]),"+f"(a[(O)+39]),"+f"(a[(O)+40]),"+f"(a[(O)+41]),"+f"(a[(O)+42]),"+f"(a[(O)+43]),"+f"(a[(O)+44]),"+f"(a[(O)+45]),"+f"(a[(O)+46]),"+f"(a[(O)+47]),"+f"(a[(O)+48]),"+f"(a[(O)+49]),"+f"(a[(O)+50]),"+f"(a[(O)+51]),"+f"(a[(O)+52]),"+f"(a[(O)+53]),"+f"(a[(O)+54]),"+f"(a[(O)+55]),"+f"(a[(O)+56]),"+f"(a[(O)+57]),"+f"(a[(O)+58]),"+f"(a[(O)+59]),"+f"(a[(O)+60]),"+f"(a[(O)+61]),"+f"(a[(O)+62]),"+f"(a[(O)+63])

extern __shared__ __align__(1024) char smem_raw[];

// gather one fp8 row of NCOLS elems into the swizzled K-major tile with ROWS-row col-block stride.
template <int ROWS, int NCOLS> __device__ __forceinline__ void gather8(fp8* sbase, int row, const fp8* src) {
#pragma unroll
    for (int c16 = 0; c16 < NCOLS / 16; ++c16)
        *reinterpret_cast<uint4*>(sbase + koff8R<ROWS>(row, c16 * 16)) = *reinterpret_cast<const uint4*>(src + c16 * 16);
}

// PART 1: QK. Q[128,256], K[64,256]. m64n64k32 (n=CN=64), 8 k-blocks over HD.
__global__ void qk256(const fp8* Q, const fp8* K, float* Sout) {
    fp8* sQ = reinterpret_cast<fp8*>(smem_raw);
    fp8* sK = sQ + CTA_M * HD;               // sQ = 128*256, sK = 64*256
    int tid = threadIdx.x;
    if (tid < CTA_M) gather8<CTA_M, HD>(sQ, tid, Q + tid * HD);
    if (tid < CN)    gather8<CN, HD>(sK, tid, K + tid * HD);
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    int cwg = tid / WG, wtid = tid % WG, warp = wtid >> 5, lane = wtid & 31;
    uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ) + (uint32_t)(cwg * 64 * 128);  // 64 q-rows/WG
    uint32_t sK_a  = (uint32_t)__cvta_generic_to_shared(sK);
    float s[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) s[i] = 0.f;
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < HD / 32; ++kb) {
        uint32_t offA = (uint32_t)((kb & 3) * 32 + (kb >> 2) * (128 * CTA_M));   // Q: CM=128 rows
        uint32_t offB = (uint32_t)((kb & 3) * 32 + (kb >> 2) * (128 * CN));      // K: CN=64 rows
        uint64_t dA = make_smem_desc(sQ_wg + offA, SW_LBO, SW_SBO, 1);
        uint64_t dB = make_smem_desc(sK_a  + offB, SW_LBO, SW_SBO, 1);
        int scaleD = (kb == 0) ? 0 : 1;
        asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %34, 0;\n wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " ACC32 ",%32,%33,p,%35,%36;\n}\n"
                     : OUT32(s) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    int group = lane >> 2, tg = lane & 3;
#pragma unroll
    for (int c = 0; c < 8; ++c)                          // n=64 -> 8 col-groups
        for (int r = 0; r < 2; ++r)
            for (int p = 0; p < 2; ++p) {
                int row = cwg * 64 + warp * 16 + group + r * 8;
                int col = 8 * c + tg * 2 + p;
                Sout[row * CN + col] = s[c * 4 + r * 2 + p];
            }
}

// PART 2: PV-SS. P[128,64] fp8, V[64,256] -> Vt[256,64] K-major. 2 boxes m64n128k32 over HDV, 2 k-blocks.
__global__ void pv256(const float* P, const fp8* V, float* Oout) {
    fp8* sP  = reinterpret_cast<fp8*>(smem_raw);         // [128 q, 64 key] (128-padded: koff8R<128>)
    fp8* sVt = sP + CTA_M * 128;                         // [256 dv, 64 key] (koff8R<256>)
    int tid = threadIdx.x;
    if (tid < CTA_M) {                                   // P[q][key] -> sP swizzled
        const float* prow = P + tid * CN;
        fp8 tmp[16];
#pragma unroll
        for (int c16 = 0; c16 < CN / 16; ++c16) {        // 64/16 = 4 chunks
#pragma unroll
            for (int e = 0; e < 16; ++e) tmp[e] = fp8(prow[c16 * 16 + e]);
            *reinterpret_cast<uint4*>(sP + koff8R<CTA_M>(tid, c16 * 16)) = *reinterpret_cast<const uint4*>(tmp);
        }
    }
    if (tid < CN) {                                      // transpose V[key][dv] -> sVt[dv][key]
        const fp8* vrow = V + tid * HDV;
#pragma unroll
        for (int dv = 0; dv < HDV; ++dv) sVt[koff8R<HDV>(dv, tid)] = vrow[dv];
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    int cwg = tid / WG, wtid = tid % WG, warp = wtid >> 5, lane = wtid & 31;
    uint32_t sP_wg = (uint32_t)__cvta_generic_to_shared(sP) + (uint32_t)(cwg * 64 * 128);
    uint32_t sVt_a = (uint32_t)__cvta_generic_to_shared(sVt);
    float o[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) o[i] = 0.f;
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int box = 0; box < HDV / 128; ++box) {          // 2 output boxes
#pragma unroll
        for (int kb = 0; kb < CN / 32; ++kb) {           // 2 k-blocks over key
            uint32_t off = (uint32_t)(kb * 32);
            uint64_t dA = make_smem_desc(sP_wg + off, SW_LBO, SW_SBO, 1);
            uint64_t dB = make_smem_desc(sVt_a + (uint32_t)(box * 128 * 128) + off, SW_LBO, SW_SBO, 1);
            int scaleD = (kb == 0) ? 0 : 1;
            asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %66, 0;\n wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 " ACC64 ",%64,%65,p,%67,%68;\n}\n"
                         : OUT64O(o, box*64) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1));
        }
    }
    asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    int group = lane >> 2, tg = lane & 3;
#pragma unroll
    for (int box = 0; box < HDV / 128; ++box)
        for (int c = 0; c < 16; ++c)
            for (int r = 0; r < 2; ++r)
                for (int p = 0; p < 2; ++p) {
                    int row = cwg * 64 + warp * 16 + group + r * 8;
                    int col = box * 128 + 8 * c + tg * 2 + p;
                    Oout[row * HDV + col] = o[box * 64 + c * 4 + r * 2 + p];
                }
}

int main() {
    // ---- QK ----
    std::vector<fp8> hQ(CTA_M * HD), hK(CN * HD);
    auto fillf = [](std::vector<fp8>& o, uint64_t seed) {
        uint64_t s = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        for (auto& x : o) { s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            x = fp8(((float)(uint32_t)(s >> 32) / 4294967296.0f) * 2.0f - 1.0f); } };
    fillf(hQ, 11); fillf(hK, 22);
    std::vector<float> Sref(CTA_M * CN);
    for (int i = 0; i < CTA_M; ++i) for (int j = 0; j < CN; ++j) {
        float a = 0; for (int d = 0; d < HD; ++d) a += float(hQ[i*HD+d]) * float(hK[j*HD+d]); Sref[i*CN+j] = a; }
    fp8 *dQ, *dK; float* dS;
    ORC_CHECK(cudaMalloc(&dQ, sizeof(fp8)*CTA_M*HD)); ORC_CHECK(cudaMalloc(&dK, sizeof(fp8)*CN*HD)); ORC_CHECK(cudaMalloc(&dS, sizeof(float)*CTA_M*CN));
    ORC_CHECK(cudaMemcpy(dQ, hQ.data(), sizeof(fp8)*CTA_M*HD, cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dK, hK.data(), sizeof(fp8)*CN*HD, cudaMemcpyHostToDevice));
    int smemQK = (CTA_M * HD + CN * HD) * (int)sizeof(fp8);
    ORC_CHECK(cudaFuncSetAttribute(qk256, cudaFuncAttributeMaxDynamicSharedMemorySize, smemQK));
    qk256<<<1, 256, smemQK>>>(dQ, dK, dS); ORC_CHECK(cudaDeviceSynchronize());
    std::vector<float> hS(CTA_M*CN); ORC_CHECK(cudaMemcpy(hS.data(), dS, sizeof(float)*CTA_M*CN, cudaMemcpyDeviceToHost));
    double md=0; int bi=-1,bj=-1; for (int i=0;i<CTA_M;++i) for (int j=0;j<CN;++j){ double d=std::fabs(hS[i*CN+j]-Sref[i*CN+j]); if(d>md){md=d;bi=i;bj=j;} }
    printf("[QK fp8 hd256] max_abs_diff=%.4f at (%d,%d) gpu=%.4f ref=%.4f  %s\n", md, bi, bj, hS[bi*CN+bj], Sref[bi*CN+bj], md < 0.6 ? "PASS" : "FAIL");

    // ---- PV ----
    std::vector<float> hP(CTA_M * CN); std::vector<fp8> hV(CN * HDV);
    { uint64_t s=7; for (auto&x:hP){ s=s*6364136223846793005ULL+1442695040888963407ULL; x=((float)(uint32_t)(s>>32)/4294967296.0f);} }
    fillf(hV, 9);
    std::vector<float> Oref(CTA_M * HDV);
    for (int i=0;i<CTA_M;++i) for (int j=0;j<HDV;++j){ float a=0; for(int k=0;k<CN;++k) a+=float(fp8(hP[i*CN+k]))*float(hV[k*HDV+j]); Oref[i*HDV+j]=a; }
    float *dP,*dO; fp8* dV;
    ORC_CHECK(cudaMalloc(&dP, sizeof(float)*CTA_M*CN)); ORC_CHECK(cudaMalloc(&dV, sizeof(fp8)*CN*HDV)); ORC_CHECK(cudaMalloc(&dO, sizeof(float)*CTA_M*HDV));
    ORC_CHECK(cudaMemcpy(dP, hP.data(), sizeof(float)*CTA_M*CN, cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dV, hV.data(), sizeof(fp8)*CN*HDV, cudaMemcpyHostToDevice));
    int smemPV = (CTA_M * 128 + HDV * 128) * (int)sizeof(fp8);
    ORC_CHECK(cudaFuncSetAttribute(pv256, cudaFuncAttributeMaxDynamicSharedMemorySize, smemPV));
    pv256<<<1, 256, smemPV>>>(dP, dV, dO); ORC_CHECK(cudaDeviceSynchronize());
    std::vector<float> hO(CTA_M*HDV); ORC_CHECK(cudaMemcpy(hO.data(), dO, sizeof(float)*CTA_M*HDV, cudaMemcpyDeviceToHost));
    double mo=0; int oi=-1,oj=-1; for (int i=0;i<CTA_M;++i) for (int j=0;j<HDV;++j){ double d=std::fabs(hO[i*HDV+j]-Oref[i*HDV+j]); if(d>mo){mo=d;oi=i;oj=j;} }
    printf("[PV fp8 hd256] max_abs_diff=%.4f at (%d,%d) gpu=%.4f ref=%.4f  %s\n", mo, oi, oj, hO[oi*HDV+oj], Oref[oi*HDV+oj], mo < 0.6 ? "PASS" : "FAIL");
    return 0;
}
