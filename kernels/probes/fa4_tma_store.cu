// fa4_tma_store.cu — validate TMA store (cp.async.bulk.tensor .global.shared) on SM100, the O
// epilogue primitive the Blackwell-native kernel uses (O tmem -> smem -> TMA store to gmem).
// Round-trips a [ROWS,HDV] bf16 tile: gmem -> smem (plain) -> TMA store -> gmem2, checks gmem2==gmem.
//
// Build: nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 fa4_tma_store.cu -o /tmp/tst -lcuda
#include <cuda.h>
#include "probe_common.cuh"

constexpr int ROWS=128, HDV=128;

__device__ __forceinline__ void tma_store(const void* src, const CUtensorMap* map, int x, int y){
    uint32_t s=(uint32_t)__cvta_generic_to_shared(src);
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%1,%2}], [%3];"
                 :: "l"(map),"r"(x),"r"(y),"r"(s) : "memory");
    asm volatile("cp.async.bulk.commit_group;");
    asm volatile("cp.async.bulk.wait_group 0;");
}

__global__ void store_kernel(const __nv_bfloat16* __restrict__ src, const CUtensorMap* __restrict__ map){
    __shared__ __align__(1024) __nv_bfloat16 sO[ROWS*HDV];   // plain row-major (no swizzle) for the store test
    int tid=threadIdx.x;
    for(int i=tid;i<ROWS*HDV;i+=blockDim.x) sO[i]=src[i];    // gmem -> smem
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    if(tid==0) tma_store(&sO[0], map, 0, 0);                 // smem -> gmem via TMA
    __syncthreads();
}

int main(){
    ORC_CHECK(cudaSetDevice(0));
    cudaDeviceProp p{}; ORC_CHECK(cudaGetDeviceProperties(&p,0));
    std::printf("device: %s\n", p.name);
    std::vector<uint16_t> Sh; orc::fill_bf16(Sh, ROWS*HDV, 11);
    __nv_bfloat16 *dSrc,*dDst; ORC_CHECK(cudaMalloc(&dSrc,ROWS*HDV*2)); ORC_CHECK(cudaMalloc(&dDst,ROWS*HDV*2));
    ORC_CHECK(cudaMemcpy(dSrc,Sh.data(),ROWS*HDV*2,cudaMemcpyHostToDevice)); ORC_CHECK(cudaMemset(dDst,0,ROWS*HDV*2));
    // tensormap over the DESTINATION [ROWS,HDV] bf16, box [HDV cols, ROWS rows], no swizzle (plain store)
    CUtensorMap map{};
    uint64_t gdim[2]={(uint64_t)HDV,(uint64_t)ROWS};
    uint64_t gstride[1]={(uint64_t)(HDV*2)};
    uint32_t bdim[2]={(uint32_t)HDV,(uint32_t)ROWS};
    uint32_t estride[2]={1u,1u};
    CUresult cr=cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dDst,
        gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(cr!=CUDA_SUCCESS){ std::printf("encode FAILED %d\n",(int)cr); return 1; }
    CUtensorMap* dmap; ORC_CHECK(cudaMalloc(&dmap,sizeof(CUtensorMap))); ORC_CHECK(cudaMemcpy(dmap,&map,sizeof(CUtensorMap),cudaMemcpyHostToDevice));
    store_kernel<<<1,128>>>(dSrc,dmap);
    ORC_CHECK(cudaGetLastError()); ORC_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> got(ROWS*HDV); ORC_CHECK(cudaMemcpy(got.data(),dDst,ROWS*HDV*2,cudaMemcpyDeviceToHost));
    int bad=0; for(int i=0;i<ROWS*HDV;i++) if(got[i]!=Sh[i]) bad++;
    std::printf("%s: TMA store round-trip (%d/%d mismatches)\n", bad==0?"PASS":"FAIL", bad, ROWS*HDV);
    return bad==0?0:1;
}
