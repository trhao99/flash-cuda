// fa4_tma_load.cu — validate TMA (cp.async.bulk.tensor) load into the koff 128B-swizzle layout on
// SM100. This is the load primitive that frees the pipeline (hardware does the swizzled copy async),
// replacing the per-element swizzled stores that made the warp-specialized probe load-bound.
//
// Loads a [ROWS,HD] bf16 tile via 2 TMA boxes (one per 64-col block) into koff-swizzled smem, then
// reads back via koff() and checks byte-exactness vs the source. Reuses FA3's proven tensormap
// recipe (make_kv_map_contig / tma_box). Build with -lcuda (driver API for cuTensorMapEncodeTiled).
//
// Build: nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 fa4_tma_load.cu -o /tmp/tma -lcuda
#include <cuda.h>
#include "probe_common.cuh"

constexpr int ROWS=128, HD=128;

__device__ __forceinline__ int koff(int r,int c){ return (c>>6)*8192 + r*64 + ((c&63)^((r&7)<<3)); }
__device__ __forceinline__ void tma_box(void* dst, const CUtensorMap* map, int x, int y, uint64_t* bar){
    uint32_t s=(uint32_t)__cvta_generic_to_shared(dst), b=(uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];"
                 :: "r"(s),"l"(map),"r"(x),"r"(y),"r"(b) : "memory"); }

__global__ void tma_kernel(const CUtensorMap* __restrict__ map, uint16_t* dumpK){
    __shared__ __align__(1024) __nv_bfloat16 sK[ROWS*HD];
    __shared__ __align__(8) uint64_t bar;
    int tid=threadIdx.x;
    if(tid==0){
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar)));
        // expect the full [ROWS,HD] tile bytes (2 col-blocks x ROWS x 64 x 2 bytes)
        uint32_t bb=(uint32_t)__cvta_generic_to_shared(&bar);
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(bb), "r"((uint32_t)(ROWS*HD*2)));
        tma_box(&sK[0],    map, 0,  0, &bar);   // col-block 0 (cols 0..63)
        tma_box(&sK[8192], map, 64, 0, &bar);   // col-block 1 (cols 64..127)
    }
    // wait (phase 0)
    asm volatile("{ .reg .pred P;\nLW: mbarrier.try_wait.parity.shared.b64 P,[%0],0,10000000;\n @P bra D; bra LW;\nD: }"
                 :: "r"((uint32_t)__cvta_generic_to_shared(&bar)));
    __syncthreads();
    // dump sK in [r,c] order via koff so host can compare to source
    for(int i=tid;i<ROWS*HD;i+=blockDim.x){ int r=i/HD,c=i%HD; __nv_bfloat16 v=sK[koff(r,c)]; uint16_t u; memcpy(&u,&v,2); dumpK[r*HD+c]=u; }
}

int main(){
    ORC_CHECK(cudaSetDevice(0));
    cudaDeviceProp p{}; ORC_CHECK(cudaGetDeviceProperties(&p,0));
    std::printf("device: %s\n", p.name);
    std::vector<uint16_t> Kh; orc::fill_bf16(Kh, ROWS*HD, 7);
    __nv_bfloat16* dK; ORC_CHECK(cudaMalloc(&dK, ROWS*HD*2));
    ORC_CHECK(cudaMemcpy(dK, Kh.data(), ROWS*HD*2, cudaMemcpyHostToDevice));

    // tensormap over [ROWS rows, HD cols] bf16, box [64 cols, ROWS rows], 128B swizzle
    CUtensorMap map{};
    uint64_t gdim[2]={(uint64_t)HD,(uint64_t)ROWS};        // inner (cols), outer (rows)
    uint64_t gstride[1]={(uint64_t)(HD*2)};                // bytes per row
    uint32_t bdim[2]={64u,(uint32_t)ROWS};                 // box: 64 cols x ROWS rows
    uint32_t estride[2]={1u,1u};
    CUresult cr=cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dK,
        gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(cr!=CUDA_SUCCESS){ std::printf("cuTensorMapEncodeTiled FAILED: %d\n",(int)cr); return 1; }
    CUtensorMap* dmap; ORC_CHECK(cudaMalloc(&dmap,sizeof(CUtensorMap)));
    ORC_CHECK(cudaMemcpy(dmap,&map,sizeof(CUtensorMap),cudaMemcpyHostToDevice));

    uint16_t* dDump; ORC_CHECK(cudaMalloc(&dDump, ROWS*HD*2)); ORC_CHECK(cudaMemset(dDump,0,ROWS*HD*2));
    tma_kernel<<<1,128>>>(dmap, dDump);
    ORC_CHECK(cudaGetLastError()); ORC_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> got(ROWS*HD); ORC_CHECK(cudaMemcpy(got.data(),dDump,ROWS*HD*2,cudaMemcpyDeviceToHost));
    int bad=0, shown=0;
    for(int i=0;i<ROWS*HD;i++){ if(got[i]!=Kh[i]){ bad++; if(shown<6){ std::printf("  mismatch i=%d r=%d c=%d got=0x%x want=0x%x\n",i,i/HD,i%HD,got[i],Kh[i]); shown++; } } }
    std::printf("%s: TMA load -> koff swizzle layout (%d/%d mismatches)\n", bad==0?"PASS":"FAIL", bad, ROWS*HD);
    return bad==0?0:1;
}
