// fa4_tmem_roundtrip.cu — Blackwell SM100 tensor-memory (tmem) round-trip probe.
//
// The foundational FA4 building block: allocate tmem, store a known per-(lane,col) pattern via
// tcgen05.st, read it back via tcgen05.ld, and verify (a) the round trip is bit-exact and (b) the
// thread->(lane,col) mapping is exactly what the ISA says:
//   * a 128-thread warpgroup's 4 warps own tmem lanes [0,32)/[32,64)/[64,96)/[96,128)
//     (parallel-thread-execution.md:23958-23965),
//   * within a warp, tcgen05{.ld,.st}.32x32b spreads the 32 threads across its 32 lanes,
//   * so thread tid (0..127) owns tmem lane tid; a `.x1` at column `c` moves tmem[lane=tid, col=c].
//   * tmem address = base + (lane<<16) + col  (DP stride 1<<16; tmem_allocator_sm100.hpp:49-53).
//
// Build:  nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 fa4_tmem_roundtrip.cu -o /tmp/rt
// Run:    /tmp/rt         (prints PASS/FAIL; exit 0 on pass)
#include "probe_common.cuh"

// ---- tcgen05 tmem intrinsics (raw inline PTX) ----------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void tmem_alloc(uint32_t smem_dst_addr, int ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_dst_addr), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols));
}
__device__ __forceinline__ void tmem_st_1(uint32_t taddr, uint32_t val) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" :: "r"(taddr), "r"(val));
}
__device__ __forceinline__ uint32_t tmem_ld_1(uint32_t taddr) {
    uint32_t v;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(v) : "r"(taddr));
    return v;
}
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;"); }
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }

constexpr int LANES = 128;   // rows == 128 threads (one warpgroup)
constexpr int COLS  = 128;   // columns to round-trip

// pattern encodes (lane, col) exactly in 32 bits: (lane<<10)|col  (lane<128, col<128).
__device__ __forceinline__ uint32_t pattern(int lane, int col) { return (uint32_t(lane) << 10) | uint32_t(col); }

__global__ void __launch_bounds__(128) tmem_roundtrip_kernel(int* n_bad, uint32_t* dump) {
    __shared__ uint32_t s_base;   // tmem base address written by alloc
    const int tid = threadIdx.x;  // 0..127
    const int warp = tid >> 5;    // 0..3
    const int lane_base = warp * 32;       // lane owned by this warp's laneid-0
    const int lane = tid;                  // ISA: thread tid -> tmem lane tid within the warpgroup

    // 1) allocate tmem columns (single warp issues alloc; result -> shared).
    if (warp == 0) { tmem_alloc(smem_u32(&s_base), COLS); tmem_relinquish(); }
    __syncthreads();
    const uint32_t base = s_base;

    // 2) store the known pattern: each thread writes its whole 128-col row, column by column.
    for (int c = 0; c < COLS; ++c) {
        uint32_t taddr = base + (uint32_t(lane) << 16) + uint32_t(c);
        tmem_st_1(taddr, pattern(lane, c));
    }
    tmem_wait_st();
    __syncthreads();

    // 3) read back and verify bit-exactly.
    int bad = 0;
    for (int c = 0; c < COLS; ++c) {
        uint32_t taddr = base + (uint32_t(lane) << 16) + uint32_t(c);
        uint32_t got = tmem_ld_1(taddr);
        tmem_wait_ld();
        uint32_t want = pattern(lane, c);
        if (dump) dump[lane * COLS + c] = got;
        if (got != want) bad++;
    }
    if (bad) atomicAdd(n_bad, bad);

    // 4) free.
    __syncthreads();
    if (warp == 0) tmem_dealloc(base, COLS);
    (void)lane_base;
}

int main() {
    int dev = 0; ORC_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{}; ORC_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device: %s cc %d.%d\n", prop.name, prop.major, prop.minor);

    int* d_bad; uint32_t* d_dump;
    ORC_CHECK(cudaMalloc(&d_bad, sizeof(int)));
    ORC_CHECK(cudaMalloc(&d_dump, sizeof(uint32_t) * LANES * COLS));
    ORC_CHECK(cudaMemset(d_bad, 0, sizeof(int)));
    ORC_CHECK(cudaMemset(d_dump, 0xff, sizeof(uint32_t) * LANES * COLS));

    tmem_roundtrip_kernel<<<1, 128>>>(d_bad, d_dump);
    ORC_CHECK(cudaGetLastError());
    ORC_CHECK(cudaDeviceSynchronize());

    int bad = 0; ORC_CHECK(cudaMemcpy(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<uint32_t> dump(LANES * COLS);
    ORC_CHECK(cudaMemcpy(dump.data(), d_dump, sizeof(uint32_t) * LANES * COLS, cudaMemcpyDeviceToHost));

    // Host cross-check of the absolute mapping: dump[lane*COLS+c] must equal pattern(lane,c).
    int host_bad = 0, shown = 0;
    for (int lane = 0; lane < LANES; ++lane)
        for (int c = 0; c < COLS; ++c) {
            uint32_t want = (uint32_t(lane) << 10) | uint32_t(c);
            if (dump[lane * COLS + c] != want) {
                host_bad++;
                if (shown < 8) { std::printf("  mismatch lane=%d col=%d got=0x%x want=0x%x\n",
                                             lane, c, dump[lane * COLS + c], want); shown++; }
            }
        }
    std::printf("device-side bad=%d  host-check bad=%d  (of %d cells)\n", bad, host_bad, LANES * COLS);
    bool pass = (bad == 0 && host_bad == 0);
    std::printf("%s: tmem alloc + st/ld .32x32b round-trip + thread->lane mapping\n", pass ? "PASS" : "FAIL");
    ORC_CHECK(cudaFree(d_bad)); ORC_CHECK(cudaFree(d_dump));
    return pass ? 0 : 1;
}
