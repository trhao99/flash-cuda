// Shared harness for the CUTLASS-oracle GEMM benchmarks.
//
// Both the CUTLASS reference GEMM and the pure-CUDA/PTX GEMM include this header so they
// (a) generate BIT-IDENTICAL bf16 inputs (deterministic LCG -> fp32 -> bf16 bits) and
// (b) time and dump their fp32 output the same way, so a Python comparator can check both
//     latency (match CUTLASS) and output (stay aligned) on the exact same problem.
//
// NOTE: this is oracle/benchmark scaffolding, NOT part of the agent's deliverable .cu.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define ORC_CHECK(call) do {                                                          \
    cudaError_t _e = (call);                                                          \
    if (_e != cudaSuccess) {                                                          \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,         \
                     __LINE__, cudaGetErrorString(_e));                               \
        std::exit(1);                                                                 \
    }                                                                                 \
} while (0)

namespace orc {

// fp32 -> bf16 bits, round-to-nearest-even (matches __nv_bfloat16 / cutlass::bfloat16_t).
inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t lsb = (x >> 16) & 1u;
    uint32_t rounding_bias = 0x7fffu + lsb;
    x += rounding_bias;
    return static_cast<uint16_t>(x >> 16);
}

inline float bf16_bits_to_f32(uint16_t h) {
    uint32_t x = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// Deterministic input fill: LCG -> value in [-1, 1) -> bf16 bits. Identical across binaries.
inline void fill_bf16(std::vector<uint16_t>& out, size_t n, uint64_t seed) {
    out.resize(n);
    uint64_t s = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    for (size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t bits = static_cast<uint32_t>(s >> 32);
        float v = (static_cast<float>(bits) / 4294967296.0f) * 2.0f - 1.0f;  // [-1, 1)
        out[i] = f32_to_bf16_bits(v);
    }
}

// CUDA-event timing of a launch lambda. Warmup then timed; returns min ms over trials, and
// fills `median_ms`. Flushes a scratch buffer through L2 between trials (H200 L2 ~ 60MB).
template <typename LaunchFn>
inline float benchmark_ms(LaunchFn&& launch, int warmup, int iters, int trials,
                          float* median_ms_out = nullptr, size_t l2_flush_bytes = 128ull << 20) {
    int* flush = nullptr;
    if (l2_flush_bytes) ORC_CHECK(cudaMalloc(&flush, l2_flush_bytes));

    cudaEvent_t a, b;
    ORC_CHECK(cudaEventCreate(&a));
    ORC_CHECK(cudaEventCreate(&b));

    for (int i = 0; i < warmup; ++i) launch();
    ORC_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;
    times.reserve(trials);
    for (int t = 0; t < trials; ++t) {
        if (flush) ORC_CHECK(cudaMemsetAsync(flush, t & 0xff, l2_flush_bytes));
        ORC_CHECK(cudaEventRecord(a));
        for (int i = 0; i < iters; ++i) launch();
        ORC_CHECK(cudaEventRecord(b));
        ORC_CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        ORC_CHECK(cudaEventElapsedTime(&ms, a, b));
        times.push_back(ms / iters);
    }
    std::sort(times.begin(), times.end());
    if (median_ms_out) *median_ms_out = times[times.size() / 2];
    if (flush) ORC_CHECK(cudaFree(flush));
    ORC_CHECK(cudaEventDestroy(a));
    ORC_CHECK(cudaEventDestroy(b));
    return times.front();
}

inline void dump_f32(const std::string& path, const std::vector<float>& v) {
    if (path.empty()) return;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return; }
    std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
}

// GFLOP for a GEMM with the given problem size (2*M*N*K).
inline double gemm_gflop(long M, long N, long K) {
    return 2.0 * double(M) * double(N) * double(K) / 1e9;
}

}  // namespace orc
