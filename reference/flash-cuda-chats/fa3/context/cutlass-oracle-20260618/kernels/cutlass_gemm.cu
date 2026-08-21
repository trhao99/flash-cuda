// CUTLASS SM90 warp-specialized GEMM oracle.
//
// Reproduces the FA3 hd128-bf16 prefill GEMM building blocks as a *standalone* CUTLASS GEMM
// so we can (a) inspect the generated PTX/SASS (wgmma + TMA + mbarrier) and (b) get the
// CUTLASS latency/output to match with a pure-CUDA/PTX kernel.
//
//   GEMM_MODE=QK : C[M,N] = A[M,K] . B[N,K]^T   (A row-major, B row-major => "TN")  K=head_dim
//   GEMM_MODE=PV : C[M,N] = A[M,K] . B[K,N]      (A row-major, B row-major => "NN")  N=head_dim_v
//
// bf16 inputs, fp32 accumulate, fp32 output. Build with: -DGEMM_MODE_QK or -DGEMM_MODE_PV.
//
// Usage: ./cutlass_gemm <M> <N> <K> [out.bin] [iters] [trials]

#include <iostream>
#include <string>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "oracle_common.cuh"

using namespace cute;

using Element  = cutlass::bfloat16_t;
using ElementAcc = float;
using ElementOut = float;

#if defined(GEMM_MODE_PV)
//  PV: C[M,N] = A[M,K] . B[K,N], both row-major  (NN)
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
static const char* kModeName = "PV";
#else
//  QK: C[M,N] = A[M,K] . B[N,K]^T  => B stored row-major (N,K) == column-major (K,N)  (TN)
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
static const char* kModeName = "QK";
#endif
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignA = 128 / cutlass::sizeof_bits<Element>::value;     // 8
constexpr int AlignB = 128 / cutlass::sizeof_bits<Element>::value;     // 8
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementOut>::value;  // 4

// Tile shape configurable so we can sweep for CUTLASS's *best* config (fair oracle baseline).
#ifndef TILE_M
#define TILE_M 128
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 64
#endif
using TileShape    = Shape<cute::Int<TILE_M>, cute::Int<TILE_N>, cute::Int<TILE_K>>;
using ClusterShape = Shape<_1, _1, _1>;        // FA3 prefill (varlen) uses no cluster

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementOut, LayoutC, AlignC,
    ElementOut, LayoutC, AlignC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Element, LayoutA, AlignA,
    Element, LayoutB, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

int main(int argc, char** argv) {
    int M = argc > 1 ? std::atoi(argv[1]) : 2048;
    int N = argc > 2 ? std::atoi(argv[2]) : 2048;
    int K = argc > 3 ? std::atoi(argv[3]) : 128;
    std::string out_path = argc > 4 ? argv[4] : "";
    int iters  = argc > 5 ? std::atoi(argv[5]) : 100;
    int trials = argc > 6 ? std::atoi(argv[6]) : 7;

    // ---- deterministic bf16 inputs (shared with the pure-CUDA kernel) ----
    std::vector<uint16_t> hA, hB;
    orc::fill_bf16(hA, size_t(M) * K, /*seed=*/1);
    orc::fill_bf16(hB, size_t(N) * K, /*seed=*/2);  // B is N*K elements either layout

    Element *dA, *dB; ElementOut *dC, *dD;
    ORC_CHECK(cudaMalloc(&dA, sizeof(Element) * size_t(M) * K));
    ORC_CHECK(cudaMalloc(&dB, sizeof(Element) * size_t(N) * K));
    ORC_CHECK(cudaMalloc(&dC, sizeof(ElementOut) * size_t(M) * N));
    ORC_CHECK(cudaMalloc(&dD, sizeof(ElementOut) * size_t(M) * N));
    ORC_CHECK(cudaMemcpy(dA, hA.data(), sizeof(Element) * size_t(M) * K, cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dB, hB.data(), sizeof(Element) * size_t(N) * K, cudaMemcpyHostToDevice));

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {dA, stride_A, dB, stride_B},
        {{/*alpha=*/1.0f, /*beta=*/0.0f}, dC, stride_C, dD, stride_D}
    };

    Gemm gemm;
    cutlass::Status st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "can_implement failed: %d\n", int(st)); return 2;
    }
    size_t ws = Gemm::get_workspace_size(args);
    void* dWs = nullptr;
    if (ws) ORC_CHECK(cudaMalloc(&dWs, ws));
    st = gemm.initialize(args, dWs);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "initialize failed: %d\n", int(st)); return 3;
    }

    auto launch = [&]() { gemm.run(); };
    launch();
    ORC_CHECK(cudaDeviceSynchronize());
    ORC_CHECK(cudaGetLastError());

    float median = 0.f;
    float best = orc::benchmark_ms(launch, /*warmup=*/50, iters, trials, &median);

    double gflop = orc::gemm_gflop(M, N, K);
    std::printf("{\"impl\": \"cutlass\", \"mode\": \"%s\", \"M\": %d, \"N\": %d, \"K\": %d, "
                "\"min_ms\": %.6f, \"median_ms\": %.6f, \"tflops\": %.2f}\n",
                kModeName, M, N, K, best, median, gflop / (median * 1e-3) / 1e3);

    if (!out_path.empty()) {
        std::vector<float> hD(size_t(M) * N);
        ORC_CHECK(cudaMemcpy(hD.data(), dD, sizeof(ElementOut) * size_t(M) * N, cudaMemcpyDeviceToHost));
        orc::dump_f32(out_path, hD);
    }
    return 0;
}
