// Math/numerics component: FA3 online softmax (log2 domain) in isolation, validated vs the textbook
// 2-pass reference. Faithfully mirrors flash::Softmax (softmax.h): RAW S, scale applied in exp via
// s2 = softmax_scale*log2(e); per-n-block running max/sum rescale; exp2f (-> ex2.approx.f32);
// LSE = softmax_scale*row_max + ln(row_sum). One thread per query row (this validates the *numerics*,
// not the wgmma data path which the GEMM oracle already covers).
//
// Usage: ./softmax_test <S> <D> [causal=1]

#include "attn_ref.cuh"

constexpr int BN = 128;   // kBlockN

// FA3-faithful online attention, one thread per query row. exp2f/__logf as in softmax.h.
__global__ void online_softmax_attn(const __nv_bfloat16* __restrict__ Q,
                                    const __nv_bfloat16* __restrict__ K,
                                    const __nv_bfloat16* __restrict__ V,
                                    float* __restrict__ O, float* __restrict__ LSE,
                                    int S, int D, int Dv, int causal, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // query row
    if (i >= S) return;
    const float s2 = scale * float(M_LOG2E);
    float q[128];
    for (int d = 0; d < D; ++d) q[d] = __bfloat162float(Q[size_t(i) * D + d]);

    float m = -INFINITY, l = 0.f, acc[128];
    for (int d = 0; d < Dv; ++d) acc[d] = 0.f;

    int kmax = causal ? i + 1 : S;
    for (int j0 = 0; j0 < kmax; j0 += BN) {
        // GEMM0 + mask: raw S for this block.
        float s_blk[BN]; float blk_max = -INFINITY;
        for (int n = 0; n < BN; ++n) {
            int j = j0 + n;
            float s = -INFINITY;
            if (j < kmax) {              // causal/seqlen mask
                s = 0.f;
                for (int d = 0; d < D; ++d) s += q[d] * __bfloat162float(K[size_t(j) * D + d]);
            }
            s_blk[n] = s;
            if (s > blk_max) blk_max = s;
        }
        bool is_first = (j0 == 0);
        // max_get_scale
        float scores_scale;
        if (is_first) { m = blk_max; scores_scale = 1.f; }
        else {
            float prev = m; m = fmaxf(m, blk_max);
            float cur = (m == -INFINITY) ? 0.f : m;
            scores_scale = exp2f((prev - cur) * s2);
            l *= scores_scale;
        }
        // online_softmax: P = exp2(S*s2 - row_max*s2); row_sum += sum P
        float max_scaled = (m == -INFINITY) ? 0.f : m * s2;
        float blk_sum = 0.f;
        for (int n = 0; n < BN; ++n) {
            float p = exp2f(s_blk[n] * s2 - max_scaled);
            s_blk[n] = p; blk_sum += p;
        }
        l = is_first ? blk_sum : l + blk_sum;
        // rescale_o then GEMM1 (O += P*V)
        for (int d = 0; d < Dv; ++d) acc[d] *= scores_scale;
        for (int n = 0; n < BN; ++n) {
            int j = j0 + n; if (j >= kmax) continue;
            float p = s_blk[n];
            for (int d = 0; d < Dv; ++d) acc[d] += p * __bfloat162float(V[size_t(j) * Dv + d]);
        }
    }
    // finalize
    float inv = (l == 0.f || l != l) ? 0.f : 1.f / l;
    for (int d = 0; d < Dv; ++d) O[size_t(i) * Dv + d] = acc[d] * inv;
    LSE[i] = (l == 0.f || l != l) ? -INFINITY : m * scale + __logf(l);
}

int main(int argc, char** argv) {
    attn::Problem p;
    p.S  = argc > 1 ? std::atoi(argv[1]) : 1024;
    p.D  = argc > 2 ? std::atoi(argv[2]) : 128;
    p.Dv = p.D;
    p.causal = argc > 3 ? std::atoi(argv[3]) : 1;
    p.scale = 1.f / std::sqrt(float(p.D));

    std::vector<uint16_t> hQ, hK, hV;
    attn::gen_inputs(p, hQ, hK, hV);
    std::vector<float> O_ref, LSE_ref;
    attn::ref_cpu(p, hQ, hK, hV, O_ref, LSE_ref);

    __nv_bfloat16 *dQ, *dK, *dV; float *dO, *dLSE;
    ORC_CHECK(cudaMalloc(&dQ, sizeof(__nv_bfloat16) * hQ.size()));
    ORC_CHECK(cudaMalloc(&dK, sizeof(__nv_bfloat16) * hK.size()));
    ORC_CHECK(cudaMalloc(&dV, sizeof(__nv_bfloat16) * hV.size()));
    ORC_CHECK(cudaMalloc(&dO, sizeof(float) * size_t(p.S) * p.Dv));
    ORC_CHECK(cudaMalloc(&dLSE, sizeof(float) * p.S));
    ORC_CHECK(cudaMemcpy(dQ, hQ.data(), sizeof(__nv_bfloat16) * hQ.size(), cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dK, hK.data(), sizeof(__nv_bfloat16) * hK.size(), cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dV, hV.data(), sizeof(__nv_bfloat16) * hV.size(), cudaMemcpyHostToDevice));

    int threads = 128, blocks = (p.S + threads - 1) / threads;
    online_softmax_attn<<<blocks, threads>>>(dQ, dK, dV, dO, dLSE, p.S, p.D, p.Dv, p.causal, p.scale);
    ORC_CHECK(cudaDeviceSynchronize());
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "launch: %s\n", cudaGetErrorString(e)); return 3; }

    std::vector<float> O(size_t(p.S) * p.Dv), LSE(p.S);
    ORC_CHECK(cudaMemcpy(O.data(), dO, sizeof(float) * O.size(), cudaMemcpyDeviceToHost));
    ORC_CHECK(cudaMemcpy(LSE.data(), dLSE, sizeof(float) * p.S, cudaMemcpyDeviceToHost));

    double o_max, o_mean, l_max, l_mean;
    attn::diff(O, O_ref, o_max, o_mean);
    attn::diff(LSE, LSE_ref, l_max, l_mean);
    std::printf("{\"comp\": \"softmax\", \"S\": %d, \"D\": %d, \"causal\": %d, "
                "\"O_max_abs\": %.3e, \"O_mean_abs\": %.3e, \"LSE_max_abs\": %.3e, \"LSE_mean_abs\": %.3e}\n",
                p.S, p.D, p.causal, o_max, o_mean, l_max, l_mean);
    // bf16 inputs + fp32 accum: O within a few e-3, LSE within ~1e-3 of the fp64-ish ref.
    return (o_max < 5e-3 && l_max < 5e-3) ? 0 : 5;
}
