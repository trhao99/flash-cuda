// Reference attention (single batch/head) + input gen + compare, shared by the softmax-numerics
// test and the fused-kernel A/B. Ground truth is a textbook 2-pass softmax in fp32 (max-subtract,
// expf). Everything bf16-in / fp32-accum, matching FA3's dtypes.
#pragma once
#include "oracle_common.cuh"

namespace attn {

struct Problem { int S; int D; int Dv; bool causal; float scale; };

// Deterministic bf16 inputs (reuse oracle LCG with distinct seeds per tensor).
inline void gen_inputs(const Problem& p, std::vector<uint16_t>& Q, std::vector<uint16_t>& K,
                       std::vector<uint16_t>& V) {
    orc::fill_bf16(Q, size_t(p.S) * p.D, 11);
    orc::fill_bf16(K, size_t(p.S) * p.D, 22);
    orc::fill_bf16(V, size_t(p.S) * p.Dv, 33);
}

// CPU ground truth: O[S,Dv] fp32, LSE[S] fp32. 2-pass stable softmax with natural exp/log.
inline void ref_cpu(const Problem& p, const std::vector<uint16_t>& Q, const std::vector<uint16_t>& K,
                    const std::vector<uint16_t>& V, std::vector<float>& O, std::vector<float>& LSE) {
    O.assign(size_t(p.S) * p.Dv, 0.f);
    LSE.assign(p.S, 0.f);
    for (int i = 0; i < p.S; ++i) {
        int kmax = p.causal ? i + 1 : p.S;     // causal: key j allowed iff j <= i (aligned S==S)
        double m = -1e30;
        for (int j = 0; j < kmax; ++j) {
            double s = 0;
            for (int d = 0; d < p.D; ++d)
                s += double(orc::bf16_bits_to_f32(Q[size_t(i) * p.D + d])) *
                     double(orc::bf16_bits_to_f32(K[size_t(j) * p.D + d]));
            s *= p.scale;
            if (s > m) m = s;
        }
        double l = 0;
        std::vector<double> acc(p.Dv, 0.0);
        for (int j = 0; j < kmax; ++j) {
            double s = 0;
            for (int d = 0; d < p.D; ++d)
                s += double(orc::bf16_bits_to_f32(Q[size_t(i) * p.D + d])) *
                     double(orc::bf16_bits_to_f32(K[size_t(j) * p.D + d]));
            s *= p.scale;
            double e = std::exp(s - m);
            l += e;
            for (int d = 0; d < p.Dv; ++d)
                acc[d] += e * double(orc::bf16_bits_to_f32(V[size_t(j) * p.Dv + d]));
        }
        double inv = l > 0 ? 1.0 / l : 0.0;
        for (int d = 0; d < p.Dv; ++d) O[size_t(i) * p.Dv + d] = float(acc[d] * inv);
        LSE[i] = float(m + std::log(l));
    }
}

// max |abs| and mean |abs| diff over two fp32 buffers.
inline void diff(const std::vector<float>& a, const std::vector<float>& b, double& maxd, double& meand) {
    maxd = 0; double s = 0; size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) { double d = std::fabs(double(a[i]) - double(b[i]));
        if (d > maxd) maxd = d; s += d; }
    meand = n ? s / n : 0;
}

}  // namespace attn
