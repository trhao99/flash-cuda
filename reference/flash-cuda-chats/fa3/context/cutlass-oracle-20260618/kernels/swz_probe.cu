// Ground-truth probe: dump cute's GMMA SW128 smem layout offsets + descriptor LBO/SBO so we can
// hardcode a bit-exact swizzle for the pure-PTX TMA path. Host-only (cute layouts are constexpr).
#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/bfloat16.h>
using namespace cute;

template <class Layout>
void dump(const char* name, Layout lay, int nI, int nJ) {
    printf("=== %s  (I=%d rows[MN], J=%d cols[K]) ===\n", name, nI, nJ);
    for (int i = 0; i < nI; ++i) {
        printf("i=%3d:", i);
        for (int j = 0; j < nJ; ++j) printf(" %5d", (int)lay(i, j));
        printf("\n");
    }
}

int main() {
    using T = cutlass::bfloat16_t;
    // K-major (K contiguous): used for Q and K operands of QK GEMM.
    auto atomK = GMMA::Layout_K_SW128_Atom<T>{};
    auto layK  = tile_to_shape(atomK, Shape<_128, _128>{});  // [MN=128, K=128]
    dump("K_SW128 [128,128]", layK, 16, 128);

    // MN-major (MN contiguous): used for V operand of PV GEMM (trans-b path).
    auto atomMN = GMMA::Layout_MN_SW128_Atom<T>{};
    auto layMN  = tile_to_shape(atomMN, Shape<_128, _128>{});  // [MN=128, K=128]
    dump("MN_SW128 [128,128]", layMN, 16, 128);

    // total sizes
    printf("cosize K=%d  MN=%d\n", (int)cosize(layK), (int)cosize(layMN));

    // ---- descriptors (LBO/SBO/base_offset/layout_type) exactly as CUTLASS builds them ----
    // dummy smem-tagged tensors (start address is garbage; LBO/SBO/lt depend only on layout)
    auto layK16  = composition(layK,  make_tile(Layout<_128>{}, Layout<_16>{}));
    auto layMN16 = composition(layMN, make_tile(Layout<_128>{}, Layout<_16>{}));
    auto tK16  = make_tensor(make_smem_ptr((T*)nullptr), layK16);
    auto tMN16 = make_tensor(make_smem_ptr((T*)nullptr), layMN16);
    printf("\n--- make_gmma_desc<Major::K>(K_SW128 [128,16]) ---\n");
    cute::print(SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::K>(tK16));
    printf("\n--- make_gmma_desc<Major::MN>(MN_SW128 [128,16]) ---\n");
    cute::print(SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::MN>(tMN16));

    // start-address advance per wgmma k-step (k+=16) at mn=0 (element offsets)
    printf("\nK k-step starts (mn=0):");
    for (int kb = 0; kb < 8; ++kb) printf(" %d", (int)layK(0, kb * 16));
    printf("\nMN k-step starts (mn=0):");
    for (int kb = 0; kb < 8; ++kb) printf(" %d", (int)layMN(0, kb * 16));
    printf("\n");
    // MN dv-block boundary (mn=dv): rows 62..66 at k=0..3 to see how dv>=64 is placed
    printf("\nMN layout around dv-block boundary (rows=dv, cols=k):\n");
    for (int mn = 62; mn <= 66; ++mn) {
        printf("dv=%3d:", mn);
        for (int k = 0; k < 4; ++k) printf(" %5d", (int)layMN(mn, k));
        printf("\n");
    }
    // K-major: full mn sweep at k=0 to confirm koff matches cute for all rows
    printf("\nK layout (mn,0) vs koff: ");
    {
        auto koffh = [](int mn, int k){ int o = mn*64 + (k&63); o ^= (((o>>7)&7)<<4); return (k>>6)*8192 + o; };
        int bad = 0;
        for (int mn = 0; mn < 128; ++mn) for (int k = 0; k < 128; ++k)
            if ((int)layK(mn,k) != koffh(mn,k)) bad++;
        printf("mismatches=%d / 16384\n", bad);
    }

    // ===== EXACT FA3 V layout: SmemLayoutVt = tile_to_shape(MN_SW128, (dv,keys,stages), Step<2,1,3>) =====
    // (mainloop_fwd_sm90_tma_gmma_ws.hpp:142-145). take<0,2> drops the stage mode -> [dv=128, keys=128].
    {
        auto atomVt = GMMA::Layout_MN_SW128_Atom<T>{};
        auto SmemLayoutVt = tile_to_shape(atomVt, make_shape(Int<128>{}, Int<128>{}, Int<2>{}),
                                          Step<_2, _1, _3>{});
        auto layVt = take<0, 2>(SmemLayoutVt);  // [dv=128, keys=128]  (dv = mode-0 = MN, keys = mode-1 = K)
        printf("\n=== FA3 SmemLayoutVt take<0,2> [dv=128, keys=128] ===\n");
        printf("layout="); cute::print(layVt); printf("\n");
        printf("cosize=%d\n", (int)cosize(layVt));
        printf("Vt dv-rows 0..3 at keys 0..7 (offsets):\n");
        for (int dv = 0; dv < 4; ++dv) {
            printf("dv=%3d:", dv);
            for (int k = 0; k < 8; ++k) printf(" %5d", (int)layVt(dv, k));
            printf("\n");
        }
        printf("Vt dv-block boundary (dv 62..66) at keys 0..3:\n");
        for (int dv = 62; dv <= 66; ++dv) {
            printf("dv=%3d:", dv);
            for (int k = 0; k < 4; ++k) printf(" %5d", (int)layVt(dv, k));
            printf("\n");
        }
        printf("Vt keys-step starts (dv=0): ");
        for (int kb = 0; kb < 8; ++kb) printf(" %d", (int)layVt(0, kb * 16));
        printf("\n");
        printf("Vt dv-step starts (keys=0): ");
        for (int db = 0; db < 8; ++db) printf(" %d", (int)layVt(db * 16, 0));
        printf("\n");
        // descriptor on the canonical K=16 slice
        auto layVt16 = composition(layVt, make_tile(Layout<_128>{}, Layout<_16>{}));
        auto tVt16 = make_tensor(make_smem_ptr((T*)nullptr), layVt16);
        printf("--- make_gmma_desc<Major::MN>(SmemLayoutVt [128,16]) ---\n");
        cute::print(SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::MN>(tVt16));
        printf("\n");
    }
    return 0;
}
