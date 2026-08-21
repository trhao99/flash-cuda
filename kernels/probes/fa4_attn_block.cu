// fa4_attn_block.cu — Blackwell SM100 single-KV-block attention core: O = softmax(Q·Kᵀ·scale)·V.
//
// Validates the entire FA4 compute inner loop end-to-end (no KV streaming / masking / TMA epilogue):
//   QK GEMM (SS, tcgen05) -> S in tmem -> per-row online softmax (ex2.approx, num_rows=1) ->
//   P (bf16) written back to tmem -> PV GEMM (TS, A=P from tmem, B=V MN-major smem) -> O in tmem.
//
// All descriptor constants confirmed against the compiled FA4 reference PTX:
//   QK idesc 0x8200490, PV idesc 0x8210490, hi 0x40004040 (128B swizzle);
//   Q/K lo LBO=16 (field 1), V lo LBO=16384 (field 1024, =reference |0x4000000);
//   QK k-advance {0,2,4,6,0x400..}; PV: P(tmem) +8 cols/tile, V desc +0x80/tile.
//
// Build: nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 fa4_attn_block.cu -o /tmp/ab
#include "probe_common.cuh"

__device__ __forceinline__ uint32_t smem_u32(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void tmem_alloc(uint32_t s,int n){ asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(s),"r"(n)); }
__device__ __forceinline__ void tmem_relinquish(){ asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int n){ asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(n)); }
__device__ __forceinline__ uint32_t tmem_ld_1(uint32_t a){ uint32_t v; asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];":"=r"(v):"r"(a)); return v; }
__device__ __forceinline__ void tmem_st_1(uint32_t a, uint32_t v){ asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"::"r"(a),"r"(v)); }
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;"); }
__device__ __forceinline__ void tmem_wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;"); }
__device__ __forceinline__ float ex2f(float x){ float r; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(r):"f"(x)); return r; }

// 128B swizzle: koff = K-major (Q,K), voff = MN-major (V). Both ROWS=128.
__device__ __forceinline__ int koff(int r,int c){ return (c>>6)*8192 + r*64 + ((c&63)^((r&7)<<3)); }
__device__ __forceinline__ int voff(int key,int dv){ return (dv>>6)*8192 + key*64 + ((dv&63)^((key&7)<<3)); }

__device__ __forceinline__ uint64_t make_desc(uint32_t addr,uint32_t lbo,uint32_t sbo,int lt){
    uint64_t d=0;
    d |= (uint64_t)((addr>>4)&0x3FC0u);
    d |= (uint64_t)((lbo>>4)&0x3FFFu)<<16;
    d |= (uint64_t)((sbo>>4)&0x3FFFu)<<32;
    d |= (uint64_t)1u<<46;
    d |= (uint64_t)(lt&0x7u)<<61;
    return d;
}
__device__ __forceinline__ uint64_t desc_add(uint64_t b,uint32_t off){
    uint32_t lo,hi; asm("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"l"(b));
    lo+=off; uint64_t r; asm("mov.b64 %0,{%1,%2};":"=l"(r):"r"(lo),"r"(hi)); return r;
}
__device__ __forceinline__ uint32_t qk_doff(int kb){ return (uint32_t)((kb&3)*2 + (kb>>2)*1024); }
__device__ __forceinline__ uint32_t v_doff(int kb){ return (uint32_t)(kb*0x80); }
__device__ __forceinline__ uint32_t pack_bf16(float a,float b){
    __nv_bfloat16 x=__float2bfloat16_rn(a),y=__float2bfloat16_rn(b);
    uint16_t ux,uy; memcpy(&ux,&x,2); memcpy(&uy,&y,2); return uint32_t(ux)|(uint32_t(uy)<<16);
}

constexpr int M=128,N=128,K=128,VD=128,NKT=K/16;
constexpr uint32_t QK_IDESC=0x8200490u, PV_IDESC=0x8210490u;
constexpr int S_COL=0, P_COL=128, O_COL=256;   // tmem column offsets

__global__ void __launch_bounds__(128) attn_kernel(const __nv_bfloat16* Qg,const __nv_bfloat16* Kg,
                                                   const __nv_bfloat16* Vg, float scale, float* Oout){
    __shared__ __align__(1024) __nv_bfloat16 sQ[M*K], sK[N*K], sV[N*VD];
    __shared__ __align__(8) uint32_t s_base;
    __shared__ __align__(8) uint64_t s_mbar;
    const int tid=threadIdx.x, warp=tid>>5;

    for(int i=tid;i<M*K;i+=128){ int r=i/K,c=i%K; sQ[koff(r,c)]=Qg[r*K+c]; }
    for(int i=tid;i<N*K;i+=128){ int r=i/K,c=i%K; sK[koff(r,c)]=Kg[r*K+c]; }
    for(int i=tid;i<N*VD;i+=128){ int key=i/VD,dv=i%VD; sV[voff(key,dv)]=Vg[key*VD+dv]; }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();

    if(warp==0){ tmem_alloc(smem_u32(&s_base),512); tmem_relinquish(); }
    if(tid==0) asm volatile("mbarrier.init.shared.b64 [%0], 1;"::"r"(smem_u32(&s_mbar)));
    __syncthreads();
    const uint32_t base=s_base, mbar=smem_u32(&s_mbar);

    // ---- QK: S = Q·Kᵀ  (S at tmem col 0) ----
    if(warp==0){
        uint64_t qd0=make_desc(smem_u32(sQ),16,1024,2), kd0=make_desc(smem_u32(sK),16,1024,2);
        for(int kb=0;kb<NKT;++kb){
            uint64_t qd=desc_add(qd0,qk_doff(kb)), kd=desc_add(kd0,qk_doff(kb));
            uint32_t pr=(kb>0)?1u:0u;
            asm volatile("{ .reg .pred lead,p; elect.sync _|lead,-1; setp.ne.b32 p,%3,0;\n"
                "  @lead tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                ::"r"(base+S_COL),"l"(qd),"l"(kd),"r"(pr),"r"(QK_IDESC));
        }
        asm volatile("{ .reg .pred lead; elect.sync _|lead,-1;\n"
            "  @lead tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]; }"::"r"(mbar));
    }
    asm volatile("{ .reg .pred P;\nLW1: mbarrier.try_wait.parity.shared.b64 P,[%0],0,10000000;\n @P bra D1; bra LW1;\nD1: }"::"r"(mbar));
    __syncthreads();

    // ---- softmax over row tid: rowmax, P=ex2((S-max)*scale_log2), rowsum ----
    const float LOG2E=1.4426950408889634f;
    float slog2 = scale*LOG2E;
    float s[N];
    float m=-INFINITY;
    for(int c=0;c<N;++c){ uint32_t b=tmem_ld_1(base+S_COL+((uint32_t)tid<<16)+c); tmem_wait_ld(); memcpy(&s[c],&b,4); m=fmaxf(m,s[c]); }
    float l=0.f;
    for(int c=0;c<N;++c){ float p=ex2f((s[c]-m)*slog2); s[c]=p; l+=p; }
    // store P (bf16, packed 2/col) to tmem col P_COL..P_COL+63
    for(int c=0;c<N/2;++c){ tmem_st_1(base+P_COL+((uint32_t)tid<<16)+c, pack_bf16(s[2*c],s[2*c+1])); }
    tmem_wait_st();
    if(tid==0) asm volatile("mbarrier.init.shared.b64 [%0], 1;"::"r"(mbar));  // reset for PV
    __syncthreads();

    // ---- PV: O = P·V  (A=P from tmem col P_COL, B=V MN-major smem; O at tmem col O_COL) ----
    if(warp==0){
        uint64_t vd0=make_desc(smem_u32(sV),16384,1024,2);  // MN-major: LBO field=1024
        for(int kb=0;kb<NKT;++kb){
            uint64_t vd=desc_add(vd0,v_doff(kb));
            uint32_t pr=(kb>0)?1u:0u;
            uint32_t pa=base+P_COL+(uint32_t)(kb*8);   // P tmem addr advances 8 cols/tile
            asm volatile("{ .reg .pred lead,p; elect.sync _|lead,-1; setp.ne.b32 p,%3,0;\n"
                "  @lead tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                ::"r"(base+O_COL),"r"(pa),"l"(vd),"r"(pr),"r"(PV_IDESC));
        }
        asm volatile("{ .reg .pred lead; elect.sync _|lead,-1;\n"
            "  @lead tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]; }"::"r"(mbar));
    }
    asm volatile("{ .reg .pred P;\nLW2: mbarrier.try_wait.parity.shared.b64 P,[%0],0,10000000;\n @P bra D2; bra LW2;\nD2: }"::"r"(mbar));
    __syncthreads();

    // ---- read O[tid][:] / rowsum, output ----
    for(int d=0;d<VD;++d){ uint32_t b=tmem_ld_1(base+O_COL+((uint32_t)tid<<16)+d); tmem_wait_ld(); float o; memcpy(&o,&b,4); Oout[tid*VD+d]=o/l; }
    __syncthreads();
    if(warp==0) tmem_dealloc(base,512);
}

int main(){
    ORC_CHECK(cudaSetDevice(0));
    cudaDeviceProp p{}; ORC_CHECK(cudaGetDeviceProperties(&p,0));
    std::printf("device: %s cc %d.%d\n",p.name,p.major,p.minor);
    std::vector<uint16_t> Qh,Kh,Vh; orc::fill_bf16(Qh,M*K,1); orc::fill_bf16(Kh,N*K,2); orc::fill_bf16(Vh,N*VD,3);
    __nv_bfloat16 *dQ,*dK,*dV; float* dO;
    ORC_CHECK(cudaMalloc(&dQ,M*K*2)); ORC_CHECK(cudaMalloc(&dK,N*K*2)); ORC_CHECK(cudaMalloc(&dV,N*VD*2)); ORC_CHECK(cudaMalloc(&dO,M*VD*4));
    ORC_CHECK(cudaMemcpy(dQ,Qh.data(),M*K*2,cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dK,Kh.data(),N*K*2,cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dV,Vh.data(),N*VD*2,cudaMemcpyHostToDevice));
    float scale=1.0f/std::sqrt((float)K);
    attn_kernel<<<1,128>>>(dQ,dK,dV,scale,dO);
    ORC_CHECK(cudaGetLastError()); ORC_CHECK(cudaDeviceSynchronize());
    std::vector<float> O(M*VD); ORC_CHECK(cudaMemcpy(O.data(),dO,M*VD*4,cudaMemcpyDeviceToHost));

    // fp32 reference: softmax(scale * Q·Kᵀ) · V (single block, full)
    double max_abs=0; int shown=0;
    for(int m=0;m<M;++m){
        std::vector<double> sc(N); double mx=-1e30;
        for(int n=0;n<N;++n){ double a=0; for(int k=0;k<K;++k) a+=(double)orc::bf16_bits_to_f32(Qh[m*K+k])*(double)orc::bf16_bits_to_f32(Kh[n*K+k]); sc[n]=a*scale; mx=std::max(mx,sc[n]); }
        double l=0; for(int n=0;n<N;++n){ sc[n]=std::exp(sc[n]-mx); l+=sc[n]; }
        for(int d=0;d<VD;++d){ double o=0; for(int n=0;n<N;++n) o+=sc[n]*(double)orc::bf16_bits_to_f32(Vh[n*VD+d]); o/=l;
            double err=std::fabs(O[m*VD+d]-o); if(err>max_abs) max_abs=err;
            if(err>2e-2 && shown<8){ std::printf("  bad m=%d d=%d got=%.4f ref=%.4f\n",m,d,O[m*VD+d],o); shown++; } }
    }
    std::printf("max_abs=%.5f\n",max_abs);
    bool pass = max_abs < 2e-2;
    std::printf("%s: single-block attention (QK -> softmax(ex2) -> PV) core\n",pass?"PASS":"FAIL");
    ORC_CHECK(cudaFree(dQ));ORC_CHECK(cudaFree(dK));ORC_CHECK(cudaFree(dV));ORC_CHECK(cudaFree(dO));
    return pass?0:1;
}
