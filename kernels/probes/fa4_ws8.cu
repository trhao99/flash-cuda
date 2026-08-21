// fa4_ws8.cu — q_stage=2 WS pipeline with a TRUE 1-PASS register-fragment softmax (replicates FA4 ref).
//
// vs fa4_ws5 (2-pass, re-reads S from tmem): this loads all 128 S values of the row into a live
// register fragment via 4x tcgen05.ld.32x32b.x32, computes the block max, then exps + writes P from
// the SAME live fragment (zero re-read) — exactly what the reference CuteDSL PTX does (REG:128
// baseline, STACK:0, no spill). The key to avoiding the local-memory spill that killed earlier .x32
// attempts: the 32-output asm is inlined DIRECTLY into named array slots with compile-time-constant
// indices (no &f[..] address-taking, no helper-pointer), and all reads use __uint_as_float (no memcpy
// address-taking), so nvcc's SROA splits f[] into 128 scalars that ptxas places in the 192-reg window.
//
// Build: nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 fa4_ws8.cu -o /tmp/ws8 -lcuda
#include <cuda.h>
#include "probe_common.cuh"

__device__ __forceinline__ uint32_t smem_u32(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void tmem_alloc(uint32_t s,int n){ asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(s),"r"(n)); }
__device__ __forceinline__ void tmem_relinquish(){ asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int n){ asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(n)); }
__device__ __forceinline__ uint32_t tmem_ld(uint32_t a){ uint32_t v; asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];":"=r"(v):"r"(a)); return v; }
__device__ __forceinline__ void tmem_st(uint32_t a, uint32_t v){ asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"::"r"(a),"r"(v)); }
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;"); }
__device__ __forceinline__ void tmem_wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;"); }
__device__ __forceinline__ void tmem_fence(){ asm volatile("tcgen05.fence::after_thread_sync;"); }
__device__ __forceinline__ float ex2f(float x){ float r; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(r):"f"(x)); return r; }
__device__ __forceinline__ int koff(int r,int c){ return (c>>6)*8192 + r*64 + ((c&63)^((r&7)<<3)); }
__device__ __forceinline__ uint64_t make_desc(uint32_t a,uint32_t lbo,uint32_t sbo,int lt){
    uint64_t d=0; d|=(uint64_t)((a>>4)&0x3FC0u); d|=(uint64_t)((lbo>>4)&0x3FFFu)<<16;
    d|=(uint64_t)((sbo>>4)&0x3FFFu)<<32; d|=(uint64_t)1u<<46; d|=(uint64_t)(lt&0x7u)<<61; return d; }
__device__ __forceinline__ uint64_t desc_add(uint64_t b,uint32_t off){
    uint32_t lo,hi; asm("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"l"(b)); lo+=off;
    uint64_t r; asm("mov.b64 %0,{%1,%2};":"=l"(r):"r"(lo),"r"(hi)); return r; }
__device__ __forceinline__ uint32_t qk_doff(int kb){ return (uint32_t)((kb&3)*2 + (kb>>2)*1024); }
__device__ __forceinline__ uint32_t v_doff(int kb){ return (uint32_t)(kb*0x80); }
__device__ __forceinline__ uint32_t pack_bf16(float a,float b){
    __nv_bfloat16 x=__float2bfloat16_rn(a),y=__float2bfloat16_rn(b); uint16_t ux,uy; memcpy(&ux,&x,2); memcpy(&uy,&y,2); return uint32_t(ux)|(uint32_t(uy)<<16); }
__device__ __forceinline__ void mbar_init(uint32_t m,int c){ asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"(m),"r"(c)); }
__device__ __forceinline__ void mbar_arrive(uint32_t m){ asm volatile("mbarrier.arrive.shared.b64 _, [%0];"::"r"(m)); }
__device__ __forceinline__ void mbar_expect(uint32_t m,int bytes){ asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"::"r"(m),"r"(bytes)); }
__device__ __forceinline__ void mbar_wait(uint32_t m,int ph){ asm volatile("{ .reg .pred P;\nLW: mbarrier.try_wait.parity.shared.b64 P,[%0],%1,10000000;\n @P bra D; bra LW;\nD: }"::"r"(m),"r"(ph)); }
__device__ __forceinline__ void mma_commit(uint32_t m){ asm volatile("{ .reg .pred l; elect.sync _|l,-1;\n @l tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]; }"::"r"(m)); }
__device__ __forceinline__ void tma_box(void* dst,const CUtensorMap* map,int x,int y,uint32_t bar){
    uint32_t s=(uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];"
                 :: "r"(s),"l"(map),"r"(x),"r"(y),"r"(bar) : "memory"); }

// Load 32 tmem cols directly into f[OFF..OFF+31] (compile-time-const slots -> SROA -> registers, no spill).
#define LD32_INTO(ADDR, F, OFF) \
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" \
        : "=r"(F[OFF+0]),"=r"(F[OFF+1]),"=r"(F[OFF+2]),"=r"(F[OFF+3]),"=r"(F[OFF+4]),"=r"(F[OFF+5]),"=r"(F[OFF+6]),"=r"(F[OFF+7]), \
          "=r"(F[OFF+8]),"=r"(F[OFF+9]),"=r"(F[OFF+10]),"=r"(F[OFF+11]),"=r"(F[OFF+12]),"=r"(F[OFF+13]),"=r"(F[OFF+14]),"=r"(F[OFF+15]), \
          "=r"(F[OFF+16]),"=r"(F[OFF+17]),"=r"(F[OFF+18]),"=r"(F[OFF+19]),"=r"(F[OFF+20]),"=r"(F[OFF+21]),"=r"(F[OFF+22]),"=r"(F[OFF+23]), \
          "=r"(F[OFF+24]),"=r"(F[OFF+25]),"=r"(F[OFF+26]),"=r"(F[OFF+27]),"=r"(F[OFF+28]),"=r"(F[OFF+29]),"=r"(F[OFF+30]),"=r"(F[OFF+31]) \
        : "r"(ADDR))

constexpr int M=128,N=128,K=128,VD=128,NKT=K/16, ST=2;
constexpr uint32_t QK_IDESC=0x8200490u, PV_IDESC=0x8210490u;
constexpr int TILE_BYTES = N*K*2;

struct Bars { uint64_t kv_full[ST], kv_empty[ST], s_full[2], p_full[2], o_ready[2]; };

__global__ void __launch_bounds__(512,1) ws8_kernel(const CUtensorMap* __restrict__ mapK, const CUtensorMap* __restrict__ mapV,
                                                   int NB, float scale, float* Oout, float* Lout, const __nv_bfloat16* Qg){
    extern __shared__ __nv_bfloat16 draw[];
    uint32_t rp=smem_u32(draw); uint32_t pad=((rp+1023u)&~1023u)-rp;
    __nv_bfloat16* dyn=(__nv_bfloat16*)((char*)draw+pad);
    __nv_bfloat16* sQ=dyn;
    __nv_bfloat16* sK[ST]; __nv_bfloat16* sV[ST];
    { __nv_bfloat16* pK=sQ+2*M*K; for(int s=0;s<ST;s++){ sK[s]=pK; pK+=N*K; } __nv_bfloat16* pV=pK; for(int s=0;s<ST;s++){ sV[s]=pV; pV+=N*VD; } }
    __shared__ __align__(16) Bars bars;
    __shared__ __align__(8) uint32_t s_base;
    const int tid=threadIdx.x, warp=tid>>5;
    const float LOG2E=1.4426950408889634f;
    const int SCOL[2]={0,128}, PCOL[2]={0,128}, OCOL[2]={256,384};

    if (warp>=8) asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");   // producer WGs (w8-15) release at top (aligned)

    for(int i=tid;i<2*M*K;i+=512){ int r=i/K,c=i%K; int t=r/M, lr=r%M; sQ[t*M*K + koff(lr,c)]=Qg[r*K+c]; }
    if(warp==8){ tmem_alloc(smem_u32(&s_base),512); tmem_relinquish(); }
    if(tid==0){ for(int s=0;s<ST;s++){ mbar_init(smem_u32(&bars.kv_full[s]),1); mbar_init(smem_u32(&bars.kv_empty[s]),1);}
                for(int b=0;b<2;b++){ mbar_init(smem_u32(&bars.s_full[b]),1); mbar_init(smem_u32(&bars.p_full[b]),1); mbar_init(smem_u32(&bars.o_ready[b]),1);} }
    __syncthreads();
    const uint32_t base=s_base;
    uint32_t KVF[ST],KVE[ST]; for(int s=0;s<ST;s++){ KVF[s]=smem_u32(&bars.kv_full[s]); KVE[s]=smem_u32(&bars.kv_empty[s]); }
    uint32_t SF[2]={smem_u32(&bars.s_full[0]),smem_u32(&bars.s_full[1])};
    uint32_t PF[2]={smem_u32(&bars.p_full[0]),smem_u32(&bars.p_full[1])};
    uint32_t OR[2]={smem_u32(&bars.o_ready[0]),smem_u32(&bars.o_ready[1])};

    if (warp==9) {
        int pe[ST]={0}; bool lead=(tid&31)==0;
        for(int i=0;i<NB;++i){ int s=i%ST;
            mbar_wait(KVE[s], pe[s]); pe[s]^=1;
            if(lead){ uint32_t bar=KVF[s]; mbar_expect(bar, 2*TILE_BYTES);
                tma_box(sK[s]+0,mapK,0,i*N,bar); tma_box(sK[s]+8192,mapK,64,i*N,bar);
                tma_box(sV[s]+0,mapV,0,i*N,bar); tma_box(sV[s]+8192,mapV,64,i*N,bar); } }
    } else if (warp==8) {
        int pf[ST]={0}, ppf[2]={0,0};
        auto qk=[&](int j,int i){ int s=i%ST; uint64_t qd0=make_desc(smem_u32(sQ+j*M*K),16,1024,2),kd0=make_desc(smem_u32(sK[s]),16,1024,2);
            for(int kb=0;kb<NKT;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb)); uint32_t pr=(kb>0)?1u:0u;
                asm volatile("{ .reg .pred l,p; elect.sync _|l,-1; setp.ne.b32 p,%3,0;\n @l tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                    ::"r"(base+SCOL[j]),"l"(qd),"l"(kd),"r"(pr),"r"(QK_IDESC)); }
            mma_commit(SF[j]); };
        auto pv=[&](int j,int i){ int s=i%ST; uint64_t vd0=make_desc(smem_u32(sV[s]),16384,1024,2);
            for(int kb=0;kb<NKT;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pr=(i>0||kb>0)?1u:0u; uint32_t pa=base+PCOL[j]+(uint32_t)(kb*8);
                asm volatile("{ .reg .pred l,p; elect.sync _|l,-1; setp.ne.b32 p,%3,0;\n @l tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                    ::"r"(base+OCOL[j]),"r"(pa),"l"(vd),"r"(pr),"r"(PV_IDESC)); }
            mma_commit(OR[j]); };
        mbar_wait(KVF[0], pf[0]); pf[0]^=1; qk(0,0); qk(1,0);
        for(int i=0;i<NB;++i){ int s=i%ST;
            mbar_wait(PF[0], ppf[0]); ppf[0]^=1; pv(0,i);
            mbar_wait(PF[1], ppf[1]); ppf[1]^=1; pv(1,i);
            mma_commit(KVE[s]);
            if(i+1<NB){ int ns=(i+1)%ST; mbar_wait(KVF[ns], pf[ns]); pf[ns]^=1; qk(0,i+1); qk(1,i+1); }
        }
    } else if (warp<8) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");   // softmax WGs (w0-7) acquire IN-BRANCH, adjacent to the reg-heavy softmax code
        // ---- softmax WG0/WG1: TRUE 1-pass register-fragment softmax ----
        const int j = warp>>2;
        const int lr = tid & 127;
        int psf=0, por=0;
        float m_i=-INFINITY, l_i=0.f;
        if(tid==0){ for(int s=0;s<ST;s++) mbar_arrive(KVE[s]); }
        for(int i=0;i<NB;++i){
            mbar_wait(SF[j], psf); psf^=1; tmem_fence();
            uint32_t a0 = base+SCOL[j]+((uint32_t)lr<<16);
            uint32_t f[128];
            LD32_INTO(a0+0,  f, 0);
            LD32_INTO(a0+32, f, 32);
            LD32_INTO(a0+64, f, 64);
            LD32_INTO(a0+96, f, 96);
            tmem_wait_ld();
            // block max over raw S*scale (fragment stays live)
            float rmax=-INFINITY;
            #pragma unroll
            for(int t=0;t<128;++t){ rmax=fmaxf(rmax, __uint_as_float(f[t])*scale); }
            float m_new=fmaxf(m_i,rmax); float corr=(m_i==-INFINITY)?1.f:ex2f((m_i-m_new)*LOG2E);
            // exp + P-write from the SAME live fragment (no re-read); P aliases S lower half, safe (S fully consumed)
            float ps=0.f;
            #pragma unroll
            for(int t=0;t<64;++t){ float s0=__uint_as_float(f[2*t])*scale, s1=__uint_as_float(f[2*t+1])*scale;
                float p0=ex2f((s0-m_new)*LOG2E), p1=ex2f((s1-m_new)*LOG2E); ps+=p0+p1;
                tmem_st(base+PCOL[j]+((uint32_t)lr<<16)+t, pack_bf16(p0,p1)); }
            // O-rescale AFTER P-write (fragment f[] now dead -> frees 128 regs before this loop's temps)
#ifndef LEAN
            if(i>0){ mbar_wait(OR[j], por); por^=1;
                for(int d=0;d<VD;++d){ uint32_t a=base+OCOL[j]+((uint32_t)lr<<16)+d; uint32_t w=tmem_ld(a); float o=__uint_as_float(w); o*=corr; tmem_st(a,__float_as_uint(o));} }
#endif
            l_i=l_i*corr+ps; m_i=m_new;
            tmem_wait_st();
            asm volatile("bar.sync %0, 128;"::"r"(1+j));
            if((tid&127)==0) mbar_arrive(PF[j]);
        }
#ifndef LEAN
        { mbar_wait(OR[j], por); por^=1; } tmem_fence();
        for(int d=0;d<VD;++d){ uint32_t w=tmem_ld(base+OCOL[j]+((uint32_t)lr<<16)+d); tmem_wait_ld(); float o=__uint_as_float(w); Oout[(j*M+lr)*VD+d]=o/l_i; }
        if(Lout) Lout[j*M+lr]=logf(l_i)+m_i;
#else
        if(Lout) Lout[j*M+lr]=l_i;   // stub to keep l_i live
#endif
    }
    __syncthreads();
    if(warp==8) tmem_dealloc(base,512);
}

static CUtensorMap make_map(const __nv_bfloat16* d, int rows, int cols){
    CUtensorMap map{};
    uint64_t gdim[2]={(uint64_t)cols,(uint64_t)rows}; uint64_t gstride[1]={(uint64_t)(cols*2)};
    uint32_t bdim[2]={64u,(uint32_t)N}; uint32_t estride[2]={1u,1u};
    CUresult cr=cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)d,
        gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(cr!=CUDA_SUCCESS){ std::printf("encode FAILED %d\n",(int)cr); std::exit(1);} return map;
}

int main(int argc,char**argv){
    int NB=(argc>1)?atoi(argv[1]):8;
    ORC_CHECK(cudaSetDevice(0));
    cudaDeviceProp p{}; ORC_CHECK(cudaGetDeviceProperties(&p,0));
    std::printf("device: %s NB=%d (2 Q-tiles/CTA, 1-pass)\n",p.name,NB);
    int TK=NB*N, MM=2*M;
    std::vector<uint16_t> Qh,Kh,Vh; orc::fill_bf16(Qh,MM*K,1); orc::fill_bf16(Kh,TK*K,2); orc::fill_bf16(Vh,TK*VD,3);
    __nv_bfloat16 *dQ,*dK,*dV; float *dO,*dL;
    ORC_CHECK(cudaMalloc(&dQ,MM*K*2)); ORC_CHECK(cudaMalloc(&dK,TK*K*2)); ORC_CHECK(cudaMalloc(&dV,TK*VD*2));
    ORC_CHECK(cudaMalloc(&dO,MM*VD*4)); ORC_CHECK(cudaMalloc(&dL,MM*4));
    ORC_CHECK(cudaMemcpy(dQ,Qh.data(),MM*K*2,cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dK,Kh.data(),TK*K*2,cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dV,Vh.data(),TK*VD*2,cudaMemcpyHostToDevice));
    CUtensorMap mapK=make_map(dK,TK,K), mapV=make_map(dV,TK,VD);
    CUtensorMap *dMK,*dMV; ORC_CHECK(cudaMalloc(&dMK,sizeof(CUtensorMap))); ORC_CHECK(cudaMalloc(&dMV,sizeof(CUtensorMap)));
    ORC_CHECK(cudaMemcpy(dMK,&mapK,sizeof(CUtensorMap),cudaMemcpyHostToDevice)); ORC_CHECK(cudaMemcpy(dMV,&mapV,sizeof(CUtensorMap),cudaMemcpyHostToDevice));
    float scale=1.0f/std::sqrt((float)K);
    size_t smem=(size_t)(2*M*K + ST*N*K + ST*N*VD)*2 + 1024;
    ORC_CHECK(cudaFuncSetAttribute(ws8_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    for(int w=0;w<20;w++) ws8_kernel<<<1,512,smem>>>(dMK,dMV,NB,scale,dO,dL,dQ);
    ORC_CHECK(cudaDeviceSynchronize());
    float ms=orc::benchmark_ms([&](){ ws8_kernel<<<1,512,smem>>>(dMK,dMV,NB,scale,dO,dL,dQ); },20,50,5,nullptr,0);
    ORC_CHECK(cudaGetLastError()); ORC_CHECK(cudaDeviceSynchronize());
    std::vector<float> O(MM*VD); ORC_CHECK(cudaMemcpy(O.data(),dO,MM*VD*4,cudaMemcpyDeviceToHost));
    double max_abs=0; int shown=0;
    for(int m=0;m<MM;++m){ std::vector<double> sc(TK); double mx=-1e30;
        for(int n=0;n<TK;++n){ double a=0; for(int k=0;k<K;++k) a+=(double)orc::bf16_bits_to_f32(Qh[m*K+k])*(double)orc::bf16_bits_to_f32(Kh[n*K+k]); sc[n]=a*scale; mx=std::max(mx,sc[n]); }
        double l=0; for(int n=0;n<TK;++n){ sc[n]=std::exp(sc[n]-mx); l+=sc[n]; }
        for(int d=0;d<VD;++d){ double o=0; for(int n=0;n<TK;++n) o+=sc[n]*(double)orc::bf16_bits_to_f32(Vh[n*VD+d]); o/=l;
            double e=std::fabs(O[m*VD+d]-o); if(e>max_abs)max_abs=e; if(e>2e-2&&shown<6){std::printf("  bad m=%d d=%d got=%.4f ref=%.4f\n",m,d,O[m*VD+d],o);shown++;} } }
    std::printf("max_abs=%.5f  %s   time=%.1f us  (per-Qtile %.1f us)\n",max_abs, max_abs<2e-2?"PASS":"FAIL", ms*1e3, ms*1e3/2);
    return max_abs<2e-2?0:1;
}
