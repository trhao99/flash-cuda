// fa4_ws2.cu — Blackwell-native warp-specialized attention pipeline (perf prototype, q_stage=1).
//
// Follows the PTX-confirmed reference blueprint (flash_fwd_sm100): dedicated roles, no __syncthreads
// between them (mbarriers only), so load/MMA/softmax overlap. Key vs the earlier WS probe: the LOAD
// is a dedicated warp doing TMA (cp.async.bulk.tensor, hardware swizzle, async) — NOT the MMA warp
// doing 32-thread stores (which serialized it). softmax warpgroup gets setmaxnreg 192 (no spill).
//
// Roles (512 threads): w0-3 = softmax WG (read S -> online softmax -> write P, rescale O, epilogue),
// w4 = MMA warp (QK ahead + PV, owns tmem), w5 = load warp (TMA K/V into 3-deep ring). Others idle.
// tmem: S0@0 S1@128 (double-buf, P aliased @+64), O@256.  KV ring depth = 3.
//
// Build: nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 fa4_ws2.cu -o /tmp/ws2 -lcuda
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
__device__ __forceinline__ void reg_inc(int n){ asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;"::"n"(192)); (void)n; }

constexpr int M=128,N=128,K=128,VD=128,NKT=K/16, ST=3;   // ST = KV ring depth
constexpr uint32_t QK_IDESC=0x8200490u, PV_IDESC=0x8210490u;
constexpr int O_COL=256;
constexpr int TILE_BYTES = N*K*2;   // bytes per K (or V) tile

struct Bars { uint64_t kv_full[ST], kv_empty[ST], s_full[2], s_empty[2], p_full[2], o_ready[2]; };

__global__ void __launch_bounds__(512,1) ws2_kernel(const CUtensorMap* __restrict__ mapK, const CUtensorMap* __restrict__ mapV,
                                                   int NB, float scale, float* Oout, float* Lout, const __nv_bfloat16* Qg){
    extern __shared__ __nv_bfloat16 draw[];
    uint32_t rp=smem_u32(draw); uint32_t pad=((rp+1023u)&~1023u)-rp;
    __nv_bfloat16* dyn=(__nv_bfloat16*)((char*)draw+pad);
    __nv_bfloat16* sQ=dyn;
    __nv_bfloat16* sK[ST]; __nv_bfloat16* sV[ST];
    { __nv_bfloat16* pK=sQ+M*K; for(int s=0;s<ST;s++){ sK[s]=pK; pK+=N*K; } __nv_bfloat16* pV=pK; for(int s=0;s<ST;s++){ sV[s]=pV; pV+=N*VD; } }
    __shared__ __align__(16) Bars bars;
    __shared__ __align__(8) uint32_t s_base;
    const int tid=threadIdx.x, warp=tid>>5;
    const float LOG2E=1.4426950408889634f;
    const int SCOL[2]={0,128}, PCOL[2]={384,448};

    // per-role register partition (softmax WG gets 192; others drop to 56). Warpgroup-aligned.
    if (warp<4) asm volatile("setmaxnreg.inc.sync.aligned.u32 128;");
    else        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");

    for(int i=tid;i<M*K;i+=512){ int r=i/K,c=i%K; sQ[koff(r,c)]=Qg[r*K+c]; }
    if(warp==4){ tmem_alloc(smem_u32(&s_base),512); tmem_relinquish(); }
    if(tid==0){ for(int s=0;s<ST;s++){ mbar_init(smem_u32(&bars.kv_full[s]),1); mbar_init(smem_u32(&bars.kv_empty[s]),1);}
                for(int b=0;b<2;b++){ mbar_init(smem_u32(&bars.s_full[b]),1); mbar_init(smem_u32(&bars.s_empty[b]),1);
                                      mbar_init(smem_u32(&bars.p_full[b]),1); mbar_init(smem_u32(&bars.o_ready[b]),1);} }
    __syncthreads();
    const uint32_t base=s_base;
    uint32_t KVF[ST],KVE[ST]; for(int s=0;s<ST;s++){ KVF[s]=smem_u32(&bars.kv_full[s]); KVE[s]=smem_u32(&bars.kv_empty[s]); }
    uint32_t SF[2]={smem_u32(&bars.s_full[0]),smem_u32(&bars.s_full[1])};
    uint32_t SE[2]={smem_u32(&bars.s_empty[0]),smem_u32(&bars.s_empty[1])};
    uint32_t PF[2]={smem_u32(&bars.p_full[0]),smem_u32(&bars.p_full[1])};
    uint32_t OR[2]={smem_u32(&bars.o_ready[0]),smem_u32(&bars.o_ready[1])};

    if (warp==5) {
        // ---- LOAD warp: TMA-stream K/V into the ST-deep ring ----
        int pe[ST]={0}; for(int s=0;s<ST;s++) pe[s]=0;
        bool lead = (tid&31)==0;
        // kv_empty pre-armed by consumer? no — arm here initially so first ST loads proceed.
        for(int i=0;i<NB;++i){ int s=i%ST;
            mbar_wait(KVE[s], pe[s]); pe[s]^=1;                 // buffer s free
            if(lead){
                uint32_t bar=KVF[s];
                mbar_expect(bar, 2*TILE_BYTES);                  // K tile + V tile bytes
                tma_box(sK[s]+0,    mapK, 0,  i*N, bar); tma_box(sK[s]+8192, mapK, 64, i*N, bar);
                tma_box(sV[s]+0,    mapV, 0,  i*N, bar); tma_box(sV[s]+8192, mapV, 64, i*N, bar);
            }
        }
    } else if (warp==4) {
        // ---- MMA warp: QK ahead + PV ----
        int pf[ST]={0}; int psf_e[2]={0,0}, ppf[2]={0,0};
        auto issue_qk=[&](int i){ int b=i&1; int s=i%ST; uint64_t qd0=make_desc(smem_u32(sQ),16,1024,2),kd0=make_desc(smem_u32(sK[s]),16,1024,2);
            for(int kb=0;kb<NKT;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb)); uint32_t pr=(kb>0)?1u:0u;
                asm volatile("{ .reg .pred l,p; elect.sync _|l,-1; setp.ne.b32 p,%3,0;\n @l tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                    ::"r"(base+SCOL[b]),"l"(qd),"l"(kd),"r"(pr),"r"(QK_IDESC)); }
            mma_commit(SF[b]); };
        auto issue_pv=[&](int i){ int b=i&1; int s=i%ST; uint64_t vd0=make_desc(smem_u32(sV[s]),16384,1024,2);
            for(int kb=0;kb<NKT;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pr=(i>0||kb>0)?1u:0u; uint32_t pa=base+PCOL[b]+(uint32_t)(kb*8);
                asm volatile("{ .reg .pred l,p; elect.sync _|l,-1; setp.ne.b32 p,%3,0;\n @l tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                    ::"r"(base+O_COL),"r"(pa),"l"(vd),"r"(pr),"r"(PV_IDESC)); }
            mma_commit(OR[b]); };
        // prologue: wait KV0, wait S buf0 free, QK0
        mbar_wait(KVF[0], pf[0]); pf[0]^=1;
        mbar_wait(SE[0], psf_e[0]); psf_e[0]^=1; issue_qk(0);
        for(int i=0;i<NB;++i){ int s=i%ST;
            if(i+1<NB){ int ns=(i+1)%ST; int nb=(i+1)&1; mbar_wait(KVF[ns], pf[ns]); pf[ns]^=1; mbar_wait(SE[nb], psf_e[nb]); psf_e[nb]^=1; issue_qk(i+1); }
            mbar_wait(PF[i&1], ppf[i&1]); ppf[i&1]^=1;   // consumer wrote P + rescaled O
            issue_pv(i);                                  // commits OR[i&1]
            mma_commit(KVE[s]);                            // K/V[s] consumed -> load may refill
        }
    } else if (warp<4) {
        // ---- softmax WG (192 regs): softmax(S)->P, rescale O, epilogue ----
        int psf[2]={0,0}, por[2]={0,0};
        float m_i=-INFINITY, l_i=0.f; float sreg[N];
        if(tid==0){ mbar_arrive(SE[0]); mbar_arrive(SE[1]); for(int s=0;s<ST;s++) mbar_arrive(KVE[s]); }  // buffers initially free
        for(int i=0;i<NB;++i){ int b=i&1;
            mbar_wait(SF[b], psf[b]); psf[b]^=1; tmem_fence();
            float rmax=-INFINITY;
            for(int c=0;c<N;++c){ uint32_t bb=tmem_ld(base+SCOL[b]+((uint32_t)tid<<16)+c); tmem_wait_ld(); float sv; memcpy(&sv,&bb,4); sreg[c]=sv; rmax=fmaxf(rmax,sv*scale); }
            asm volatile("bar.sync 1, 128;");
            if((tid&127)==0) mbar_arrive(SE[b]);
            float m_new=fmaxf(m_i,rmax); float corr=(m_i==-INFINITY)?1.f:ex2f((m_i-m_new)*LOG2E); float ps=0.f;
            for(int c=0;c<N;++c){ float p=ex2f(sreg[c]*scale-m_new); sreg[c]=p; ps+=p; }
            l_i=l_i*corr+ps; m_i=m_new;
            for(int c=0;c<N/2;++c) tmem_st(base+PCOL[b]+((uint32_t)tid<<16)+c, pack_bf16(sreg[2*c],sreg[2*c+1]));
            if(i>0){ int pb=(i-1)&1; mbar_wait(OR[pb], por[pb]); por[pb]^=1;
                for(int d=0;d<VD;++d){ uint32_t a=base+O_COL+((uint32_t)tid<<16)+d; uint32_t bb=tmem_ld(a); tmem_wait_ld(); float o; memcpy(&o,&bb,4); o*=corr; uint32_t ob; memcpy(&ob,&o,4); tmem_st(a,ob);} }
            tmem_wait_st();
            asm volatile("bar.sync 1, 128;");
            if((tid&127)==0) mbar_arrive(PF[b]);
        }
        { int pb=(NB-1)&1; mbar_wait(OR[pb], por[pb]); por[pb]^=1; } tmem_fence();
        for(int d=0;d<VD;++d){ uint32_t bb=tmem_ld(base+O_COL+((uint32_t)tid<<16)+d); tmem_wait_ld(); float o; memcpy(&o,&bb,4); Oout[tid*VD+d]=o/l_i; }
        if(Lout) Lout[tid]=logf(l_i)+m_i;
    }
    __syncthreads();
    if(warp==4) tmem_dealloc(base,512);
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
    std::printf("device: %s NB=%d\n",p.name,NB);
    int TK=NB*N;
    std::vector<uint16_t> Qh,Kh,Vh; orc::fill_bf16(Qh,M*K,1); orc::fill_bf16(Kh,TK*K,2); orc::fill_bf16(Vh,TK*VD,3);
    __nv_bfloat16 *dQ,*dK,*dV; float *dO,*dL;
    ORC_CHECK(cudaMalloc(&dQ,M*K*2)); ORC_CHECK(cudaMalloc(&dK,TK*K*2)); ORC_CHECK(cudaMalloc(&dV,TK*VD*2));
    ORC_CHECK(cudaMalloc(&dO,M*VD*4)); ORC_CHECK(cudaMalloc(&dL,M*4));
    ORC_CHECK(cudaMemcpy(dQ,Qh.data(),M*K*2,cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dK,Kh.data(),TK*K*2,cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dV,Vh.data(),TK*VD*2,cudaMemcpyHostToDevice));
    CUtensorMap mapK=make_map(dK,TK,K), mapV=make_map(dV,TK,VD);
    CUtensorMap *dMK,*dMV; ORC_CHECK(cudaMalloc(&dMK,sizeof(CUtensorMap))); ORC_CHECK(cudaMalloc(&dMV,sizeof(CUtensorMap)));
    ORC_CHECK(cudaMemcpy(dMK,&mapK,sizeof(CUtensorMap),cudaMemcpyHostToDevice)); ORC_CHECK(cudaMemcpy(dMV,&mapV,sizeof(CUtensorMap),cudaMemcpyHostToDevice));
    float scale=1.0f/std::sqrt((float)K);
    size_t smem=(size_t)(M*K + ST*N*K + ST*N*VD)*2 + 1024;
    ORC_CHECK(cudaFuncSetAttribute(ws2_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    for(int w=0;w<20;w++) ws2_kernel<<<1,512,smem>>>(dMK,dMV,NB,scale,dO,dL,dQ);
    ORC_CHECK(cudaDeviceSynchronize());
    float ms=orc::benchmark_ms([&](){ ws2_kernel<<<1,512,smem>>>(dMK,dMV,NB,scale,dO,dL,dQ); },20,50,5,nullptr,0);
    ORC_CHECK(cudaGetLastError()); ORC_CHECK(cudaDeviceSynchronize());
    std::vector<float> O(M*VD); ORC_CHECK(cudaMemcpy(O.data(),dO,M*VD*4,cudaMemcpyDeviceToHost));
    double max_abs=0; int shown=0;
    for(int m=0;m<M;++m){ std::vector<double> sc(TK); double mx=-1e30;
        for(int n=0;n<TK;++n){ double a=0; for(int k=0;k<K;++k) a+=(double)orc::bf16_bits_to_f32(Qh[m*K+k])*(double)orc::bf16_bits_to_f32(Kh[n*K+k]); sc[n]=a*scale; mx=std::max(mx,sc[n]); }
        double l=0; for(int n=0;n<TK;++n){ sc[n]=std::exp(sc[n]-mx); l+=sc[n]; }
        for(int d=0;d<VD;++d){ double o=0; for(int n=0;n<TK;++n) o+=sc[n]*(double)orc::bf16_bits_to_f32(Vh[n*VD+d]); o/=l;
            double e=std::fabs(O[m*VD+d]-o); if(e>max_abs)max_abs=e; if(e>2e-2&&shown<6){std::printf("  bad m=%d d=%d got=%.4f ref=%.4f\n",m,d,O[m*VD+d],o);shown++;} } }
    std::printf("max_abs=%.5f  %s   time=%.1f us\n",max_abs, max_abs<2e-2?"PASS":"FAIL", ms*1e3);
    return max_abs<2e-2?0:1;
}
