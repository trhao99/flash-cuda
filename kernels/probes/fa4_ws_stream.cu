// fa4_ws_stream.cu — warp-specialized, pipelined streaming attention (SM100) — perf prototype.
//
// Overlaps QK(block i+1) MMA with softmax(block i) via a dedicated MMA warp + double-buffered S/P/KV,
// coordinated purely by mbarriers (no __syncthreads between the MMA warp and the softmax warpgroup),
// to attack the NCU bottleneck (1 warp/scheduler stalling on tmem with nothing to hide it).
//
// Roles (160 threads): warps 0-3 = consumer warpgroup (softmax, tmem lanes 0-127); warp 4 = MMA warp
// (loads K/V, issues QK & PV, owns tmem). The O accumulator is a serial chain (PV[i-1] -> rescale[i]
// -> PV[i]); overlap is on the S/P path. tmem: S0@0 S1@128 O@256 P0@384 P1@448 = 512.
//
// Build: nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 fa4_ws_stream.cu -o /tmp/ws
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
__device__ __forceinline__ int voff(int key,int dv){ return (dv>>6)*8192 + key*64 + ((dv&63)^((key&7)<<3)); }
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
__device__ __forceinline__ void mbar_wait(uint32_t m,int ph){ asm volatile("{ .reg .pred P;\nLW: mbarrier.try_wait.parity.shared.b64 P,[%0],%1,10000000;\n @P bra D; bra LW;\nD: }"::"r"(m),"r"(ph)); }
__device__ __forceinline__ void mma_commit(uint32_t m){ asm volatile("{ .reg .pred l; elect.sync _|l,-1;\n @l tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]; }"::"r"(m)); }

constexpr int M=128,N=128,K=128,VD=128,NKT=K/16;
constexpr uint32_t QK_IDESC=0x8200490u, PV_IDESC=0x8210490u;
constexpr int O_COL=256;

struct Bars { uint64_t s_full[2], s_empty[2], p_full[2], o_ready[2]; };

__global__ void __launch_bounds__(160) ws_kernel(const __nv_bfloat16* Qg,const __nv_bfloat16* Kg,
                                                const __nv_bfloat16* Vg, int NB, float scale, float* Oout, float* Lout){
    extern __shared__ __nv_bfloat16 draw[];
    uint32_t rp=smem_u32(draw); uint32_t pad=((rp+1023u)&~1023u)-rp;
    __nv_bfloat16* dyn=(__nv_bfloat16*)((char*)draw+pad);
    __nv_bfloat16* sQ=dyn;
    __nv_bfloat16* sK[2]={sQ+M*K, sQ+M*K+N*K};
    __nv_bfloat16* sV[2]={sK[1]+N*K, sK[1]+N*K+N*VD};
    __shared__ __align__(16) Bars bars;
    __shared__ __align__(8) uint32_t s_base;
    const int tid=threadIdx.x, warp=tid>>5;
    const float LOG2E=1.4426950408889634f;
    const int SCOL[2]={0,128}, PCOL[2]={384,448};

    for(int i=tid;i<M*K;i+=160){ int r=i/K,c=i%K; sQ[koff(r,c)]=Qg[r*K+c]; }
    if(warp==4){ tmem_alloc(smem_u32(&s_base),512); tmem_relinquish(); }
    if(tid==0){ for(int b=0;b<2;b++){ mbar_init(smem_u32(&bars.s_full[b]),1); mbar_init(smem_u32(&bars.s_empty[b]),1);
                                       mbar_init(smem_u32(&bars.p_full[b]),1); mbar_init(smem_u32(&bars.o_ready[b]),1); } }
    __syncthreads();
    const uint32_t base=s_base;
    uint32_t SF[2]={smem_u32(&bars.s_full[0]),smem_u32(&bars.s_full[1])};
    uint32_t SE[2]={smem_u32(&bars.s_empty[0]),smem_u32(&bars.s_empty[1])};
    uint32_t PF[2]={smem_u32(&bars.p_full[0]),smem_u32(&bars.p_full[1])};
    uint32_t OR[2]={smem_u32(&bars.o_ready[0]),smem_u32(&bars.o_ready[1])};

    if (warp==4) {
        // ---- MMA warp: load K/V, issue QK ahead, then PV (O chain serial) ----
        int pse[2]={0,0}, ppf[2]={0,0};
        int lane=tid&31;
        auto load_kv=[&](int i){ int b=i&1; const __nv_bfloat16* Kb=Kg+i*N*K; const __nv_bfloat16* Vb=Vg+i*N*VD;
            for(int j=lane;j<N*K;j+=32){ int r=j/K,c=j%K; sK[b][koff(r,c)]=Kb[r*K+c]; }
            for(int j=lane;j<N*VD;j+=32){ int key=j/VD,dv=j%VD; sV[b][voff(key,dv)]=Vb[key*VD+dv]; }
            asm volatile("fence.proxy.async.shared::cta;"); };
        auto issue_qk=[&](int i){ int b=i&1; uint64_t qd0=make_desc(smem_u32(sQ),16,1024,2),kd0=make_desc(smem_u32(sK[b]),16,1024,2);
            for(int kb=0;kb<NKT;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb)); uint32_t pr=(kb>0)?1u:0u;
                asm volatile("{ .reg .pred l,p; elect.sync _|l,-1; setp.ne.b32 p,%3,0;\n @l tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                    ::"r"(base+SCOL[b]),"l"(qd),"l"(kd),"r"(pr),"r"(QK_IDESC)); }
            mma_commit(SF[b]); };
        auto issue_pv=[&](int i){ int b=i&1; uint64_t vd0=make_desc(smem_u32(sV[b]),16384,1024,2);
            for(int kb=0;kb<NKT;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pr=(i>0||kb>0)?1u:0u; uint32_t pa=base+PCOL[b]+(uint32_t)(kb*8);
                asm volatile("{ .reg .pred l,p; elect.sync _|l,-1; setp.ne.b32 p,%3,0;\n @l tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                    ::"r"(base+O_COL),"r"(pa),"l"(vd),"r"(pr),"r"(PV_IDESC)); }
            mma_commit(OR[b]); };
        load_kv(0); mbar_wait(SE[0], pse[0]); pse[0]^=1; issue_qk(0);
        for(int i=0;i<NB;++i){ int b=i&1;
            if(i+1<NB){ int nb=(i+1)&1; load_kv(i+1); mbar_wait(SE[nb], pse[nb]); pse[nb]^=1; issue_qk(i+1); }
            mbar_wait(PF[b], ppf[b]); ppf[b]^=1;   // consumer wrote P[b] + rescaled O
            issue_pv(i);                            // O += P[b]·V[b], commits OR[b]
        }
    } else if (warp<4) {
        // ---- consumer warpgroup: softmax(S[b]) -> P[b], rescale O (O chain serial) ----
        int psf[2]={0,0}, por[2]={0,0};
        float m_i=-INFINITY, l_i=0.f; float sreg[N];
        if(tid==0){ mbar_arrive(SE[0]); mbar_arrive(SE[1]); }   // buffers initially free
        for(int i=0;i<NB;++i){ int b=i&1;
            mbar_wait(SF[b], psf[b]); psf[b]^=1; tmem_fence();
            float rmax=-INFINITY;
            for(int c=0;c<N;++c){ uint32_t bb=tmem_ld(base+SCOL[b]+((uint32_t)tid<<16)+c); tmem_wait_ld(); float sv; memcpy(&sv,&bb,4); sreg[c]=sv; rmax=fmaxf(rmax,sv*scale); }
            asm volatile("bar.sync 1, 128;");
            if((tid&127)==0) mbar_arrive(SE[b]);   // S[b] consumed -> MMA may reuse
            float m_new=fmaxf(m_i,rmax);
            float corr=(m_i==-INFINITY)?1.f:ex2f((m_i-m_new)*LOG2E);
            float ps=0.f;
            for(int c=0;c<N;++c){ float p=ex2f(sreg[c]*scale-m_new); sreg[c]=p; ps+=p; }
            l_i=l_i*corr+ps; m_i=m_new;
            for(int c=0;c<N/2;++c) tmem_st(base+PCOL[b]+((uint32_t)tid<<16)+c, pack_bf16(sreg[2*c],sreg[2*c+1]));
            if(i>0){ int pb=(i-1)&1; mbar_wait(OR[pb], por[pb]); por[pb]^=1;   // PV[i-1] done
                for(int d=0;d<VD;++d){ uint32_t a=base+O_COL+((uint32_t)tid<<16)+d; uint32_t bb=tmem_ld(a); tmem_wait_ld(); float o; memcpy(&o,&bb,4); o*=corr; uint32_t ob; memcpy(&ob,&o,4); tmem_st(a,ob);} }
            tmem_wait_st();
            asm volatile("bar.sync 1, 128;");
            if((tid&127)==0) mbar_arrive(PF[b]);   // P[b] written + O rescaled
        }
        // wait final PV (PV[NB-1] commits OR[(NB-1)&1]); then epilogue
        { int pb=(NB-1)&1; mbar_wait(OR[pb], por[pb]); por[pb]^=1; }
        tmem_fence();
        for(int d=0;d<VD;++d){ uint32_t bb=tmem_ld(base+O_COL+((uint32_t)tid<<16)+d); tmem_wait_ld(); float o; memcpy(&o,&bb,4); Oout[tid*VD+d]=o/l_i; }
        if(Lout) Lout[tid]=logf(l_i)+m_i;
    }
    __syncthreads();
    if(warp==4) tmem_dealloc(base,512);
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
    float scale=1.0f/std::sqrt((float)K);
    size_t smem=(size_t)(M*K + 2*N*K + 2*N*VD)*2 + 1024;
    ORC_CHECK(cudaFuncSetAttribute(ws_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    // warmup + timed
    for(int w=0;w<20;w++) ws_kernel<<<1,160,smem>>>(dQ,dK,dV,NB,scale,dO,dL);
    ORC_CHECK(cudaDeviceSynchronize());
    float ms = orc::benchmark_ms([&](){ ws_kernel<<<1,160,smem>>>(dQ,dK,dV,NB,scale,dO,dL); }, 20, 50, 5, nullptr, 0);
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
