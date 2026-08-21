// fa4_attn_stream.cu — Blackwell SM100 MULTI-BLOCK streaming flash-attention forward (compute core).
//
// Extends fa4_attn_block to NB KV blocks with the online-softmax running max/sum + O rescale:
//   for each KV block b:  S=Q·K_bᵀ -> m_new=max(m,rowmax(S)); corr=ex2((m-m_new)*slog2);
//                         P=ex2((S-m_new)*slog2); l=l*corr+sum(P); O=O*corr; O+=P·V_b
//   epilogue: O/=l; LSE = (m + ln(l))/... (natural log).  Verifies the complete FA-fwd math.
//
// All tcgen05 primitives (tmem, QK SS, PV TS, descriptors, 128B swizzle) are the ones proven in
// fa4_qk_gemm.cu / fa4_attn_block.cu. O rescale is done in tmem (ld -> *corr -> st) before the
// accumulating PV MMA. tcgen05.fence::after_thread_sync added before readback ld (tracer note).
//
// Build: nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 fa4_attn_stream.cu -o /tmp/as
#include "probe_common.cuh"

__device__ __forceinline__ uint32_t smem_u32(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void tmem_alloc(uint32_t s,int n){ asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(s),"r"(n)); }
__device__ __forceinline__ void tmem_relinquish(){ asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int n){ asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(n)); }
__device__ __forceinline__ uint32_t tmem_ld_1(uint32_t a){ uint32_t v; asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];":"=r"(v):"r"(a)); return v; }
__device__ __forceinline__ void tmem_st_1(uint32_t a, uint32_t v){ asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"::"r"(a),"r"(v)); }
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;"); }
__device__ __forceinline__ void tmem_wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;"); }
__device__ __forceinline__ void tmem_fence_after_sync(){ asm volatile("tcgen05.fence::after_thread_sync;"); }
__device__ __forceinline__ float ex2f(float x){ float r; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(r):"f"(x)); return r; }

__device__ __forceinline__ int koff(int r,int c){ return (c>>6)*8192 + r*64 + ((c&63)^((r&7)<<3)); }
__device__ __forceinline__ int voff(int key,int dv){ return (dv>>6)*8192 + key*64 + ((dv&63)^((key&7)<<3)); }
__device__ __forceinline__ uint64_t make_desc(uint32_t addr,uint32_t lbo,uint32_t sbo,int lt){
    uint64_t d=0; d|=(uint64_t)((addr>>4)&0x3FC0u); d|=(uint64_t)((lbo>>4)&0x3FFFu)<<16;
    d|=(uint64_t)((sbo>>4)&0x3FFFu)<<32; d|=(uint64_t)1u<<46; d|=(uint64_t)(lt&0x7u)<<61; return d; }
__device__ __forceinline__ uint64_t desc_add(uint64_t b,uint32_t off){
    uint32_t lo,hi; asm("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"l"(b));
    lo+=off; uint64_t r; asm("mov.b64 %0,{%1,%2};":"=l"(r):"r"(lo),"r"(hi)); return r; }
__device__ __forceinline__ uint32_t qk_doff(int kb){ return (uint32_t)((kb&3)*2 + (kb>>2)*1024); }
__device__ __forceinline__ uint32_t v_doff(int kb){ return (uint32_t)(kb*0x80); }
__device__ __forceinline__ uint32_t pack_bf16(float a,float b){
    __nv_bfloat16 x=__float2bfloat16_rn(a),y=__float2bfloat16_rn(b); uint16_t ux,uy;
    memcpy(&ux,&x,2); memcpy(&uy,&y,2); return uint32_t(ux)|(uint32_t(uy)<<16); }

constexpr int M=128,N=128,K=128,VD=128,NKT=K/16;
constexpr uint32_t QK_IDESC=0x8200490u, PV_IDESC=0x8210490u;
constexpr int S_COL=0, P_COL=128, O_COL=256;

__device__ __forceinline__ void mma_commit(uint32_t mbar){
    asm volatile("{ .reg .pred lead; elect.sync _|lead,-1;\n"
        "  @lead tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]; }"::"r"(mbar)); }
__device__ __forceinline__ void mma_wait(uint32_t mbar,int phase){
    asm volatile("{ .reg .pred P;\nLW: mbarrier.try_wait.parity.shared.b64 P,[%0],%1,10000000;\n @P bra D; bra LW;\nD: }"::"r"(mbar),"r"(phase)); }

__global__ void __launch_bounds__(128) stream_kernel(const __nv_bfloat16* Qg,const __nv_bfloat16* Kg,
                                                    const __nv_bfloat16* Vg, int NB, float scale, float* Oout, float* Lout){
    extern __shared__ __nv_bfloat16 dyn_raw[];
    // 128B-swizzle tiles must be 1024-byte aligned (descriptor uses (addr>>4)&0x3FC0).
    uint32_t raw = smem_u32(dyn_raw);
    uint32_t pad = ((raw + 1023u) & ~1023u) - raw;   // bytes to next 1024 boundary
    __nv_bfloat16* dyn = (__nv_bfloat16*)((char*)dyn_raw + pad);
    __nv_bfloat16* sQ=dyn;                 // [M*K]
    __nv_bfloat16* sK=sQ+M*K;              // [N*K]
    __nv_bfloat16* sV=sK+N*K;              // [N*VD]
    __shared__ __align__(8) uint32_t s_base;
    __shared__ __align__(8) uint64_t s_mbar;
    const int tid=threadIdx.x, warp=tid>>5;
    const float LOG2E=1.4426950408889634f, slog2=scale*LOG2E;

    for(int i=tid;i<M*K;i+=128){ int r=i/K,c=i%K; sQ[koff(r,c)]=Qg[r*K+c]; }
    asm volatile("fence.proxy.async.shared::cta;");
    if(warp==0){ tmem_alloc(smem_u32(&s_base),512); tmem_relinquish(); }
    __syncthreads();
    const uint32_t base=s_base, mbar=smem_u32(&s_mbar);

    float m_i=-INFINITY, l_i=0.f;

    for(int blk=0; blk<NB; ++blk){
        // stage this block's K,V into swizzled smem
        const __nv_bfloat16* Kb=Kg+blk*N*K; const __nv_bfloat16* Vb=Vg+blk*N*VD;
        for(int i=tid;i<N*K;i+=128){ int r=i/K,c=i%K; sK[koff(r,c)]=Kb[r*K+c]; }
        for(int i=tid;i<N*VD;i+=128){ int key=i/VD,dv=i%VD; sV[voff(key,dv)]=Vb[key*VD+dv]; }
        asm volatile("fence.proxy.async.shared::cta;");
        if(tid==0) asm volatile("mbarrier.init.shared.b64 [%0], 1;"::"r"(mbar));
        __syncthreads();

        // QK -> S at S_COL
        if(warp==0){
            uint64_t qd0=make_desc(smem_u32(sQ),16,1024,2), kd0=make_desc(smem_u32(sK),16,1024,2);
            for(int kb=0;kb<NKT;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb)); uint32_t pr=(kb>0)?1u:0u;
                asm volatile("{ .reg .pred lead,p; elect.sync _|lead,-1; setp.ne.b32 p,%3,0;\n"
                    "  @lead tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                    ::"r"(base+S_COL),"l"(qd),"l"(kd),"r"(pr),"r"(QK_IDESC)); }
            mma_commit(mbar);
        }
        mma_wait(mbar, 0);
        tmem_fence_after_sync();
        __syncthreads();

        // softmax step (row = tid)
        float s[N]; float rmax=-INFINITY;
        for(int c=0;c<N;++c){ uint32_t b=tmem_ld_1(base+S_COL+((uint32_t)tid<<16)+c); tmem_wait_ld(); memcpy(&s[c],&b,4); rmax=fmaxf(rmax,s[c]); }
        float m_new=fmaxf(m_i,rmax);
        float corr = (m_i==-INFINITY)?1.f:ex2f((m_i-m_new)*slog2);
        float ps=0.f;
        for(int c=0;c<N;++c){ float p=ex2f((s[c]-m_new)*slog2); s[c]=p; ps+=p; }
        l_i = l_i*corr + ps; m_i = m_new;
        // store P (bf16 packed) to P_COL
        for(int c=0;c<N/2;++c) tmem_st_1(base+P_COL+((uint32_t)tid<<16)+c, pack_bf16(s[2*c],s[2*c+1]));
        // rescale existing O by corr (skip on first block)
        if(blk>0){ for(int d=0;d<VD;++d){ uint32_t a=base+O_COL+((uint32_t)tid<<16)+d; uint32_t b=tmem_ld_1(a); tmem_wait_ld(); float o; memcpy(&o,&b,4); o*=corr; uint32_t ob; memcpy(&ob,&o,4); tmem_st_1(a,ob); } }
        tmem_wait_st();
        if(tid==0) asm volatile("mbarrier.init.shared.b64 [%0], 1;"::"r"(mbar));
        __syncthreads();

        // PV: O += P·V_b  (enable_d = blk>0, i.e. accumulate onto rescaled O; first block inits O)
        if(warp==0){
            uint64_t vd0=make_desc(smem_u32(sV),16384,1024,2);
            for(int kb=0;kb<NKT;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pr=(blk>0||kb>0)?1u:0u; uint32_t pa=base+P_COL+(uint32_t)(kb*8);
                asm volatile("{ .reg .pred lead,p; elect.sync _|lead,-1; setp.ne.b32 p,%3,0;\n"
                    "  @lead tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                    ::"r"(base+O_COL),"r"(pa),"l"(vd),"r"(pr),"r"(PV_IDESC)); }
            mma_commit(mbar);
        }
        mma_wait(mbar, 0);
        tmem_fence_after_sync();
        __syncthreads();
    }

    // epilogue: O/=l, LSE = ln(l) + m/ (scale applied in log domain): here LSE(nat) = m*scale? we used raw S so
    // LSE_natural = ln(sum exp(scale*(S-m))) + scale*m = ln(l) + scale*m.
    for(int d=0;d<VD;++d){ uint32_t b=tmem_ld_1(base+O_COL+((uint32_t)tid<<16)+d); tmem_wait_ld(); float o; memcpy(&o,&b,4); Oout[tid*VD+d]=o/l_i; }
    if(Lout) Lout[tid]=logf(l_i)+scale*m_i;
    __syncthreads();
    if(warp==0) tmem_dealloc(base,512);
}

int main(int argc,char**argv){
    int NB = (argc>1)?atoi(argv[1]):3;
    ORC_CHECK(cudaSetDevice(0));
    cudaDeviceProp p{}; ORC_CHECK(cudaGetDeviceProperties(&p,0));
    std::printf("device: %s cc %d.%d  NB=%d (total keys=%d)\n",p.name,p.major,p.minor,NB,NB*N);
    int TK=NB*N;
    std::vector<uint16_t> Qh,Kh,Vh; orc::fill_bf16(Qh,M*K,1); orc::fill_bf16(Kh,TK*K,2); orc::fill_bf16(Vh,TK*VD,3);
    __nv_bfloat16 *dQ,*dK,*dV; float *dO,*dL;
    ORC_CHECK(cudaMalloc(&dQ,M*K*2)); ORC_CHECK(cudaMalloc(&dK,TK*K*2)); ORC_CHECK(cudaMalloc(&dV,TK*VD*2));
    ORC_CHECK(cudaMalloc(&dO,M*VD*4)); ORC_CHECK(cudaMalloc(&dL,M*4));
    ORC_CHECK(cudaMemcpy(dQ,Qh.data(),M*K*2,cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dK,Kh.data(),TK*K*2,cudaMemcpyHostToDevice));
    ORC_CHECK(cudaMemcpy(dV,Vh.data(),TK*VD*2,cudaMemcpyHostToDevice));
    float scale=1.0f/std::sqrt((float)K);
    size_t smem=(M*K+N*K+N*VD)*sizeof(__nv_bfloat16)+1024;
    ORC_CHECK(cudaFuncSetAttribute(stream_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    stream_kernel<<<1,128,smem>>>(dQ,dK,dV,NB,scale,dO,dL);
    ORC_CHECK(cudaGetLastError()); ORC_CHECK(cudaDeviceSynchronize());
    std::vector<float> O(M*VD),L(M); ORC_CHECK(cudaMemcpy(O.data(),dO,M*VD*4,cudaMemcpyDeviceToHost));
    ORC_CHECK(cudaMemcpy(L.data(),dL,M*4,cudaMemcpyDeviceToHost));

    double max_abs=0,max_lse=0; int shown=0;
    for(int m=0;m<M;++m){
        std::vector<double> sc(TK); double mx=-1e30;
        for(int n=0;n<TK;++n){ double a=0; for(int k=0;k<K;++k) a+=(double)orc::bf16_bits_to_f32(Qh[m*K+k])*(double)orc::bf16_bits_to_f32(Kh[n*K+k]); sc[n]=a*scale; mx=std::max(mx,sc[n]); }
        double l=0; for(int n=0;n<TK;++n){ sc[n]=std::exp(sc[n]-mx); l+=sc[n]; }
        double lse=std::log(l)+mx;
        if(std::fabs(L[m]-lse)>max_lse) max_lse=std::fabs(L[m]-lse);
        for(int d=0;d<VD;++d){ double o=0; for(int n=0;n<TK;++n) o+=sc[n]*(double)orc::bf16_bits_to_f32(Vh[n*VD+d]); o/=l;
            double err=std::fabs(O[m*VD+d]-o); if(err>max_abs) max_abs=err;
            if(err>2e-2&&shown<8){ std::printf("  bad m=%d d=%d got=%.4f ref=%.4f\n",m,d,O[m*VD+d],o); shown++; } }
    }
    std::printf("max_abs(O)=%.5f  max_abs(LSE)=%.5f\n",max_abs,max_lse);
    bool pass = max_abs<2e-2 && max_lse<2e-2;
    std::printf("%s: %d-block streaming flash-attention forward (online softmax + O rescale + LSE)\n",pass?"PASS":"FAIL",NB);
    return pass?0:1;
}
