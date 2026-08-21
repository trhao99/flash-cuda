// regtest.cu — isolate: can nvcc/ptxas hold a 128-wide fp32 tmem fragment 1-pass, spill-free,
#include <cuda_bf16.h>
// under setmaxnreg 192, with NOTHING else competing? Binary-search the live-fragment width.
#include <cuda.h>
#include <cstdio>
#include <cstdint>
__device__ __forceinline__ float ex2f(float x){ float r; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(r):"f"(x)); return r; }
__device__ __forceinline__ uint32_t tmem_ld(uint32_t a){ uint32_t v; asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];":"=r"(v):"r"(a)); return v; }
__device__ __forceinline__ void tmem_st(uint32_t a, uint32_t v){ asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"::"r"(a),"r"(v)); }
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;"); }
__device__ __forceinline__ void tmem_wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;"); }
__device__ __forceinline__ void tmem_fence(){ asm volatile("tcgen05.fence::after_thread_sync;"); }
__device__ __forceinline__ void mbar_wait(uint32_t m,int ph){ asm volatile("{ .reg .pred P;\nLW: mbarrier.try_wait.parity.shared.b64 P,[%0],%1,10000000;\n @P bra D; bra LW;\nD: }"::"r"(m),"r"(ph)); }
__device__ __forceinline__ void mbar_arrive(uint32_t m){ asm volatile("mbarrier.arrive.shared.b64 _, [%0];"::"r"(m)); }
#define LD32_INTO(ADDR, F, OFF) \
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" \
        : "=r"(F[OFF+0]),"=r"(F[OFF+1]),"=r"(F[OFF+2]),"=r"(F[OFF+3]),"=r"(F[OFF+4]),"=r"(F[OFF+5]),"=r"(F[OFF+6]),"=r"(F[OFF+7]), \
          "=r"(F[OFF+8]),"=r"(F[OFF+9]),"=r"(F[OFF+10]),"=r"(F[OFF+11]),"=r"(F[OFF+12]),"=r"(F[OFF+13]),"=r"(F[OFF+14]),"=r"(F[OFF+15]), \
          "=r"(F[OFF+16]),"=r"(F[OFF+17]),"=r"(F[OFF+18]),"=r"(F[OFF+19]),"=r"(F[OFF+20]),"=r"(F[OFF+21]),"=r"(F[OFF+22]),"=r"(F[OFF+23]), \
          "=r"(F[OFF+24]),"=r"(F[OFF+25]),"=r"(F[OFF+26]),"=r"(F[OFF+27]),"=r"(F[OFF+28]),"=r"(F[OFF+29]),"=r"(F[OFF+30]),"=r"(F[OFF+31]) \
        : "r"(ADDR))

#ifndef WIDTH
#define WIDTH 128
#endif
// only warps 0-3 do the fragment; rest just setmaxnreg.dec (mimic producer warps)
__global__ void __launch_bounds__(512,1) regtest(uint32_t base,float scale,int NB,float* Oout){
    const int tid=threadIdx.x, warp=tid>>5, lr=tid&127;
    if(warp<8) asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
    else       asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
#ifdef WITH_PRODUCER
    // mimic MMA/load producer warps: descriptor math + mma asm loop, guarded by warp>=4
    if(warp>=8){
        uint64_t acc=(uint64_t)base*0x9E3779B97F4A7C15ull;
        for(int i=0;i<NB;++i){
            uint64_t qd=acc+i, kd=acc^(i*1024);
            #pragma unroll
            for(int kb=0;kb<8;++kb){ uint64_t q2=qd+kb*2, k2=kd+kb*2; uint32_t pr=(kb>0);
                asm volatile("{ .reg .pred l,p; elect.sync _|l,-1; setp.ne.b32 p,%3,0;\n @l tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                    ::"r"(base),"l"(q2),"l"(k2),"r"(pr),"r"(0x8200490u)); }
        }
        if(threadIdx.x==999) Oout[1]=(float)acc;
        return;
    }
#endif
    if(warp>=8) return;
    const float LOG2E=1.4426950408889634f;
#ifdef WITH_BARS
    // kernel-wide live mbarrier-address arrays (like ws8): held across the whole loop
    uint32_t SF[2]={base+8,base+16}, PF[2]={base+24,base+32}, OR[2]={base+40,base+48};
    uint32_t KVF[2]={base+56,base+64}, KVE[2]={base+72,base+80};
    const int jj=warp>>2;
    if((threadIdx.x&127)==0){ mbar_arrive(KVE[0]); mbar_arrive(KVF[jj]); }
#endif
    float m_i=-INFINITY,l_i=0.f;
    for(int i=0;i<NB;++i){
#ifdef WITH_SYNC
        mbar_wait(base+8, i&1); tmem_fence();
#endif
#ifdef WITH_J
        const int j = warp>>2; const int SCOL[2]={0,128}, PCOL[2]={0,128};
        uint32_t a0=base+SCOL[j]+((uint32_t)lr<<16);
#else
        uint32_t a0=base+((uint32_t)lr<<16);
#endif
        uint32_t f[WIDTH];
        #pragma unroll
        for(int g=0; g<WIDTH; g+=32) LD32_INTO(a0+g, f, g);
        tmem_wait_ld();
        float rmax=-INFINITY;
        #pragma unroll
        for(int t=0;t<WIDTH;++t) rmax=fmaxf(rmax,__uint_as_float(f[t])*scale);
        float m_new=fmaxf(m_i,rmax); float corr=(m_i==-INFINITY)?1.f:ex2f((m_i-m_new)*LOG2E);
        float ps=0.f;
        #pragma unroll
        for(int t=0;t<WIDTH;t+=2){ float s0=__uint_as_float(f[t])*scale,s1=__uint_as_float(f[t+1])*scale;
            float p0=ex2f((s0-m_new)*LOG2E),p1=ex2f((s1-m_new)*LOG2E); ps+=p0+p1;
            __nv_bfloat16 x=__float2bfloat16_rn(p0),y=__float2bfloat16_rn(p1); uint16_t ux,uy; memcpy(&ux,&x,2); memcpy(&uy,&y,2);
#ifdef WITH_J
            tmem_st(base+PCOL[j]+((uint32_t)lr<<16)+(t>>1), uint32_t(ux)|(uint32_t(uy)<<16)); }
#else
            tmem_st(a0+(t>>1), uint32_t(ux)|(uint32_t(uy)<<16)); }
#endif
        tmem_wait_st();
        l_i=l_i*corr+ps; m_i=m_new;
#ifdef WITH_SYNC
        asm volatile("bar.sync 1, 128;");
        if((tid&127)==0) mbar_arrive(base+16);
#endif
#ifdef WITH_ORESC
        // inline O-rescale (128 tmem ld/st) held in same warp as the fragment
        for(int d=0;d<128;++d){ uint32_t a=base+256+((uint32_t)lr<<16)+d; uint32_t w=tmem_ld(a); float o=__uint_as_float(w)*corr; tmem_st(a,__float_as_uint(o)); }
#endif
    }
#ifdef WITH_EPI
    for(int d=0;d<128;++d){ uint32_t w=tmem_ld(base+256+((uint32_t)lr<<16)+d); tmem_wait_ld(); Oout[lr*128+d]=__uint_as_float(w)/l_i; }
#endif
    if(lr==0 && warp==0) Oout[0]=l_i;
}
int main(){ printf("built WIDTH=%d\n",WIDTH); return 0; }
