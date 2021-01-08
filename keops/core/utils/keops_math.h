#pragma once

#include <cmath>


namespace keops {
	
template < typename TYPE > DEVICE INLINE TYPE keops_fma(TYPE x, TYPE y, TYPE z) { return fma(x,y,z); }
template < typename TYPE > DEVICE INLINE TYPE keops_pow(TYPE x, int n) { return pow(x,n); }
template < typename TYPE > DEVICE INLINE TYPE keops_log(TYPE x) { return log(x); }
template < typename TYPE > DEVICE INLINE TYPE keops_xlogx(TYPE x) { return x ? x * keops_log(x) : 0.0f; }
template < typename TYPE > DEVICE INLINE TYPE keops_rcp(TYPE x) { return 1.0f / x; }
template < typename TYPE > DEVICE INLINE TYPE keops_abs(TYPE x) { return fabs(x); }
template < typename TYPE > DEVICE INLINE TYPE keops_exp(TYPE x) { return exp(x); }
template < typename TYPE > DEVICE INLINE TYPE keops_cos(TYPE x) { return cos(x); }
template < typename TYPE > DEVICE INLINE TYPE keops_sin(TYPE x) { return sin(x); }
template < typename TYPE > DEVICE INLINE TYPE keops_relu(TYPE x) { return (x<0.0f)? 0.0f : x; }
template < typename TYPE > DEVICE INLINE TYPE keops_step(TYPE x) { return (x<0.0f)? 0.0f : 1.0f; }
template < typename TYPE > DEVICE INLINE TYPE keops_sign(TYPE x) { return (x>0.0f)? 1.0f : ( (x<0.0f)? -1.0f : 0.0f ); }
template < typename TYPE > DEVICE INLINE TYPE keops_clamp(TYPE x, TYPE a, TYPE b) { return (x<a)? a : ( (x>b)? b : x ); }
template < typename TYPE > DEVICE INLINE TYPE keops_clampint(TYPE x, int a, int b) { return (x<a)? a : ( (x>b)? b : x ); }
template < typename TYPE > DEVICE INLINE TYPE keops_diffclampint(TYPE x, int a, int b) { return (x<a)? 0.0f : ( (x>b)? 0.0f : 1.0f ); }
template < typename TYPE > DEVICE INLINE TYPE keops_sqrt(TYPE x) { return sqrt(x); }
template < typename TYPE > DEVICE INLINE TYPE keops_rsqrt(TYPE x) { return 1.0f / sqrt(x); }
template < typename TYPE > DEVICE INLINE TYPE keops_acos(TYPE x) { return acos(x); }
template < typename TYPE > DEVICE INLINE TYPE keops_asin(TYPE x) { return asin(x); }
template < typename TYPE > DEVICE INLINE TYPE keops_atan(TYPE x) { return atan(x); }

#ifdef __CUDA_ARCH__  
DEVICE INLINE float keops_pow(float x, int n) { return powf(x,n); } 
DEVICE INLINE float keops_fma(float x, float y, float z) { return fmaf(x,y,z); } 
DEVICE INLINE float keops_log(float x) { return logf(x); } 
DEVICE INLINE float keops_rcp(float x) { return __frcp_rn(x); } 
DEVICE INLINE float keops_abs(float x) { return fabsf(x); } 
DEVICE INLINE float keops_exp(float x) { return expf(x); } 
DEVICE INLINE float keops_cos(float x) { return cosf(x); } 
DEVICE INLINE float keops_sin(float x) { return sinf(x); } 
DEVICE INLINE float keops_sqrt(float x) { return sqrtf(x); } 
DEVICE INLINE float keops_rsqrt(float x) { return rsqrtf(x); } 
DEVICE INLINE float keops_acos(float x) { return acosf(x); }
DEVICE INLINE float keops_asin(float x) { return asinf(x); }
DEVICE INLINE float keops_atan(float x) { return atanf(x); }
DEVICE INLINE double keops_rsqrt(double x) { return rsqrt(x); } 
   
#if USE_HALF 
DEVICE INLINE half2 keops_fma(half2 x, half2 y, half2 z) { return __hfma2(x,y,z); }
DEVICE INLINE half2 keops_rcp(half2 x) { return h2rcp(x); } 
DEVICE INLINE half2 keops_xlogx(half2 x) { return x * h2log(x + __heq2(x,__float2half2_rn(0.0f))); } 
DEVICE INLINE half2 keops_relu(half2 x) { return __hlt2(__float2half2_rn(0.0f),x) * x; }  // (0<x) * x (element-wise) 
DEVICE INLINE half2 keops_sign(half2 x) { return __hgt2(x,__float2half2_rn(0.0f)) - __hlt2(x,__float2half2_rn(0.0f)); } // (x>0) - (x<0) (element-wise)
DEVICE INLINE half2 keops_clamp(half2 x,half2 a,half2 b) { return __hlt2(a,x)*(x-a) + a - __hlt2(b,x); } // (a<x)(x-a) + a - (b<x)(x-b) (element-wise)
DEVICE INLINE half2 keops_clampint(half2 x,int a,int b) { 
	half2 ah2 = __float2half2_rn((float)a);
	half2 bh2 = __float2half2_rn((float)b);
	return __hlt2(ah2,x)*(x-ah2) + ah2 - __hlt2(bh2,x); 
}
DEVICE INLINE half2 keops_diffclampint(half2 x,int a,int b) { 
	half2 ah2 = __float2half2_rn((float)a);
	half2 bh2 = __float2half2_rn((float)b);
	return __hlt2(ah2,x)*__hlt2(x,bh2); 
}
DEVICE INLINE half2 keops_step(half2 x) { return __hgt2(x,__float2half2_rn(0.0f)); }
DEVICE INLINE half2 keops_sqrt(half2 x) { return h2sqrt(x); }



// N.B. Some of dedicated operations for half2 type : h2exp,...
// appear to be very slow (doing experiments on RTX 2080 Ti card),
// so we use float type functions twice instead :
DEVICE INLINE half2 keops_log(half2 x) {
    float a = __logf(__low2float(x));
    float b = __logf(__high2float(x));
    return __floats2half2_rn(a,b);
}

DEVICE INLINE half2 keops_exp(half2 x) {
    float a = __expf(__low2float(x));
    float b = __expf(__high2float(x));
    return __floats2half2_rn(a,b);
}

DEVICE INLINE half2 keops_cos(half2 x) {
    float a = __cosf(__low2float(x));
    float b = __cosf(__high2float(x));
    return __floats2half2_rn(a,b);
}

DEVICE INLINE half2 keops_sin(half2 x) {
    float a = __sinf(__low2float(x));
    float b = __sinf(__high2float(x));
    return __floats2half2_rn(a,b);
}

    #if CUDART_VERSION < 10020
  
// absolute value operation for half2 type is only available with Cuda version >= 10.2...
DEVICE INLINE half2 keops_abs(half2 x) {
        __half2 cond = __hlt2(__float2half2_rn(0.0f),x);                  // cond = (0 < x) (element-wise)
        __half2 coef = __float2half2_rn(2.0f) * cond - __float2half2_rn(1.0f);  // coef = 2*cond-1
        return coef * x;   
}
  
    #else
  
DEVICE INLINE half2 keops_abs(half2 x) { return __habs2(x); }

    #endif
  #endif
#endif


}
