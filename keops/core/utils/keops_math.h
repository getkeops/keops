#pragma once

#include <cmath>

namespace keops {
	
template < typename TYPE > DEVICE INLINE TYPE keops_abs(TYPE& x) { return fabs(x); }
template < typename TYPE > DEVICE INLINE TYPE keops_exp(TYPE& x) { return exp(x); }
template < typename TYPE > DEVICE INLINE TYPE keops_cos(TYPE& x) { return cos(x); }
template < typename TYPE > DEVICE INLINE TYPE keops_sin(TYPE& x) { return sin(x); }

#ifdef __CUDA_ARCH__
  
DEVICE INLINE float keops_abs(float& x) { return fabsf(x); } 
DEVICE INLINE float keops_exp(float& x) { return expf(x); } 
DEVICE INLINE float keops_cos(float& x) { return cosf(x); } 
DEVICE INLINE float keops_sin(float& x) { return sinf(x); } 
    
  #if USE_HALF 

// N.B. Some of dedicated operations for half2 type : h2exp,...
// appear to be very slow (doing experiments on RTX 2080 Ti card),
// so we use float type functions twice instead :
DEVICE INLINE half2& keops_exp(half2& x) {
    float a = __expf(__low2float(x);
    float b = __expf(__high2float(x));
    return __floats2half2_rn(a,b);
}

DEVICE INLINE half2& keops_cos(half2& x) {
    float a = __cosf(__low2float(x);
    float b = __cosf(__high2float(x));
    return __floats2half2_rn(a,b);
}

DEVICE INLINE half2& keops_sin(half2& x) {
    float a = __sinf(__low2float(x);
    float b = __sinf(__high2float(x));
    return __floats2half2_rn(a,b);
}

    #if CUDART_VERSION < 10020
  
// absolute value operation for half2 type is only available with Cuda version >= 10.2...
DEVICE INLINE half2& keops_abs(half2& x) {
        __half2 cond = __hlt2(__float2half2_rn(0.0f),x);                  // cond = (0 < x) (element-wise)
        __half2 coef = __float2half2_rn(2.0f) * cond - __float2half2_rn(1.0f);  // coef = 2*cond-1
        return coef * x;   
}
  
    #else
  
DEVICE INLINE half2& keops_abs(half2& x) { return __habs(x); }

    #endif
  #endif
#endif


}
