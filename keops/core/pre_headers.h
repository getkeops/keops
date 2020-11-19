#pragma once

namespace keops {


/////////////////////////////////////////////
//         Compilation Options             //
/////////////////////////////////////////////

#ifdef __CUDA_ARCH__
  #define HOST_DEVICE __host__ __device__
  #define DEVICE __device__
  #define INLINE __forceinline__
  #define GPU_ON 1
#else
  #define HOST_DEVICE
  #define DEVICE
  #define INLINE inline
  #define GPU_ON 0
#endif

#define __INDEX__ int32_t // use int instead of double

#ifndef __TYPE__
  #define __TYPE__ float
  #define USE_DOUBLE 0
  #define USE_HALF 0
#endif

#ifndef C_CONTIGUOUS
  #define C_CONTIGUOUS 0
#endif


/////////////////////////////////////////////
//               Newsyntax                 //
/////////////////////////////////////////////

// This two dummy classes are used to prevent the compiler to be lost
// during the resolution of the templated formula.

template < class F >
struct KeopsNS : public F {};

template < class F >
F InvKeopsNS(KeopsNS< F > kf) {
  return F();
}

#define Ind(...) std::index_sequence< __VA_ARGS__ >

}
