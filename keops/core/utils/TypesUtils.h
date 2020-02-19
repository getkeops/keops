#pragma once

//////////////////////////////////////////////
//         //
//////////////////////////////////////////////

namespace keops {

template < class FUN, int DIM, typename TYPEOUT, typename TYPEIN > 
DEVICE INLINE void VectApply(TYPEOUT *out, TYPEIN *arg1, TYPEIN arg2) {
  #pragma unroll
  for(int k=0; k<DIM; k++)
    FUN()(out+k, arg1[k], arg2);
}

template < int DIM, typename TYPE > 
DEVICE INLINE void VectAssign(TYPE *out, TYPE val) {
  #pragma unroll
  for(int k=0; k<DIM; k++)
    out[k] = val;
}

template < typename TYPEOUT, typename TYPEIN > 
DEVICE INLINE TYPEOUT cast_to(TYPEIN x) {
  return x;
}

template < typename TYPEOUT, typename TYPEIN > 
DEVICE INLINE void add_to(TYPEOUT *out, TYPEIN in) {
  *out = in;
}

#ifdef __CUDACC__

template < >
DEVICE INLINE float2 cast_to<float2,float>(float in) {
  float2 out;
  out.x = in;
  out.y = in;
  return out;
}

DEVICE INLINE void operator+=(float2& out, const float2& in) {
  out.x += in.x;
  out.y += in.y;
}

#if USE_HALF
template < >
DEVICE INLINE half2 cast_to<half2,float>(float in) {
  return __float2half2_rn(in);
}

template < >
DEVICE INLINE half2 cast_to<half2,float2>(float2 in) {
  return __float22half2_rn(in);
}

template < >
DEVICE INLINE float2 cast_to<float2,half2>(half2 in) {
  return __half22float2(in);
}
#endif
#endif



}
