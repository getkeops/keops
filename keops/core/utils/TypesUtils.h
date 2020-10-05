#pragma once

//////////////////////////////////////////////
//         //
//////////////////////////////////////////////

namespace keops {
	
template < typename TYPEOUT, typename TYPEIN > 
DEVICE INLINE TYPEOUT cast_to(TYPEIN x) {
  return (TYPEOUT)x;
}

template < typename TYPEOUT, typename TYPEIN > 
DEVICE INLINE void add_to(TYPEOUT *out, TYPEIN in) {
  *out = (TYPEOUT)in;
}

#ifdef __CUDACC__

template < >
DEVICE INLINE float2 cast_to<float2, float>(float in) {
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

#define STATIC_MAX(a,b) ((a < b) ? b : a)

template < class FUN, int DIMOUT, int DIMIN, typename TYPEOUT, typename TYPEIN > 
DEVICE INLINE void VectApply(TYPEOUT *out, TYPEIN *arg) {
	static const int DIMLOOP = STATIC_MAX(DIMOUT,DIMIN);
	static_assert( ((DIMOUT==DIMLOOP)||(DIMOUT==1)) &&
				   ((DIMIN ==DIMLOOP)||(DIMIN ==1)) ,"incompatible dimensions in VectApply");
	static const int incr_out = (DIMOUT==DIMLOOP) ? 1 : 0;
	static const int incr_in  = (DIMIN ==DIMLOOP) ? 1 : 0;
	#pragma unroll
 	for(int k=0; k<DIMLOOP; k++)
    	FUN()(out[k*incr_out], arg[k*incr_in]);
}

template < class FUN, int DIMOUT, int DIMIN1, int DIMIN2, typename TYPEOUT, typename TYPEIN > 
DEVICE INLINE void VectApply(TYPEOUT *out, TYPEIN *arg1, TYPEIN *arg2) {
	static const int DIMLOOP = STATIC_MAX(DIMOUT,STATIC_MAX(DIMIN1,DIMIN2));
	static_assert( ((DIMOUT==DIMLOOP)||(DIMOUT==1)) &&
				   ((DIMIN1==DIMLOOP)||(DIMIN1==1)) &&
				   ((DIMIN2==DIMLOOP)||(DIMIN2==1)) ,"incompatible dimensions in VectApply");
	static const int incr_out = (DIMOUT==DIMLOOP) ? 1 : 0;
	static const int incr_in1 = (DIMIN1==DIMLOOP) ? 1 : 0;
	static const int incr_in2 = (DIMIN2==DIMLOOP) ? 1 : 0;
	#pragma unroll
  	for(int k=0; k<DIMLOOP; k++)
    	FUN()(out[k*incr_out], arg1[k*incr_in1], arg2[k*incr_in2]);
}

template < class FUN, int DIMOUT, int DIMIN1, int DIMIN2, int DIMIN3, typename TYPEOUT, typename TYPEIN > 
DEVICE INLINE void VectApply(TYPEOUT *out, TYPEIN *arg1, TYPEIN *arg2, TYPEIN *arg3) {
	static const int DIMLOOP = STATIC_MAX(DIMOUT,STATIC_MAX(DIMIN1,STATIC_MAX(DIMIN2,DIMIN3)));
	static_assert( ((DIMOUT==DIMLOOP)||(DIMOUT==1)) &&
	   			   ((DIMIN1==DIMLOOP)||(DIMIN1==1)) &&
		   		   ((DIMIN2==DIMLOOP)||(DIMIN2==1)) &&
				   ((DIMIN3==DIMLOOP)||(DIMIN3==1)) ,"incompatible dimensions in VectApply");
	static const int incr_out = (DIMOUT==DIMLOOP) ? 1 : 0;
	static const int incr_in1 = (DIMIN1==DIMLOOP) ? 1 : 0;
	static const int incr_in2 = (DIMIN2==DIMLOOP) ? 1 : 0;
	static const int incr_in3 = (DIMIN3==DIMLOOP) ? 1 : 0;
	#pragma unroll
  	for(int k=0; k<DIMLOOP; k++)
    	FUN()(out[k*incr_out], arg1[k*incr_in1], arg2[k*incr_in2], arg3[k*incr_in3]);
}

template < int DIM, typename TYPEOUT, typename TYPEIN > 
DEVICE INLINE void VectAssign(TYPEOUT *out, TYPEIN val) {
  #pragma unroll
  for(int k=0; k<DIM; k++)
    out[k] = cast_to<TYPEOUT>(val);
}

template < int DIM, typename TYPEOUT, typename TYPEIN > 
DEVICE INLINE void VectCopy(TYPEOUT *out, TYPEIN *in) {
  #pragma unroll
  for(int k=0; k<DIM; k++)
    out[k] = cast_to<TYPEOUT>(in[k]);
}



}
