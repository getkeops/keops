#pragma once

#include "core/pack/UnivPack.h"
#include "core/pack/CondType.h"
#include "core/pack/IsSame.h"
#include "core/pack/IndVal.h"
#include "core/pack/MergePacks.h"
#include "core/autodiff/Grad.h"
#include "core/formulas/Factorize.h"
#include "core/formulas/maths/Minus.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Exp.h"
#include "core/formulas/maths/Subtract.h"
#include "core/formulas/kernels/ScalarRadialKernels.h"

#include "core/pre_headers.h"


namespace keops {

template < class C, class X, class Y, class B >
using GaussKernel = ScalarRadialKernel_1<X,Y,B,GaussFunction,C>;

#define GaussKernel(C,X,Y,B) KeopsNS<GaussKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()


//////////////////////////////////////////////////////////////
////                 FACTORIZED GAUSS KERNEL              ////
//////////////////////////////////////////////////////////////
template < class C, class X, class Y, class B >
using GaussKernel_Factorized = Factorize< GaussKernel<C,X,Y,B>, Subtract<X,Y> >;



//////////////////////////////////////////////////////////////
////   DIRECT IMPLEMENTATIONS FOR SCALAR RADIAL KERNELS   ////
////    (FOR FASTER COMPUTATIONS)                         ////
//////////////////////////////////////////////////////////////

// specific implementation of the gaussian kernel and its gradient wrt to X


template < class C, class X, class Y, class B > struct GaussKernel_specific;
template < class C, class X, class Y, class B, class V, class GRADIN > struct GradGaussKernel_specific;

template < class C, class X, class Y, class B >
struct GaussKernel_specific {

  static_assert(C::DIM==1,"First template argument must be a of dimension 1 for GaussKernel_specific");
  static_assert(C::CAT==2,"First template argument must be a parameter variable (CAT=2) for GaussKernel_specific");
  static_assert(X::CAT!=Y::CAT,"Second and third template arguments must not be of the same category for GaussKernel_specific");
  static_assert(Y::CAT==B::CAT,"Third and fourth template arguments must be of the same category for GaussKernel_specific");

  using GenericVersion = GaussKernel<C,X,Y,B>;

  static const int DIM = GenericVersion::DIM;
  static const int DIMPOINT = X::DIM;
  static const int DIMVECT = DIM;

  static void PrintId(::std::stringstream& str) {
    str << "GaussKernel_specific(";
    C::PrintId(str);
    str << ",";
    X::PrintId(str);
    str << ",";
    Y::PrintId(str);
    str << ",";
    B::PrintId(str);
    str << ")";
  }

  template < int CAT >
  using VARS = typename GenericVersion::template VARS<CAT>;

  using THIS = GaussKernel_specific<C,X,Y,B>;

  template<class U, class V>
  using Replace = CondType< V, THIS, IsSameType<U,THIS>::val >;

  using AllTypes = univpack<THIS>;

  template < class INDS, typename... ARGS >
  static DEVICE INLINE void Eval(__TYPE__* gammai, ARGS... args) {
    __TYPE__* params = Get<IndVal_Alias<INDS,C::N>::ind>(args...);
    __TYPE__* xi = Get<IndVal_Alias<INDS,X::N>::ind>(args...);
    __TYPE__* yj = Get<IndVal_Alias<INDS,Y::N>::ind>(args...);
    __TYPE__* betaj = Get<IndVal_Alias<INDS,B::N>::ind>(args...);
#if USE_HALF && GPU_ON
    __TYPE__ r2 = __float2half2_rn(0.0f);
#elif USE_HALF
#else
    __TYPE__ r2 = 0.0f;
#endif
    __TYPE__ temp;
#pragma unroll
    for(int k=0; k<DIMPOINT; k++) {
#if USE_HALF && GPU_ON
      temp =  __hsub2(yj[k],xi[k]);
      r2 = __hfma2(r2, temp, temp);
#elif USE_HALF
      //temp =  yj[k]-xi[k];
      //r2 = r2 + temp*temp;
#else
      temp =  yj[k]-xi[k];
      r2 += temp*temp;
#endif
    }
#if USE_HALF && GPU_ON
    __TYPE__ s = h2exp(_hneg2(__hmul2(r2,params[0])));
#pragma unroll
    for(int k=0; k<DIMVECT; k++)
      gammai[k] = __hmul2(s,betaj[k]);
#elif USE_HALF
    __TYPE__ s;
    //__TYPE__ s = exp((float)(-r2*params[0]));
//#pragma unroll
    //for(int k=0; k<DIMVECT; k++)
    //  gammai[k] = s * betaj[k];
#else
    __TYPE__ s = exp(-r2*params[0]);
#pragma unroll
    for(int k=0; k<DIMVECT; k++)
      gammai[k] = s * betaj[k];
#endif
  }

  template < class V, class GRADIN >
  using DiffT = GradGaussKernel_specific<C,X,Y,B,V,GRADIN>;

};

// by default we link to the standard autodiff versions of the gradients
template < class C, class X, class Y, class B, class V, class GRADIN >
struct GradGaussKernel_specific {
  using GenericVersion = Grad<GaussKernel<C,X,Y,B>,V,GRADIN>;

  static const int DIM = GenericVersion::DIM;
  static const int DIMPOINT = X::DIM;
  static const int DIMVECT = DIM;

  template < int CAT >
  using VARS = typename GenericVersion::template VARS<CAT>;

  static void PrintId(::std::stringstream& str) {
    str << "GradGaussKernel_specific(";
    C::PrintId(str);
    str << ",";
    X::PrintId(str);
    str << ",";
    Y::PrintId(str);
    str << ",";
    B::PrintId(str);
    str << ",";
    V::PrintId(str);
    str << ",";
    GRADIN::PrintId(str);
    str << ")";
  }

  using THIS = GradGaussKernel_specific<C,X,Y,B,V,GRADIN>;

  template<class E, class F>
  using Replace = CondType< F, GradGaussKernel_specific<C,X,Y,B,V,typename GRADIN::template Replace<E,F>>, IsSameType<E,THIS>::val >;

  using AllTypes = MergePacks < univpack<THIS,V>, typename GRADIN::AllTypes >;

  template < class INDS, typename... ARGS >
  static DEVICE INLINE void Eval(__TYPE__* gammai, ARGS... args) {
    GenericVersion::template Eval<INDS>(gammai,args...);
  }

  template < class V2, class GRADIN2 >
  using DiffT = Grad<GenericVersion,V2,GRADIN2>;

};

// specific implementation of gradient wrt X
template < class C, class X, class Y, class B, class GRADIN >
struct GradGaussKernel_specific<C,X,Y,B,X,GRADIN> {
  using GenericVersion = Grad<GaussKernel<C,X,Y,B>,X,GRADIN>;

  static const int DIM = GenericVersion::DIM;
  static const int DIMPOINT = X::DIM;
  static const int DIMVECT = DIM;

  template < int CAT >
  using VARS = typename GenericVersion::template VARS<CAT>;

  using THIS = GradGaussKernel_specific<C,X,Y,B,X,GRADIN>;

  static void PrintId(::std::stringstream& str) {
    str << "GradGaussKernel_specific(";
    C::PrintId(str);
    str << ",";
    X::PrintId(str);
    str << ",";
    Y::PrintId(str);
    str << ",";
    B::PrintId(str);
    str << ",";
    X::PrintId(str);
    str << ",";
    GRADIN::PrintId(str);
    str << ")";
  }

  template<class U, class V>
  using Replace = CondType< V, GradGaussKernel_specific<C,X,Y,B,X,typename GRADIN::template Replace<U,V>>, IsSameType<U,THIS>::val >;

  using AllTypes = MergePacks < univpack<THIS,X>, typename GRADIN::AllTypes >;

  template < class INDS, typename... ARGS >
  static DEVICE INLINE void Eval(__TYPE__* gammai, ARGS... args) {
    __TYPE__* params = Get<IndVal_Alias<INDS,C::N>::ind>(args...);
    __TYPE__* xi = Get<IndVal_Alias<INDS,X::N>::ind>(args...);
    __TYPE__* yj = Get<IndVal_Alias<INDS,Y::N>::ind>(args...);
    __TYPE__* betaj = Get<IndVal_Alias<INDS,B::N>::ind>(args...);
    __TYPE__* etai = Get<IndVal_Alias<INDS,GRADIN::N>::ind>(args...);
    __TYPE__ xmy[DIMPOINT];
#if USE_HALF && GPU_ON
    __TYPE__ r2 = __float2half2_rn(0.0f), sga =__float2half2_rn(0.0f);                 // Don't forget to initialize at 0.0
#elif USE_HALF
#else
    __TYPE__ r2 = 0.0f, sga = 0.0f;                 // Don't forget to initialize at 0.0
#endif
#pragma unroll
    for(int k=0; k<DIMPOINT; k++) {                 // Compute the L2 squared distance r2 = | x_i-y_j |_2^2
#if USE_HALF && GPU_ON
      xmy[k] =  __hsub2(xi[k], yj[k]);
      r2 = __hfma2(r2, xmy[k], xmy[k]);
#elif USE_HALF
#else
      xmy[k] =  xi[k]-yj[k];
      r2 += xmy[k]*xmy[k];
#endif
    }
#pragma unroll
    for(int k=0; k<DIMVECT; k++)                    // Compute the L2 dot product <a_i, b_j>
#if USE_HALF && GPU_ON
      sga = hfma(sga, betaj[k], etai[k]);
    __TYPE__ s = _hneg2(__hmul2(__hmul2(2.0,sga), h2exp(_hneg2(__hmul2(r2,params[0])))));  // Don't forget the 2 !
#elif USE_HALF
    {}
    //  sga = sga + betaj[k]*etai[k];
    //__TYPE__ s = - 2.0 * (float)sga * exp((float)(-r2*params[0]));  // Don't forget the 2 !
#else
      sga += betaj[k]*etai[k];
    __TYPE__ s = - 2.0 * sga * exp(-r2*params[0]);  // Don't forget the 2 !
#endif
#pragma unroll
    for(int k=0; k<DIMPOINT; k++)                   // Increment the output vector gammai - which is a POINT
#if USE_HALF && GPU_ON
      gammai[k] = __hmul2(s, xmy[k]);
#elif USE_HALF
      {}
#else
      gammai[k] = s * xmy[k];
#endif
  }

  // direct implementation stops here, so we link back to the usual autodiff module
  template < class V2, class GRADIN2 >
  using DiffT = Grad<GenericVersion,V2,GRADIN2>;

};


}
