#pragma once


#include <sstream>

#include "core/Pack.h"
#include "core/autodiff.h"

#include "core/formulas/constants.h"
#include "core/formulas/factorize.h"

// import all math implementations
#include "core/formulas/maths/Minus.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Subtract.h"
#include "core/formulas/maths/Exp.h"
#include "core/formulas/maths/Inv.h"
#include "core/formulas/maths/Sqrt.h"


// import all operation on vector implementations
#include "core/formulas/norms/Scalprod.h"
#include "core/formulas/norms/SqDist.h"



namespace keops {

//////////////////////////////////////////////////////////////
////             STANDARD RADIAL FUNCTIONS                ////
//////////////////////////////////////////////////////////////

template < class R2, class C >
using GaussFunction = Exp<Scal<C,Minus<R2>>>;

template < class R2, class C >
using CauchyFunction = Inv<Add<IntConstant<1>,Scal<C,R2>>>;

template < class R2, class C >
using LaplaceFunction = Exp<Minus< Scal<C,Sqrt<R2>>>>;

template < class R2, class C >
using InverseMultiquadricFunction = Inv<Sqrt<Add< Inv<C>,R2>>>;

template < class R2, class C, class W >
using SumGaussFunction = Scalprod<W,Exp<Scal<Minus<R2>,C>>>;

//////////////////////////////////////////////////////////////
////                 SCALAR RADIAL KERNELS                ////
//////////////////////////////////////////////////////////////

// Utility function

// for some reason the following variadic template version shoudl work but the nvcc compiler does not like it :
//template < class X, class Y, class B, template<class,class...> class F, class... PARAMS >
//using ScalarRadialKernel = Scal<F<SqDist<X,Y>,PARAMS...>,B>;

// so we use two distinct ScalarRadialKernel aliases, depending on the number of parameters :

template < class X, class Y, class B, template<class,class> class F, class PARAMS >
using ScalarRadialKernel_1 = Scal<F<SqDist<X,Y>,PARAMS>,B>;

template < class X, class Y, class B, template<class,class,class> class F, class PARAMS1, class PARAMS2 >
using ScalarRadialKernel_2 = Scal<F<SqDist<X,Y>,PARAMS1,PARAMS2>,B>;

// Utility aliases :
template < class C, class X, class Y, class B >
using GaussKernel = ScalarRadialKernel_1<X,Y,B,GaussFunction,C>;

template < class C, class X, class Y, class B >
using CauchyKernel = ScalarRadialKernel_1<X,Y,B,CauchyFunction,C>;

template < class C, class X, class Y, class B >
using LaplaceKernel = ScalarRadialKernel_1<X,Y,B,LaplaceFunction,C>;

template < class C, class X, class Y, class B >
using InverseMultiquadricKernel = ScalarRadialKernel_1<X,Y,B,InverseMultiquadricFunction,C>;

template < class C, class W, class X, class Y, class B >
using SumGaussKernel = ScalarRadialKernel_2<X,Y,B,SumGaussFunction,C,W>;

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

  static void PrintId(std::stringstream& str) {
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
  static HOST_DEVICE INLINE void Eval(__TYPE__* gammai, ARGS... args) {
    __TYPE__* params = Get<IndVal_Alias<INDS,C::N>::ind>(args...);
    __TYPE__* xi = Get<IndVal_Alias<INDS,X::N>::ind>(args...);
    __TYPE__* yj = Get<IndVal_Alias<INDS,Y::N>::ind>(args...);
    __TYPE__* betaj = Get<IndVal_Alias<INDS,B::N>::ind>(args...);
    __TYPE__ r2 = 0.0f;
    __TYPE__ temp;
    for(int k=0; k<DIMPOINT; k++) {
      temp =  yj[k]-xi[k];
      r2 += temp*temp;
    }
    __TYPE__ s = exp(-r2*params[0]);
    for(int k=0; k<DIMVECT; k++)
      gammai[k] = s * betaj[k];
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

  static void PrintId(std::stringstream& str) {
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
  static HOST_DEVICE INLINE void Eval(__TYPE__* gammai, ARGS... args) {
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

  static void PrintId(std::stringstream& str) {
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
  static HOST_DEVICE INLINE void Eval(__TYPE__* gammai, ARGS... args) {
    __TYPE__* params = Get<IndVal_Alias<INDS,C::N>::ind>(args...);
    __TYPE__* xi = Get<IndVal_Alias<INDS,X::N>::ind>(args...);
    __TYPE__* yj = Get<IndVal_Alias<INDS,Y::N>::ind>(args...);
    __TYPE__* betaj = Get<IndVal_Alias<INDS,B::N>::ind>(args...);
    __TYPE__* etai = Get<IndVal_Alias<INDS,GRADIN::N>::ind>(args...);

    __TYPE__ r2 = 0.0f, sga = 0.0f;                 // Don't forget to initialize at 0.0
    __TYPE__ xmy[DIMPOINT];
#pragma unroll
    for(int k=0; k<DIMPOINT; k++) {                 // Compute the L2 squared distance r2 = | x_i-y_j |_2^2
      xmy[k] =  xi[k]-yj[k];
      r2 += xmy[k]*xmy[k];
    }
#pragma unroll
    for(int k=0; k<DIMVECT; k++)                    // Compute the L2 dot product <a_i, b_j>
      sga += betaj[k]*etai[k];
    __TYPE__ s = - 2.0 * sga * exp(-r2*params[0]);  // Don't forget the 2 !
#pragma unroll
    for(int k=0; k<DIMPOINT; k++)                   // Increment the output vector gammai - which is a POINT
      gammai[k] = s * xmy[k];
  }

  // direct implementation stops here, so we link back to the usual autodiff module
  template < class V2, class GRADIN2 >
  using DiffT = Grad<GenericVersion,V2,GRADIN2>;

};

}
