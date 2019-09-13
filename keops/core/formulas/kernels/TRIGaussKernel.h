#pragma once

#include "core/formulas/constants/IntConst.h"
#include "core/formulas/Factorize.h"
#include "core/formulas/maths/Subtract.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/kernels/DivFreeGaussKernel.h"
#include "core/formulas/kernels/CurlFreeGaussKernel.h"
#include "core/pre_headers.h"

namespace keops {


// Weighted combination of the CurlFree and DivFree kernels, which gives a Translation and Rotation Invariant kernel
// with gaussian base function.
//
//    k_tri(x,y)b = lambda * k_df(x,y)b + (1-lambda) * k_cf(x,y)b
//
// The weight lambda must be specified as the second parameter (_P<1>) when calling Eval()

template < class L, class C, class X, class Y, class B >
struct TRIGaussKernel_helper {
  static_assert(L::DIM==1,"First template argument must be a of dimension 1 for TRIGaussKernel");
  static_assert(L::CAT==2,"First template argument must be a parameter variable (CAT=2) for TRIGaussKernel");
  static_assert(C::DIM==1,"Second template argument must be a of dimension 1 for TRIGaussKernel");
  static_assert(C::CAT==2,"Second template argument must be a parameter variable (CAT=2) for TRIGaussKernel");
  static_assert(X::CAT!=Y::CAT,"Third and fourth template arguments must not be of the same category for TRIGaussKernel");
  static_assert(X::DIM==Y::DIM,"Third and fourth template arguments must have the same dimensions for TRIGaussKernel");
  static_assert(Y::CAT==B::CAT,"Fourth and fifth template arguments must be of the same category for TRIGaussKernel");
  static const int DIM = X::DIM;
  using OML = Subtract<IntConstant<1>,L>;                   // 1-lambda
  using DF = DivFreeGaussKernel_helper<C,X,Y,B>;            // k_df(x,y)b (the helper struct, because we need it below)
  using CF = CurlFreeGaussKernel_helper<C,X,Y,B>;           // k_cf(x,y)b (the helper struct, because we need it below)
  using type =  Add<Scal<L,typename DF::type>,Scal<OML,typename CF::type>>;    // final formula (not factorized)
  // here we can factorize a lot ; we look at common expressions in Div Free and Curl Free kernels:
  using G = typename DF::G;                                 // exp(-r^2/s2) can be factorized
  using D = typename DF::D;                                 // <b,x-y>(x-y) can be factorized
  using XMY = typename DF::XMY;                             // x-y can be factorized
  using R2 = typename DF::R2;                               // r^2 can be factorized
  using factorized_type = Factorize<Factorize<Factorize<Factorize<type,G>,D>,R2>,XMY>;
};

template < class L, class C, class X, class Y, class B >
using TRIGaussKernel = typename TRIGaussKernel_helper<L,C,X,Y,B>::factorized_type;

#define TRIGaussKernel(L,C,X,Y,B) KeopsNS<TRIGaussKernel<decltype(InvKeopsNS(L)),decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()

}