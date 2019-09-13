#pragma once

#include "core/formulas/constants/IntConst.h"
#include "core/formulas/Factorize.h"
#include "core/formulas/maths/Divide.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Subtract.h"
#include "core/formulas/norms/SqDist.h"
#include "core/formulas/norms/Scalprod.h"
#include "core/formulas/kernels/ScalarRadialKernels.h"
#include "core/pre_headers.h"

namespace keops {

// Div-free and curl-free kernel with gaussian functions.
// k_df(x,y)b = exp(-r^2/s2)*(((d-1)/(2c)-r^2)b + <b,x-y>(x-y))
// k_cf(x,y)b = exp(-r^2/s2)*(       (1/(2c)) b   - <b,x-y>(x-y))
// The value of 1/s2 must be given as first parameter (_P<0,1>) when calling Eval()
// We do not use the previous template because exp(-r^2/s2) is factorized
// Matrix-valued kernels : implementations from Micheli/Glaunes paper

template<class C, class X, class Y, class B>
struct CurlFreeGaussKernel_helper {
  static_assert(C::DIM == 1, "First template argument must be a of dimension 1 for CurlFreeGaussKernel");
  static_assert(C::CAT == 2, "First template argument must be a parameter variable (CAT=2) for CurlFreeGaussKernel");
  static_assert(X::CAT != Y::CAT,
                "Second and third template arguments must not be of the same category for CurlFreeGaussKernel");
  static_assert(X::DIM == Y::DIM,
                "Second and third template arguments must have the same dimensions for CurlFreeGaussKernel");
  static_assert(Y::CAT == B::CAT,
                "Third and fourth template arguments must be of the same category for CurlFreeGaussKernel");
  static const int DIM = X::DIM;
  using R2 = SqDist<X, Y>;                                       // r2=|x-y|^2
  using XMY = Subtract<X, Y>;                                    // x-y
  using G = GaussFunction<R2, C>;                                // exp(-r^2/s2)
  using TWOC = Scal <IntConstant<2>, C>;                          // 2c
  using D1 = Divide <IntConstant<1>, TWOC>;                       // 1/(2c)
  using D2 = Scal<D1, B>;                                        // (1/(2c))b
  using BDOTXMY = Scalprod<B, XMY>;                              // <b,x-y>
  using D = Scal<BDOTXMY, XMY>;                                  // <b,x-y>(x-y)
  using type = Scal <G, Subtract<D2, D>>;                          // final formula
  using factorized_type = Factorize <Factorize<type, R2>, XMY>;    // formula, factorized by r2 and x-y
};

template<class C, class X, class Y, class B>
using CurlFreeGaussKernel = typename CurlFreeGaussKernel_helper<C, X, Y, B>::factorized_type;


#define CurlFreeGaussKernel(C,X,Y,B) KeopsNS<CurlFreeGaussKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()

}
