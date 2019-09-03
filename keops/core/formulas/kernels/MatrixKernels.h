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
#include "core/formulas/maths/Divide.h"

// import all operation on vector implementations
#include "core/formulas/norms/Scalprod.h"
#include "core/formulas/norms/SqDist.h"


namespace keops {




//////////////////////////////////////////////////////////////
////                 MATRIX-VALUED KERNELS                ////
//////////////////////////////////////////////////////////////

// Matrix-valued kernels : implementations from Micheli/Glaunes paper

// TRI kernel - general form :
// k(x,y)b = k_ortho(r2)b + k_tilde(r2)<b,x-y>(x-y),   where r2=|x-y|^2
// which gives the formula below.

// we construct the formula step by step (we use a struct just as a namespace, to avoid defining these temporary alias in the global scope)
template < template<class,class...> class FORTHO, template<class,class...> class FTILDE, class X, class Y, class B, class... PARAMS >
struct TRI_Kernel_helper {
  static_assert(X::CAT!=Y::CAT,"Second and third template arguments must not be of the same category for TRI_Kernel");
  static_assert(Y::CAT==B::CAT,"Third and fourth template arguments must be of the same category for TRI_Kernel");
  using R2 = SqDist<X,Y>;                                      // r2=|x-y|^2
  using KORTHOR2 = FORTHO<R2,PARAMS...>;                       // k_ortho(r2)
  using KTILDER2 = FTILDE<R2,PARAMS...>;                       // k_tilde(r2)
  using XMY = Subtract<X,Y>;                                   // x-y
  using BDOTXMY = Scalprod<B,XMY>;                             // <b,x-y>
  using D = Scalprod<BDOTXMY,XMY>;                             // <b,x-y>(x-y)
  using type = Add<Scal<KORTHOR2,B>,Scal<KTILDER2,D>>;         // final formula
  using factorized_type = Factorize<Factorize<type,R2>,XMY>;   // formula, factorized by r2 and x-y
};

// final definition is here
template < template<class,class...> class FORTHO, template<class,class...> class FTILDE, class X, class Y, class B, class... PARAMS >
using TRI_Kernel = typename TRI_Kernel_helper<FORTHO,FTILDE,X,Y,B,PARAMS...>::factorized_type;

// Div-free and curl-free kernel with gaussian functions.
// k_df(x,y)b = exp(-r^2/s2)*(((d-1)/(2c)-r^2)b + <b,x-y>(x-y))
// k_cf(x,y)b = exp(-r^2/s2)*(       (1/(2c)) b   - <b,x-y>(x-y))
// The value of 1/s2 must be given as first parameter (_P<0,1>) when calling Eval()
// We do not use the previous template because exp(-r^2/s2) is factorized

template < class C, class X, class Y, class B >
struct DivFreeGaussKernel_helper {
  static_assert(C::DIM==1,"First template argument must be a of dimension 1 for DivFreeGaussKernel");
  static_assert(C::CAT==2,"First template argument must be a parameter variable (CAT=2) for DivFreeGaussKernel");
  static_assert(X::CAT!=Y::CAT,"Second and third template arguments must not be of the same category for DivFreeGaussKernel");
  static_assert(X::DIM==Y::DIM,"Second and third template arguments must have the same dimensions for DivFreeGaussKernel");
  static_assert(Y::CAT==B::CAT,"Third and fourth template arguments must be of the same category for DivFreeGaussKernel");
  static const int DIM = X::DIM;
  using R2 = SqDist<X,Y>;                                       // r2=|x-y|^2
  using XMY = Subtract<X,Y>;                                    // x-y
  using G = GaussFunction<R2,C>;                                // exp(-r^2/s2)
  using TWOC = Scal<IntConstant<2>,C>;                          // 2c
  using D1 = Divide<IntConstant<DIM-1>,TWOC>;                   // (d-1)/(2c)
  using D2 = Scal<Subtract<D1,R2>,B>;                           // ((d-1)/(2c)-r^2)b
  using BDOTXMY = Scalprod<B,XMY>;                              // <b,x-y>
  using D = Scal<BDOTXMY,XMY>;                                  // <b,x-y>(x-y)
  using type = Scal<G,Add<D2,D>>;                               // final formula
  using factorized_type = Factorize<Factorize<type,R2>,XMY>;    // formula, factorized by r2 and x-y
};

template < class C, class X, class Y, class B >
using DivFreeGaussKernel = typename DivFreeGaussKernel_helper<C,X,Y,B>::factorized_type;

template < class C, class X, class Y, class B >
struct CurlFreeGaussKernel_helper {
  static_assert(C::DIM==1,"First template argument must be a of dimension 1 for CurlFreeGaussKernel");
  static_assert(C::CAT==2,"First template argument must be a parameter variable (CAT=2) for CurlFreeGaussKernel");
  static_assert(X::CAT!=Y::CAT,"Second and third template arguments must not be of the same category for CurlFreeGaussKernel");
  static_assert(X::DIM==Y::DIM,"Second and third template arguments must have the same dimensions for CurlFreeGaussKernel");
  static_assert(Y::CAT==B::CAT,"Third and fourth template arguments must be of the same category for CurlFreeGaussKernel");
  static const int DIM = X::DIM;
  using R2 = SqDist<X,Y>;                                       // r2=|x-y|^2
  using XMY = Subtract<X,Y>;                                    // x-y
  using G = GaussFunction<R2,C>;                                // exp(-r^2/s2)
  using TWOC = Scal<IntConstant<2>,C>;                          // 2c
  using D1 = Divide<IntConstant<1>,TWOC>;                       // 1/(2c)
  using D2 = Scal<D1,B>;                                        // (1/(2c))b
  using BDOTXMY = Scalprod<B,XMY>;                              // <b,x-y>
  using D = Scal<BDOTXMY,XMY>;                                  // <b,x-y>(x-y)
  using type = Scal<G,Subtract<D2,D>>;                          // final formula
  using factorized_type = Factorize<Factorize<type,R2>,XMY>;    // formula, factorized by r2 and x-y
};

template < class C, class X, class Y, class B >
using CurlFreeGaussKernel = typename CurlFreeGaussKernel_helper<C,X,Y,B>::factorized_type;

// Weighted combination of the two previous kernels, which gives a Translation and Rotation Invariant kernel with gaussian base function.
// k_tri(x,y)b = lambda * k_df(x,y)b + (1-lambda) * k_cf(x,y)b
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
}