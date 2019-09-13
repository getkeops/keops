#pragma once

#include "core/formulas/Factorize.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Subtract.h"
#include "core/formulas/norms/SqDist.h"
#include "core/formulas/norms/Scalprod.h"
#include "core/pre_headers.h"

namespace keops {


// TRI kernel - general form :
// k(x,y)b = k_ortho(r2)b + k_tilde(r2)<b,x-y>(x-y),   where r2=|x-y|^2
// which gives the formula below.

// we construct the formula step by step (we use a struct just as a namespace, to avoid defining these temporary alias in the global scope)
template<template<class, class...> class FORTHO,
    template<class, class...> class FTILDE, class X, class Y, class B, class... PARAMS>
struct TRI_Kernel_helper {
  static_assert(X::CAT != Y::CAT,
                "Second and third template arguments must not be of the same category for TRI_Kernel");
  static_assert(Y::CAT == B::CAT, "Third and fourth template arguments must be of the same category for TRI_Kernel");
  using R2 = SqDist<X, Y>;                                      // r2=|x-y|^2
  using KORTHOR2 = FORTHO<R2, PARAMS...>;                       // k_ortho(r2)
  using KTILDER2 = FTILDE<R2, PARAMS...>;                       // k_tilde(r2)
  using XMY = Subtract<X, Y>;                                   // x-y
  using BDOTXMY = Scalprod<B, XMY>;                             // <b,x-y>
  using D = Scalprod<BDOTXMY, XMY>;                             // <b,x-y>(x-y)
  using type = Add<Scal<KORTHOR2, B>, Scal<KTILDER2, D>>;         // final formula
  using factorized_type = Factorize<Factorize<type, R2>, XMY>;   // formula, factorized by r2 and x-y
};

// final definition is here
template<template<class, class...> class FORTHO,
    template<class, class...> class FTILDE, class X, class Y, class B, class... PARAMS>
using TRI_Kernel = typename TRI_Kernel_helper<FORTHO, FTILDE, X, Y, B, PARAMS...>::factorized_type;


}