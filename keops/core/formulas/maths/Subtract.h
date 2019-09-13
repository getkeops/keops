#pragma once

#include <assert.h>

#include "core/pack/CondType.h"
#include "core/autodiff/BinaryOp.h"
#include "core/formulas/constants/Zero.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Sum.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Minus.h"

#include "core/pre_headers.h"

namespace keops {

template < class FA, class FB >
struct Subtract_Impl;
template < class FA, class FB >
struct Subtract_Alias;
template < class FA, class FB >
using Subtract = typename Subtract_Alias< FA, FB >::type;

//////////////////////////////////////////////////////////////
////             SUBTRACT : F-G                           ////
//////////////////////////////////////////////////////////////


template < class FA, class FB >
struct Subtract_Impl : BinaryOp< Subtract_Impl, FA, FB > {
  // Output dim = FA::DIM = FB::DIM
  static const int DIM = FA::DIM;
  static_assert(DIM == FB::DIM, "Dimensions must be the same for Subtract");

  static void PrintIdString(::std::stringstream &str) {
    str << "-";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = outA[k] - outB[k];
  }

  // [\partial_V (A - B) ] . gradin = [\partial_V A ] . gradin  - [\partial_V B ] . gradin
  template < class V, class GRADIN >
  using DiffT = Subtract< typename FA::template DiffT< V, GRADIN >, typename FB::template DiffT< V, GRADIN > >;

};

template < class FA, class FB >
struct Subtract_Impl_Broadcast : BinaryOp< Subtract_Impl_Broadcast, FA, FB > {
  // Output dim = FB::DIM
  static const int DIM = FB::DIM;

  static void PrintIdString(::std::stringstream &str) {
    str << "-";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = *outA - outB[k];
  }

  // [\partial_V (A - B) ] . gradin = [\partial_V A ] . gradin  - [\partial_V B ] . gradin
  template < class V, class GRADIN >
  using DiffT = Subtract< typename FA::template DiffT< V, Sum < GRADIN>>, typename FB::template DiffT< V, GRADIN > >;

};

// Simplification rules

// third stage

// base class : this redirects to the implementation
template < class FA, class FB >
struct Subtract_Alias0 {
  using type1 = CondType< Subtract_Impl_Broadcast< FA, FB >, Subtract_Impl< FA, FB >, FA::DIM == 1 >;
  using type2 = CondType <Add_Impl_Broadcast< Minus < FB >, FA>, type1, FB::DIM == 1>;
  using type = CondType< Subtract_Impl< FA, FB >, type2, FA::DIM == FB::DIM >;
};

// A - A = 0
template < class F >
struct Subtract_Alias0< F, F > {
  using type = Zero< F::DIM >;
};

// A - B*A = (1-B)*A
template < class F, class G >
struct Subtract_Alias0< F, Scal_Impl < G, F>> {
using type = Scal <Subtract< IntConstant < 1 >, G>, F>;
};

// B*A - A = (-1+B)*A
template < class F, class G >
struct Subtract_Alias0< Scal_Impl < G, F >, F> {
using type = Scal <Add< IntConstant < -1 >, G>, F>;
};

// second stage

// base class : this redirects to third stage
template < class FA, class FB >
struct Subtract_Alias1 {
  using type = typename Subtract_Alias0< FA, FB >::type;
};

// B*A - C*A = (B-C)*A
template < class F, class G, class H >
struct Subtract_Alias1< Scal_Impl < G, F >, Scal_Impl <H, F>> {
using type = Scal <Subtract< G, H >, F>;
};

// A-n = -n+A (brings integers constants to the left)
template < int N, class F >
struct Subtract_Alias1< F, IntConstant_Impl < N>> {
using type = Add <IntConstant< -N >, F>;
};

// first stage

// base class, redirects to second stage
template < class FA, class FB >
struct Subtract_Alias {
  using type = typename Subtract_Alias1< FA, FB >::type;
};

// A - 0 = A
template < class FA, int DIM >
struct Subtract_Alias< FA, Zero < DIM>> {
static_assert(DIM == FA::DIM, "Dimensions must be the same for Subtract");
using type = FA;
};

// 0 - B = -B
template < class FB, int DIM >
struct Subtract_Alias< Zero < DIM >, FB> {
static_assert(DIM == FB::DIM, "Dimensions must be the same for Subtract");
using type = Minus< FB >;
};

// 0 - 0 = la tete a Toto
template < int DIM1, int DIM2 >
struct Subtract_Alias< Zero < DIM1 >, Zero <DIM2>> {
static_assert(DIM1 == DIM2, "Dimensions must be the same for Subtract");
using type = Zero< DIM1 >;
};

// m-n = m-n
template < int M, int N >
struct Subtract_Alias< IntConstant_Impl < M >, IntConstant_Impl <N>> {
using type = IntConstant< M - N >;
};

template < class FA, class FB >
KeopsNS <Subtract< FA, FB>> operator-(KeopsNS <FA> fa, KeopsNS <FB> fb) {
  return KeopsNS < Subtract < FA, FB >> ();
}
#define Subtract(fa, fb) KeopsNS<Subtract<decltype(InvKeopsNS(fa)),decltype(InvKeopsNS(fb))>>()

}
